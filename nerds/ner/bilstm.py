from pathlib import Path

import anago

from nerds.config.base import BiLSTMModelConfiguration
from nerds.ner.base import NamedEntityRecognitionModel
from nerds.doc.bio import transform_annotated_documents_to_bio_format, transform_bio_tags_to_annotated_document
from nerds.util.file import mkdir
from nerds.util.logging import get_logger, log_progress
from nerds.util.nlp import text_to_tokens

log = get_logger()

KEY = "bilstm_ner"


class BidirectionalLSTM(NamedEntityRecognitionModel):
    def __init__(self, entity_labels=None):
        super().__init__(entity_labels)
        self.key = KEY
        self.config = BiLSTMModelConfiguration()
        if self.entity_labels:
            self.config.set_parameter("entity_labels", self.entity_labels)

    def fit(self,
            X,
            y=None,
            char_emb_size=32,
            word_emb_size=128,
            char_lstm_units=32,
            word_lstm_units=128,
            dropout=0.5,
            batch_size=8,
            num_epochs=10):
        """ Trains the NER model. The input is a list of
            `AnnotatedDocument` instances.

            We should be careful with batch size:
            it must satisfy len(X) % batch_size == 0.
            Otherwise, Anago crushes with an error from time to time.
            An example here is a token assigned a tag (the BIO scheme).
        """

        log.info("Checking parameters...")
        self.config.set_parameters({
            "num_epochs": num_epochs,
            "dropout": dropout,
            "batch_size": batch_size,
            "char_emb_size": char_emb_size,
            "word_emb_size": word_emb_size,
            "char_lstm_units": char_lstm_units,
            "word_lstm_units": word_lstm_units
        })
        self.config.validate()

        # Anago splits the BIO tags on the dash "-", so if the label contains
        # a dash, it corrupts it. This is a workaround for this behavior.
        label_map = {}
        for annotated_document in X:
            for annotation in annotated_document.annotations:
                if "-" in annotation.label:
                    label_map[annotation.label.split("-")[-1]] = annotation.label
                else:
                    label_map["B_" + annotation.label] = annotation.label
                    label_map["I_" + annotation.label] = annotation.label
        self.config.set_parameter("label_map", label_map)

        self.model = anago.Sequence(
            char_embedding_dim=self.config.get_parameter("char_emb_size"),
            word_embedding_dim=self.config.get_parameter("word_emb_size"),
            char_lstm_size=self.config.get_parameter("char_lstm_units"),
            word_lstm_size=self.config.get_parameter("word_lstm_units"),
            dropout=self.config.get_parameter("dropout"))

        log.info("Transforming {} items to BIO format...".format(len(X)))
        train_data = transform_annotated_documents_to_bio_format(X, entity_labels=self.entity_labels)

        # new version does not use numpy arrays as arguments
        # BIO_X = np.asarray([x_i for x_i in training_data[0] if len(x_i) > 0])
        # BIO_y = np.asarray([y_i for y_i in training_data[1] if len(y_i) > 0])

        # validation is not necessary as we normally use Optimizer
        # X_train, X_valid, y_train, y_valid = train_test_split(
        # 	BIO_X, BIO_y, test_size=0.1)

        X_train = [x_i for x_i in train_data[0]]
        y_train = [y_i for y_i in train_data[1]]

        # check sizes
        if len(X_train) != len(y_train):
            log.error("Got {} feature vectors but {} labels, cannot train!".format(len(X_train), len(y_train)))
            return self

        # number of examples must be divisible by batch_size,
        # so skip examples in the end if needed
        exm_num = len(X_train)
        X_train = X_train[:exm_num - exm_num % batch_size]
        y_train = y_train[:exm_num - exm_num % batch_size]

        log.info("Training BiLSTM...")
        self.model.fit(
            X_train,
            y_train,
            epochs=self.config.get_parameter("num_epochs"),
            batch_size=self.config.get_parameter("batch_size"))
        return self

    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
            input and returns a list of `AnnotatedDocument` objects.
        """
        log.info("Annotating named entities in {} documents with BiLSTM...".format(len(X)))
        annotated_documents = []
        for idx, document in enumerate(X):
            # changed signature of analyze(): removed "content = sentence_to_tokens(document.plain_text_)"
            # the previous code to find offsets (function "_get_offsets_with_fuzzy_matching") is no longer needed
            # process the output of the tagger
            output = self.model.analyze(document.plain_text_, tokenizer=text_to_tokens)
            tokens = output["words"]
            tags = []
            scores = []
            prev_idx = 0
            for entity in output["entities"]:
                # check assigned types
                # ignore auxiliary types such as "<pad>"
                if entity["type"] not in self.config.get_parameter("label_map"):
                    continue

                tags += ["O" for _ in range(prev_idx, entity["beginOffset"])]
                scores += [1.0 for _ in range(prev_idx, entity["beginOffset"])]

                tags += [entity["type"] for _ in range(entity["beginOffset"], entity["endOffset"])]
                scores += [entity["score"] for _ in range(entity["beginOffset"], entity["endOffset"])]

                prev_idx = entity["endOffset"]

            # add remaining tags
            tags += ["O" for _ in range(prev_idx, len(tokens))]
            scores += [1.0 for _ in range(prev_idx, len(tokens))]

            if len(tokens) != len(tags):
                log.error("Number of tokens differs from the number of tags: {} != {}".format(len(tokens), len(tags)))

            # this only gives tags with maximal scores
            tag_scores = []
            for i, tag in enumerate(tags):
                tag_scores += [{tag: scores[i]}]

            annotated_documents.append(transform_bio_tags_to_annotated_document(tokens, tag_scores, document))
            # info
            log_progress(log, idx, len(X))

        return annotated_documents

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        save_path = Path(file_path)
        mkdir(save_path)
        model_save_path = save_path.joinpath("BiLSTM.model")
        mkdir(model_save_path)
        config_save_path = save_path.joinpath("BiLSTM.config")
        weights_save_path = model_save_path.joinpath("weights.h5")
        params_save_path = model_save_path.joinpath("params.json")
        preproc_save_path = model_save_path.joinpath("preprocessor.pickle")
        self.model.save(weights_save_path, params_save_path, preproc_save_path)
        self.config.save(config_save_path)

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model_load_path = load_path.joinpath("BiLSTM.model")
        config_load_path = load_path.joinpath("BiLSTM.config")
        weights_load_path = model_load_path.joinpath("weights.h5")
        params_load_path = model_load_path.joinpath("params.json")
        preproc_load_path = model_load_path.joinpath("preprocessor.pickle")
        self.model = anago.Sequence.load(weights_load_path, params_load_path, preproc_load_path)
        self.config.load(config_load_path)
        return self


def _get_offsets_with_fuzzy_matching(haystack, needle, offset_init=0):
    """ Private function for internal use in the BiLSTM implementation.

        The entities we get in the predicted text are space-separated
        tokens e.g. "anti - HIV", although in the original text the space
        may not be there e.g. "anti-HIV". If such matching occurs, this
        function will return the appropriate offsets.
    """
    search_idx = offset_init
    start_idx = None
    end_idx = None
    tokens = needle.split()

    # If this isn't a multi-term annotation, no need for fancy matchings.
    if len(tokens) == 1:
        start_idx = haystack.find(tokens[0], search_idx)
        end_idx = start_idx + len(tokens[0])
        return start_idx, end_idx

    # Otherwise we iterate the tokens 2-by-2 until we get a full match.
    token_idx = 1
    cur_idx, cur_token = -1, ""
    while token_idx < len(tokens):
        prv_token = tokens[token_idx - 1]
        prv_idx = haystack.find(prv_token, search_idx)
        prv_offset = prv_idx + len(prv_token)
        cur_token = tokens[token_idx]
        cur_idx = haystack.find(cur_token, prv_offset)
        # In every iteration we check if the next word starts from where
        # the previous ends.
        if cur_idx in (prv_offset, prv_offset + 1):
            if start_idx is None:
                start_idx = prv_idx
            token_idx += 1
        # If it doesn't, then it must be an accidental match, reset the index.
        else:
            start_idx = None
            token_idx = 1
        search_idx = prv_idx + len(prv_token)
    end_idx = cur_idx + len(cur_token)

    return start_idx, end_idx
