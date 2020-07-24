from pathlib import Path

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

from nerds.doc.bio import transform_annotated_documents_to_bio_format
from nerds.features.base import UNKNOWN_WORD
from nerds.features.fword import WordFeatureExtractor
from nerds.util.file import mkdir
from nerds.util.logging import get_logger
from nerds.util.nlp import tokens_to_pos_tags

log = get_logger()

KEY = "fcontext"


class WordContextFeatureExtractor(WordFeatureExtractor):
    def __init__(self, ngram_slice=1, window=1):
        super().__init__(ngram_slice=ngram_slice)
        self.key = KEY
        self.config.set_parameter("window", window)

    def transform(self, X, y=None, entity_labels=None):
        """ Transforms the list of `Document` objects that are provided as
            input to the BIO format and returns features, tokens, and labels per word.
            Features of a word stored as a list of values.
        """
        log.info("Generating features for {} documents...".format(len(X)))
        tokens_per_doc, labels_per_doc = \
            transform_annotated_documents_to_bio_format(X, entity_labels=entity_labels)
        tokens_flat = [token for tokens in tokens_per_doc for token in tokens]
        labels_flat = [label for labels in labels_per_doc for label in labels]
        pos_tags_flat = [pos_tag for tokens in tokens_per_doc for pos_tag in tokens_to_pos_tags(tokens)]

        features_flat = [self._word_to_context(tokens_flat, pos_tags_flat, idx)
                         for idx in range(len(tokens_flat))]

        if not self.encoders:
            # first time run
            for idx in range(len(features_flat[0])):
                if isinstance(features_flat[0][idx], str):
                    self.encoders[idx] = LabelEncoder()
                    column_vector = [features_flat[i][idx] for i in range(len(features_flat))]
                    column_vector.append(UNKNOWN_WORD)
                    self.encoders[idx].fit(column_vector)

        for idx in self.encoders:
            column_vector = [features_flat[i][idx] for i in range(len(features_flat))]
            self._process_unknown_values(column_vector, self.encoders[idx].classes_.tolist(), UNKNOWN_WORD)
            column_vector = self.encoders[idx].transform(column_vector).tolist()
            for i in range(len(features_flat)):
                features_flat[i][idx] = column_vector[i]

        return features_flat, tokens_flat, labels_flat

    def _word_to_context(self, words, pos_tags, index):
        """ Extracts features from the context of a given word.
        """
        features = []
        window = self.config.get_parameter("window")
        for idx in range(index - window, index + window + 1):
            if idx < 0:
                idx_adj = 0
            elif idx >= len(words):
                idx_adj = len(words) - 1
            else:
                idx_adj = idx
            features.extend(self._word_to_features(words[idx_adj]))
            features.append(pos_tags[idx_adj])
        return features

    def save(self, file_path):
        save_path = Path(file_path)
        mkdir(save_path)
        # save config
        config_save_path = save_path.joinpath("context.config")
        self.config.save(config_save_path)
        # save dictionaries
        for i in self.encoders:
            words_path = save_path.joinpath("f{}.dict".format(i))
            joblib.dump(self.encoders[i], words_path)

    def load(self, file_path):
        load_path = Path(file_path)
        # load config
        config_load_path = load_path.joinpath("context.config")
        self.config.load(config_load_path)
        # load dictionaries
        for dict_file in load_path.glob("*.dict"):
            idx = int(dict_file.name.split(".")[0][1:])
            self.encoders[idx] = joblib.load(dict_file)
        return self
