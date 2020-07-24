from pathlib import Path

import numpy as np
import random
import json

from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from keras.layers import Bidirectional
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Concatenate
from keras.models import Model, Input, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_contrib.utils import save_load_utils

from nerds.config.base import BiLSTMModelConfiguration
from nerds.doc.bio import transform_annotated_documents_to_bio_format, transform_bio_tags_to_annotated_document, \
    transform_annotated_document_to_bio_format
from nerds.ner.base import NamedEntityRecognitionModel
from nerds.util.file import mkdir
from nerds.util.logging import get_logger, log_progress
from nerds.util.nlp import tokens_to_pos_tags

log = get_logger()

KEY = "keras_ner"

UNKNOWN_WORD = "UNKNOWN_WORD"
PAD_WORD = "PAD_WORD"
UNKNOWN_TAG = "UNKNOWN_TAG"
PAD_TAG = "PAD_TAG"
UNKNOWN_CHAR = "UNKNOWN_CHAR"
PAD_CHAR = "UNKNOWN_CHAR"
UNKNOWN_POS = "UNKNOWN_POS"
PAD_POS = "PAD_POS"


class DataEncoder(object):

    def __init__(self, word_embeddings=None):
        self.params = dict()
        self.params["use_word_emb"] = True if word_embeddings else False
        self.word_embeddings = word_embeddings

    def fit(self, X, Y, use_chars=False, pos_tags=None):
        self.params["use_chars"] = use_chars
        self.params["use_pos_tags"] = True if pos_tags else False
        # measure maximal length and make vocabularies
        max_len, max_len_char = 0, 0
        word_vocab, char_vocab, tag_vocab, pos_vocab = set(), set(), set(), set()
        for y in Y:
            tag_vocab.update(y)
        for idx, x in enumerate(X):
            if len(x) > max_len:
                max_len = len(x)
            if not self.word_embeddings:
                word_vocab.update(x)
            if self.params["use_chars"]:
                for word in x:
                    chars = [c for c in word]
                    char_vocab.update(chars)
                    if len(chars) > max_len_char:
                        max_len_char = len(chars)
            if pos_tags:
                pos_vocab.update(pos_tags[idx])
        self.params["max_len"] = max_len
        self.params["max_len_char"] = max_len_char
        # make dictionaries
        # tags
        self.tag2idx = {t: i + 1 for i, t in enumerate(tag_vocab)}
        self.tag2idx[UNKNOWN_TAG] = len(tag_vocab) + 1
        self.tag2idx[PAD_TAG] = 0
        self.idx2tag = {i: t for t, i in self.tag2idx.items()}
        # words
        if not self.word_embeddings:
            self.word2idx = {w: i + 1 for i, w in enumerate(word_vocab)}
            self.word2idx[UNKNOWN_WORD] = len(word_vocab) + 1
            self.word2idx[PAD_WORD] = 0
            self.idx2word = {i: w for w, i in self.word2idx.items()}
        else:
            self.word2idx = {w: i + 1 for i, w in enumerate(self.word_embeddings.index2word)}
            self.word2idx[UNKNOWN_WORD] = len(self.word_embeddings.index2word) + 1
            self.word2idx[PAD_WORD] = 0
            self.idx2word = {i: w for w, i in self.word2idx.items()}
            factor = 0.01
            self.word_vectors = np.array(
                [[random.uniform(-1, 1) * factor for _ in range(self.word_embeddings.vector_size)]] +
                [self.word_embeddings[self.idx2word[idx]] for idx in self.idx2word
                 if idx != self.word2idx[PAD_WORD] and idx != self.word2idx[UNKNOWN_WORD]] +
                [[random.uniform(-1, 1) * factor for _ in range(self.word_embeddings.vector_size)]])
        # chars
        if self.params["use_chars"]:
            self.char2idx = {c: i + 1 for i, c in enumerate(char_vocab)}
            self.char2idx[UNKNOWN_CHAR] = len(char_vocab) + 1
            self.char2idx[PAD_CHAR] = 0
            self.idx2char = {i: c for c, i in self.char2idx.items()}
        # POS tags
        if pos_tags:
            self.pos2idx = {p: i + 1 for i, p in enumerate(pos_vocab)}
            self.pos2idx[UNKNOWN_POS] = len(pos_vocab) + 1
            self.pos2idx[PAD_POS] = 0
            self.idx2pos = {i: p for p, i in self.pos2idx.items()}

    def save(self, file_path):
        save_path = Path(file_path)
        mkdir(save_path)
        param_path = Path(save_path.joinpath("params.json"))
        with open(param_path, "w") as fp:
            fp.write(json.dumps(self.params))
        word_index_path = Path(save_path.joinpath("word2idx.json"))
        with open(word_index_path, "w") as fp:
            fp.write(json.dumps(self.word2idx))
        tag_index_path = Path(save_path.joinpath("tag2idx.json"))
        with open(tag_index_path, "w") as fp:
            fp.write(json.dumps(self.tag2idx))
        if self.params["use_chars"]:
            char_index_path = Path(save_path.joinpath("char2idx.json"))
            with open(char_index_path, "w") as fp:
                fp.write(json.dumps(self.char2idx))
        if self.params["use_pos_tags"]:
            pos_index_path = Path(save_path.joinpath("pos2idx.json"))
            with open(pos_index_path, "w") as fp:
                fp.write(json.dumps(self.pos2idx))
        if self.params["use_word_emb"]:
            word_emb_path = Path(save_path.joinpath("word_emb.csv"))
            np.savetxt(fname=str(word_emb_path), X=self.word_vectors, delimiter=",")

    def load(self, file_path):
        load_path = Path(file_path)
        param_path = Path(load_path.joinpath("params.json"))
        with open(param_path, "r") as fp:
            self.params = json.loads(fp.read().strip())
        word_index_path = Path(load_path.joinpath("word2idx.json"))
        with open(word_index_path, "r") as fp:
            self.word2idx = json.loads(fp.read().strip())
            self.idx2word = {i: w for w, i in self.word2idx.items()}
        tag_index_path = Path(load_path.joinpath("tag2idx.json"))
        with open(tag_index_path, "r") as fp:
            self.tag2idx = json.loads(fp.read().strip())
            self.idx2tag = {i: t for t, i in self.tag2idx.items()}
        if self.params["use_chars"]:
            char_index_path = Path(load_path.joinpath("char2idx.json"))
            with open(char_index_path, "r") as fp:
                self.char2idx = json.loads(fp.read().strip())
                self.idx2char = {i: c for c, i in self.char2idx.items()}
        if self.params["use_pos_tags"]:
            pos_index_path = Path(load_path.joinpath("pos2idx.json"))
            with open(pos_index_path, "r") as fp:
                self.pos2idx = json.loads(fp.read().strip())
                self.idx2pos = {i: p for p, i in self.pos2idx.items()}
        if self.params["use_word_emb"]:
            word_emb_path = Path(load_path.joinpath("word_emb.csv"))
            self.word_vectors = np.loadtxt(fname=str(word_emb_path), delimiter=",")
        return self

    def encode_word(self, word):
        if word not in self.word2idx:
            return self.word2idx[UNKNOWN_WORD]
        return self.word2idx[word]

    def encode_tag(self, tag):
        if tag not in self.tag2idx:
            return self.tag2idx[UNKNOWN_TAG]
        return self.tag2idx[tag]

    def encode_char(self, char):
        if char not in self.char2idx:
            return self.char2idx[UNKNOWN_CHAR]
        return self.char2idx[char]

    def encode_pos(self, pos):
        if pos not in self.pos2idx:
            return self.pos2idx[UNKNOWN_POS]
        return self.pos2idx[pos]

    def decode_word(self, idx):
        if idx not in self.idx2word:
            return UNKNOWN_WORD
        return self.idx2word[idx]

    def decode_tag(self, idx):
        if idx not in self.idx2tag:
            return UNKNOWN_TAG
        return self.idx2tag[idx]

    def decode_char(self, idx):
        if idx not in self.idx2char:
            return UNKNOWN_CHAR
        return self.idx2char[idx]

    def decode_pos(self, idx):
        if idx not in self.idx2pos:
            return UNKNOWN_POS
        return self.idx2pos[idx]

    @property
    def max_len(self):
        return self.params["max_len"]

    @property
    def max_len_char(self):
        return self.params["max_len_char"]

    @property
    def word_count(self):
        return len(self.word2idx)

    @property
    def tag_count(self):
        return len(self.tag2idx)

    @property
    def char_count(self):
        return len(self.char2idx) if self.params["use_chars"] else 0

    @property
    def pos_count(self):
        return len(self.pos2idx) if self.params["use_pos_tags"] else 0


class KerasNERModel(NamedEntityRecognitionModel):
    def __init__(self, entity_labels=None, word_embeddings=None):
        super().__init__(entity_labels)
        self.key = KEY
        self.config = BiLSTMModelConfiguration()
        if self.entity_labels:
            self.config.set_parameter("entity_labels", self.entity_labels)
        self.encoder = None
        if word_embeddings and not isinstance(word_embeddings, WordEmbeddingsKeyedVectors):
            raise TypeError("Input word embeddings must be of type gensim.models.KeyedVectors")
        self.word_embeddings = word_embeddings
        self.encoder = DataEncoder(word_embeddings=self.word_embeddings)

    def fit(self,
            X,
            y=None,
            char_emb_size=32,
            word_emb_size=128,
            char_lstm_units=32,
            word_lstm_units=128,
            pos_emb_size=16,
            dropout=0.5,
            batch_size=8,
            num_epochs=10,
            use_crf=False,
            use_char_emb=False,
            shuffle=False,
            use_pos_emb=False):
        """ Trains the NER model. The input is a list of
            `AnnotatedDocument` instances.

            We should be careful with batch size:
            it must satisfy len(X) % batch_size == 0.
            Otherwise, it crushes with an error from time to time.
            An example here is a token assigned a tag (the BIO scheme).
        """

        log.info("Checking parameters...")
        self.config.set_parameters({
            "num_epochs": num_epochs,
            "dropout": dropout,
            "batch_size": batch_size,
            "char_emb_size": char_emb_size,
            "word_emb_size": self.word_embeddings.vector_size if self.word_embeddings else word_emb_size,
            "char_lstm_units": char_lstm_units,
            "word_lstm_units": word_lstm_units,
            "pos_emb_size": pos_emb_size,
            "use_crf": use_crf,
            "use_char_emb": use_char_emb,
            "shuffle": shuffle,
            "use_pos_emb": use_pos_emb
        })
        self.config.validate()

        log.info("Transforming {} items to BIO format...".format(len(X)))
        X_train, Y_train = self._transform_to_bio(X)

        pos_tags = []
        if use_pos_emb:
            log.info("Getting POS tags for {} items...".format(len(X)))
            for idx, x in enumerate(X_train):
                pos_tags.append(tokens_to_pos_tags(x))
                log_progress(log, idx, len(X))

        # fit encoder
        self.encoder.fit(X=X_train, Y=Y_train,
                         use_chars=self.config.get_parameter("use_char_emb"),
                         pos_tags=pos_tags)

        # compile the model architecture
        self._compile_model()

        # encode and pad word sequences
        X = [[self.encoder.encode_word(word) for word in x] for x in X_train]
        X = pad_sequences(maxlen=self.encoder.max_len,
                          sequences=X,
                          padding="post",
                          value=self.encoder.encode_word(PAD_WORD))

        # add X to inputs
        inputs = [X]

        # encode and pad tag sequences
        Y = [[self.encoder.encode_tag(tag) for tag in y] for y in Y_train]
        Y = pad_sequences(maxlen=self.encoder.max_len,
                          sequences=Y,
                          padding="post",
                          value=self.encoder.encode_tag(PAD_TAG))
        Y = np.array([to_categorical(y, num_classes=self.encoder.tag_count) for y in Y])

        # encode and pad character sequences if needed
        if self.config.get_parameter("use_char_emb"):
            C = []
            for x in X_train:
                c = [[self.encoder.encode_char(char) for char in word] for word in x]
                c = pad_sequences(maxlen=self.encoder.max_len_char,
                                  sequences=c,
                                  padding="post",
                                  value=self.encoder.encode_char(PAD_CHAR)).tolist()
                # add padding chars for padding words
                for i in range(len(x), self.encoder.max_len):
                    c.append([self.encoder.encode_char(PAD_CHAR)] * self.encoder.max_len_char)

                C.append(c)

            C = np.array(C, ndmin=3)
            inputs.append(C)

        # encode and pad POS tag sequences if needed
        if self.config.get_parameter("use_pos_emb"):
            P = [[self.encoder.encode_pos(pos) for pos in pos_seq] for pos_seq in pos_tags]
            P = pad_sequences(maxlen=self.encoder.max_len,
                              sequences=P,
                              padding="post",
                              value=self.encoder.encode_pos(PAD_POS))
            inputs.append(P)

        # train model
        log.info("Training BiLSTM...")
        self.model.fit(
            x=inputs,
            y=Y,
            epochs=self.config.get_parameter("num_epochs"),
            batch_size=self.config.get_parameter("batch_size"),
            verbose=1,
            shuffle=self.config.get_parameter("shuffle"))
        return self

    def _transform_to_bio(self, X):
        train_data = transform_annotated_documents_to_bio_format(X, entity_labels=self.entity_labels)

        X_train = [x_i for x_i in train_data[0]]
        y_train = [y_i for y_i in train_data[1]]

        # check sizes
        if len(X_train) != len(y_train):
            log.error("Got {} feature vectors but {} labels, cannot train!".format(len(X_train), len(y_train)))
            return self

        # number of examples must be divisible by batch_size,
        # so skip examples in the end if needed
        exm_num = len(X_train)
        X_train = X_train[:exm_num - exm_num % self.config.get_parameter("batch_size")]
        y_train = y_train[:exm_num - exm_num % self.config.get_parameter("batch_size")]

        return X_train, y_train

    def _compile_model(self):
        """ Defines the model architecture and compiles it """
        # input layer for words
        word_input = Input(shape=(self.encoder.max_len,))
        # embedding for words
        if self.word_embeddings:
            word_embeddings = Embedding(input_dim=self.encoder.word_count,
                                        output_dim=self.config.get_parameter("word_emb_size"),
                                        input_length=self.encoder.max_len,
                                        weights=[self.encoder.word_vectors],
                                        mask_zero=True)(word_input)
        else:
            word_embeddings = Embedding(input_dim=self.encoder.word_count,
                                        output_dim=self.config.get_parameter("word_emb_size"),
                                        input_length=self.encoder.max_len,
                                        mask_zero=True)(word_input)
        # add word input
        inputs = [word_input]

        # use character embeddings if needed
        if self.config.get_parameter("use_char_emb"):
            # input layer for characters
            char_input = Input(shape=(self.encoder.max_len, self.encoder.max_len_char,))
            # embedding for characters
            char_embeddings = TimeDistributed(Embedding(input_dim=self.encoder.char_count,
                                                        output_dim=self.config.get_parameter("char_emb_size"),
                                                        input_length=self.encoder.max_len_char,
                                                        mask_zero=True))(char_input)

            # character LSTM to get word encodings by characters
            char_encodings = TimeDistributed(Bidirectional(LSTM(units=self.config.get_parameter("char_lstm_units"),
                                                                return_sequences=False,
                                                                recurrent_dropout=self.config.get_parameter(
                                                                    "dropout"))))(char_embeddings)

            # concatenate word and character embeddings
            word_embeddings = Concatenate()([word_embeddings, char_encodings])

            # add character input
            inputs.append(char_input)

        # use POS tags if needed
        if self.config.get_parameter("use_pos_emb"):
            # input layer for POS tags
            pos_input = Input(shape=(self.encoder.max_len,))
            # embedding for POS tags
            pos_embeddings = Embedding(input_dim=self.encoder.pos_count,
                                       output_dim=self.config.get_parameter("pos_emb_size"),
                                       input_length=self.encoder.max_len,
                                       mask_zero=True)(pos_input)

            # concatenate word and POS tag embeddings
            word_embeddings = Concatenate()([word_embeddings, pos_embeddings])

            # add POS tag input
            inputs.append(pos_input)

        # add dropout
        model = Dropout(self.config.get_parameter("dropout"))(word_embeddings)

        # add main LSTM
        model = Bidirectional(LSTM(units=self.config.get_parameter("word_lstm_units"),
                                   return_sequences=True,
                                   recurrent_dropout=self.config.get_parameter("dropout")))(model)

        # add CRF if needed
        if self.config.get_parameter("use_crf"):
            model = TimeDistributed(Dense(units=self.config.get_parameter("word_lstm_units"),
                                          activation="relu"))(model)
            crf = CRF(self.encoder.tag_count)
            output = crf(model)
            self.model = Model(inputs, output)
            self.model.compile(optimizer="adam", loss=crf_loss, metrics=[crf_accuracy])
        else:
            output = TimeDistributed(Dense(units=self.encoder.tag_count, activation="softmax"))(model)
            self.model = Model(inputs, output)
            self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # noinspection PyTypeChecker
    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
            input and returns a list of `AnnotatedDocument` objects.
        """
        log.info("Annotating named entities in {} documents with BiLSTM...".format(len(X)))
        annotated_documents = []
        for idx, document in enumerate(X):

            # get tokens
            tokens, _ = transform_annotated_document_to_bio_format(document)

            # encode tokens and pad the sequence
            coded_tokens = [self.encoder.encode_word(token) for token in tokens]
            x = pad_sequences(maxlen=self.encoder.max_len,
                              sequences=[coded_tokens],
                              padding="post",
                              value=self.encoder.encode_word(PAD_WORD))
            inputs = [x]

            # add encoded and padded char sequences if needed
            if self.config.get_parameter("use_char_emb"):
                c = [[self.encoder.encode_char(char) for char in token] for token in tokens]
                c = pad_sequences(maxlen=self.encoder.max_len_char,
                                  sequences=c,
                                  padding="post",
                                  value=self.encoder.encode_char(PAD_CHAR)).tolist()
                # add padding chars for padding words
                for i in range(len(tokens), self.encoder.max_len):
                    c.append([self.encoder.encode_char(PAD_CHAR)] * self.encoder.max_len_char)
                c = np.array([c], ndmin=3)
                inputs.append(c)

            # add encoded and padded POS tag sequences if needed
            if self.config.get_parameter("use_pos_emb"):
                pos_tags = tokens_to_pos_tags(tokens)
                coded_pos_tags = [self.encoder.encode_pos(pos) for pos in pos_tags]
                p = pad_sequences(maxlen=self.encoder.max_len,
                                  sequences=[coded_pos_tags],
                                  padding="post",
                                  value=self.encoder.encode_pos(PAD_POS))
                inputs.append(p)

            # get predicted tags
            output = self.model.predict(x=inputs)
            coded_tags = np.argmax(output, axis=-1)[0]
            tags = [self.encoder.decode_tag(idx) for idx in coded_tags]
            tags = tags[: len(tokens)]

            # annotate a document
            annotated_documents.append(transform_bio_tags_to_annotated_document(tokens, tags, document))
            # info
            log_progress(log, idx, len(X))

        return annotated_documents

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        save_path = Path(file_path)
        mkdir(save_path)
        model_save_path = save_path.joinpath("KerasNER.model")
        config_save_path = save_path.joinpath("KerasNER.config")
        arch_save_path = save_path.joinpath("KerasNER.json")
        encoder_save_path = save_path.joinpath("encoder")
        if self.config.get_parameter("use_crf"):
            save_load_utils.save_all_weights(self.model, str(model_save_path))
        else:
            self.model.save(str(model_save_path))
        self.config.save(config_save_path)
        # human-readable model architecture in json
        with open(arch_save_path, "w") as wf:
            wf.write(self.model.to_json())
        self.encoder.save(encoder_save_path)

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model_load_path = load_path.joinpath("KerasNER.model")
        config_load_path = load_path.joinpath("KerasNER.config")
        encoder_load_path = load_path.joinpath("encoder")
        self.config.load(config_load_path)
        if self.config.get_parameter("use_crf"):
            save_load_utils.load_all_weights(self.model, str(model_load_path))
        else:
            self.model = load_model(str(model_load_path))
        self.encoder.load(encoder_load_path)
        return self
