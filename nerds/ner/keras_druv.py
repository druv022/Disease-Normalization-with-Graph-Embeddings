from pathlib import Path

import numpy as np
import random
import json
import keras as K
from keras import backend as Kb

from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from keras.layers import Bidirectional
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Concatenate,Activation, Conv1D, GlobalMaxPooling1D
from keras.models import Model, Input, model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_contrib.utils import save_load_utils
from keras.optimizers import Adam

from nerds.config.base import BiLSTMModelConfiguration, BiLSTMModelConfigurationModified
from nerds.doc.bio import transform_annotated_documents_to_bio_format, transform_bio_tags_to_annotated_document, \
    transform_annotated_document_to_bio_format
from nerds.ner.base import NamedEntityRecognitionModel
from nerds.util.file import mkdir
from nerds.util.logging import get_logger, log_progress
from nerds.util.nlp import tokens_to_pos_tags
import tensorflow as tf
import os, pickle, math
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from callbacks import F1score


log = get_logger()

KEY = "keras_ner"

# from anago
class Vocabulary(object):
    """A vocabulary that maps tokens to ints (storing a vocabulary).

    Attributes:
        _token_count: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocabulary.
        _token2id: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        _id2token: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, max_size=None, lower=True, unk_token=True, specials=('<pad>',)):
        """Create a Vocabulary object.

        Args:
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            lower: boolean. Whether to convert the texts to lowercase.
            unk_token: boolean. Whether to add unknown token.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary. Default: ('<pad>',)
        """
        self._max_size = max_size
        self._lower = lower
        self._unk = unk_token
        self._token2id = {token: i for i, token in enumerate(specials)}
        self._id2token = list(specials)
        self._token_count = Counter()

    def __len__(self):
        return len(self._token2id)

    def add_token(self, token):
        """Add token to vocabulary.

        Args:
            token (str): token to add.
        """
        token = self.process_token(token)
        self._token_count.update([token])

    def add_documents(self, docs):
        """Update dictionary from a collection of documents. Each document is a list
        of tokens.

        Args:
            docs (list): documents to add.
        """
        for sent in docs:
            sent = map(self.process_token, sent)
            self._token_count.update(sent)

    def doc2id(self, doc):
        """Get the list of token_id given doc.

        Args:
            doc (list): document.

        Returns:
            list: int id of doc.
        """
        doc = map(self.process_token, doc)
        return [self.token_to_id(token) for token in doc]

    def id2doc(self, ids):
        """Get the token list.

        Args:
            ids (list): token ids.

        Returns:
            list: token list.
        """
        return [self.id_to_token(idx) for idx in ids]

    def build(self):
        """
        Build vocabulary.
        """
        token_freq = self._token_count.most_common(self._max_size)
        idx = len(self.vocab)
        for token, _ in token_freq:
            self._token2id[token] = idx
            self._id2token.append(token)
            idx += 1
        if self._unk:
            unk = '<unk>'
            self._token2id[unk] = idx
            self._id2token.append(unk)

    def process_token(self, token):
        """Process token before following methods:
        * add_token
        * add_documents
        * doc2id
        * token_to_id

        Args:
            token (str): token to process.

        Returns:
            str: processed token string.
        """
        if self._lower:
            token = token.lower()

        return token

    def token_to_id(self, token):
        """Get the token_id of given token.

        Args:
            token (str): token from vocabulary.

        Returns:
            int: int id of token.
        """
        token = self.process_token(token)
        return self._token2id.get(token, len(self._token2id) - 1)

    def id_to_token(self, idx):
        """token-id to token (string).

        Args:
            idx (int): token id.

        Returns:
            str: string of given token id.
        """
        return self._id2token[idx]

    @property
    def vocab(self):
        """Return the vocabulary.

        Returns:
            dict: get the dict object of the vocabulary.
        """
        return self._token2id

    @property
    def reverse_vocab(self):
        """Return the vocabulary as a reversed dict object.

        Returns:
            dict: reversed vocabulary object.
        """
        return self._id2token

class IndexTransformer(BaseEstimator, TransformerMixin):
    """Convert a collection of raw documents to a document id matrix.

    Attributes:
        _use_char: boolean. Whether to use char feature.
        _num_norm: boolean. Whether to normalize text.
        _word_vocab: dict. A mapping of words to feature indices.
        _char_vocab: dict. A mapping of chars to feature indices.
        _label_vocab: dict. A mapping of labels to feature indices.
    """

    def __init__(self, lower=True, num_norm=True,
                 use_char=True, initial_vocab=None):
        """Create a preprocessor object.

        Args:
            lower: boolean. Whether to convert the texts to lowercase.
            use_char: boolean. Whether to use char feature.
            num_norm: boolean. Whether to normalize text.
            initial_vocab: Iterable. Initial vocabulary for expanding word_vocab.
        """
        self._num_norm = num_norm
        self._use_char = use_char
        self._word_vocab = Vocabulary(lower=lower)
        self._char_vocab = Vocabulary(lower=False)
        self._label_vocab = Vocabulary(lower=False, unk_token=False)

        if initial_vocab:
            self._word_vocab.add_documents([initial_vocab])
            self._char_vocab.add_documents(initial_vocab)

    def fit(self, X, y):
        """Learn vocabulary from training set.

        Args:
            X : iterable. An iterable which yields either str, unicode or file objects.

        Returns:
            self : IndexTransformer.
        """
        self._word_vocab.add_documents(X)
        self._label_vocab.add_documents(y)
        if self._use_char:
            for doc in X:
                self._char_vocab.add_documents(doc)

        self._word_vocab.build()
        self._char_vocab.build()
        self._label_vocab.build()

        return self

    def transform(self, X, y=None):
        """Transform documents to document ids.

        Uses the vocabulary learned by fit.

        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.
            y : iterabl, label strings.

        Returns:
            features: document id matrix.
            y: label id matrix.
        """
        word_ids = [self._word_vocab.doc2id(doc) for doc in X]
        word_ids = pad_sequences(word_ids, padding='post')

        if self._use_char:
            char_ids = [[self._char_vocab.doc2id(w) for w in doc] for doc in X]
            char_ids = pad_nested_sequences(char_ids)
            features = [word_ids, char_ids]
        else:
            features = word_ids

        if y is not None:
            y = [self._label_vocab.doc2id(doc) for doc in y]
            y = pad_sequences(y, padding='post')
            y = to_categorical(y, self.label_size).astype(int)
            # In 2018/06/01, to_categorical is a bit strange.
            # >>> to_categorical([[1,3]], num_classes=4).shape
            # (1, 2, 4)
            # >>> to_categorical([[1]], num_classes=4).shape
            # (1, 4)
            # So, I expand dimensions when len(y.shape) == 2.
            y = y if len(y.shape) == 3 else np.expand_dims(y, axis=0)
            return features, y
        else:
            return features

    def fit_transform(self, X, y=None, **params):
        """Learn vocabulary and return document id matrix.

        This is equivalent to fit followed by transform.

        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.

        Returns:
            list : document id matrix.
            list: label id matrix.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y, lengths=None):
        """Return label strings.

        Args:
            y: label id matrix.
            lengths: sentences length.

        Returns:
            list: list of list of strings.
        """
        y = np.argmax(y, -1)
        inverse_y = [self._label_vocab.id2doc(ids) for ids in y]
        if lengths is not None:
            inverse_y = [iy[:l] for iy, l in zip(inverse_y, lengths)]

        return inverse_y

    @property
    def word_vocab_size(self):
        return len(self._word_vocab)

    @property
    def char_vocab_size(self):
        return len(self._char_vocab)

    @property
    def label_size(self):
        return len(self._label_vocab)

    @property
    def label_vocab(self):
        return self._label_vocab.vocab

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)

        return p

class NERSequence(Sequence):

    def __init__(self, x, y, batch_size=1, preprocess=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.preprocess = preprocess

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return self.preprocess(batch_x, batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


def pad_nested_sequences(sequences, dtype='int32'):
    """Pads nested sequences to the same length.

    This function transforms a list of list sequences
    into a 3D Numpy array of shape `(num_samples, max_sent_len, max_word_len)`.

    Args:
        sequences: List of lists of lists.
        dtype: Type of the output sequences.

    # Returns
        x: Numpy array.
    """
    max_sent_len = 0
    max_word_len = 0
    for sent in sequences:
        max_sent_len = max(len(sent), max_sent_len)
        for word in sent:
            max_word_len = max(len(word), max_word_len)

    x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)
    for i, sent in enumerate(sequences):
        for j, word in enumerate(sent):
            x[i, j, :len(word)] = word

    return x

def filter_embeddings(embeddings, vocab, dim):
    """Loads word vectors in numpy array.

    Args:
        embeddings (dict): a dictionary of numpy array.
        vocab (dict): word_index lookup table.

    Returns:
        numpy array: an array of word embeddings.
    """
    _embeddings = np.zeros([len(vocab), dim])
    for word in vocab:
        if word in embeddings:
            word_idx = vocab[word]
            _embeddings[word_idx] = embeddings[word][0:dim]

    return _embeddings

class KerasNERModel(NamedEntityRecognitionModel):
    def __init__(self, entity_labels=None, word_embeddings=None):
        super().__init__(entity_labels)
        self.key = KEY
        # TODO: select config based on model
        self.config = BiLSTMModelConfigurationModified()
        if self.entity_labels:
            self.config.set_parameter("entity_labels", self.entity_labels)
        self.word_embeddings = word_embeddings

        self.hparams = {'optimizer': Adam(lr=0.01)}

    def fit(self,
            X,
            y=None,
            X_valid=None,
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
            use_pos_emb=False,
            hparams_1={},# specific to config 
            hparams_2={}): # other params
        """ Trains the NER model. The input is a list of
            `AnnotatedDocument` instances.

            We should be careful with batch size:
            it must satisfy len(X) % batch_size == 0.
            Otherwise, it crashes with an error from time to time.
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
            "word_lstm_units": word_lstm_units,
            "pos_emb_size": pos_emb_size,
            "use_crf": use_crf,
            "use_char_emb": use_char_emb,
            "shuffle": shuffle,
            "use_pos_emb": use_pos_emb
        })

        if hparams_1:
            self.config.set_parameters(hparams_1)
        self.config.validate()
        if hparams_2:
            self.hparams .update(hparams_2)

        log.info("Transforming {} items to BIO format...".format(len(X)))
        X_train, Y_train = self._transform_to_bio(X)

        pos_tags = []
        if use_pos_emb:
            log.info("Getting POS tags for {} items...".format(len(X)))
            for idx, x in enumerate(X_train):
                pos_tags.append(tokens_to_pos_tags(x))
                log_progress(log, idx, len(X))

        self.p = IndexTransformer(use_char=self.config.get_parameter("use_char_emb"))
        self.p.fit(X_train, Y_train)
        self.word_embeddings = filter_embeddings(self.word_embeddings, self.p._word_vocab.vocab, self.config.get_parameter("word_emb_size"))

        # compile the model architecture
        self._compile_model()

        train_seq = NERSequence(X_train, Y_train, self.config.get_parameter("batch_size"), preprocess=self.p.transform)

        if X_valid:
            X_valid, Y_valid = self._transform_to_bio(X_valid)
            valid_seq = NERSequence(X_valid, Y_valid, batch_size, self.p.transform)
            f1 = F1score(valid_seq, preprocessor=self.p)

        # train model
        log.info("Training BiLSTM...")
        self.model.fit_generator(
            generator= train_seq,
            epochs=self.config.get_parameter("num_epochs"),
            validation_data=valid_seq if valid_seq else None,
            verbose=1,
            shuffle=self.config.get_parameter("shuffle"),
            callbacks= self.hparams['callbacks']+[f1] if 'callbacks' in self.hparams else [f1])

        return self

    def _transform_to_bio(self, X):
        train_data = transform_annotated_documents_to_bio_format(X, entity_labels=self.entity_labels)

        X_train = [x_i for x_i in train_data[0]]
        y_train = [y_i for y_i in train_data[1]]

        # check sizes
        if len(X_train) != len(y_train):
            log.error("Got {} feature vectors but {} labels, cannot train!".format(len(X_train), len(y_train)))
            return self

        # TODO: Solve batch size problem if possible
        # number of examples must be divisible by batch_size,
        # so skip examples in the end if needed
        exm_num = len(X_train)
        X_train = X_train[:exm_num - exm_num % self.config.get_parameter("batch_size")]
        y_train = y_train[:exm_num - exm_num % self.config.get_parameter("batch_size")]

        return X_train, y_train

    def _compile_model(self):
        """ Defines the model architecture and compiles it """
        self.model = BiLSTMCRFModified(self.config, self)
        self.model, loss, metrics = self.model.build()
        self.model.compile(optimizer=self.hparams['optimizer'],loss=loss, metrics=metrics)
        self.model.summary(125)
        

    # noinspection PyTypeChecker
    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
            input and returns a list of `AnnotatedDocument` objects.
        """
        log.info("Annotating named entities in {} documents with BiLSTM...".format(len(X)))
        annotated_documents = []
        # x_test, _ = self._transform_to_bio(X)
        for idx, document in enumerate(X):

            # # get tokens
            tokens, _ = transform_annotated_document_to_bio_format(document)

            lengths = map(len, [tokens])
            x_test = self.p.transform([tokens])

            # get predicted tags
            output = self.model.predict(x=x_test)
            tags = self.p.inverse_transform(output, lengths)

            # annotate a document
            annotated_documents.append(transform_bio_tags_to_annotated_document(tokens, tags[0], document))
            # info
            log_progress(log, idx, len(X))

        return annotated_documents

    def predict_for_EL(self, X, y=None):
        predictions = []
        annotated_documents= []
        layer_weights = []
        model = Model(inputs=self.model.input, outputs=self.model.get_layer(name='scaled_dot_product').output)
        for idx, document in enumerate(X):

            # # get tokens
            tokens, _ = transform_annotated_document_to_bio_format(document)
            lengths = map(len, [tokens])
            x_test = self.p.transform([tokens])

            output = self.model.predict(x=x_test)
            tags = self.p.inverse_transform(output, lengths)
            predictions.append(tags)

            annotated_documents.append(transform_bio_tags_to_annotated_document(tokens, tags[0], document))

            layer_weights.append(model.predict(x=x_test))

        return annotated_documents, predictions, layer_weights
    
    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        save_path = Path(file_path)
        mkdir(save_path)
        model_save_path = save_path.joinpath("KerasNER.model")
        config_save_path = save_path.joinpath("KerasNER.config")
        arch_save_path = save_path.joinpath("KerasNER.json")
        encoder_save_path = save_path.joinpath("encoder")
        with open(arch_save_path, 'w') as f:
            params = self.model.to_json()
            json.dump(json.loads(params), f, sort_keys=True, indent=4)
            self.model.save_weights(model_save_path)
        self.config.save(config_save_path)
        self.p.save(encoder_save_path)

    def load(self, file_path, checkpoint=None):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model_load_path = load_path.joinpath("KerasNER.model")
        config_load_path = load_path.joinpath("KerasNER.config")
        arch_load_path = load_path.joinpath("KerasNER.json")
        encoder_load_path = load_path.joinpath("encoder")
        self.config.load(config_load_path)
        if self.config.get_parameter("use_crf"):
            self.p = IndexTransformer.load(encoder_load_path)
            with open(arch_load_path) as f:
                self.model = model_from_json(f.read(), custom_objects={'CRF':CRF, 'ScaledDotProduct':ScaledDotProduct,'VarOne':VarOne })
                self.model.load_weights(model_load_path)
        else:
            with open(arch_load_path) as f:
                self.model = model_from_json(f.read())
                self.model.load_weights(model_load_path)
        if checkpoint:
            self.model.load_weights(checkpoint)
        
        return self

# modification on Anago
# model classes are tightly bound to recieve config object and caller model object.
# TODO: Use parameters and define the neural models independent of caller method object. 
#       Also find a suitable location for this types neural model definition.
class BiLSTMCRFModified(object):
    """
    A Keras implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self, config_obj, NERmodel_obj):
        """Build a Bi-LSTM CRF model.

        Args:
            config_obj: Model configuration 
            model_obj: Object of KerasNERModel
        """
        super(BiLSTMCRFModified).__init__()
        self.config = config_obj
        self.model_obj = NERmodel_obj

        
    def build(self):
        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
        inputs = [word_ids]
        if self.model_obj.word_embeddings is None:
            word_embeddings = Embedding(input_dim=self.model_obj.p.word_vocab_size,
                                                        output_dim=self.config.get_parameter("word_emb_size"),
                                                        mask_zero=True,
                                                        name='word_embedding')(word_ids)
        else:
            word_embeddings = Embedding(input_dim=self.model_obj.word_embeddings.shape[0],
                                                        output_dim=self.model_obj.word_embeddings.shape[1],
                                                        mask_zero=True,
                                                        trainable=False,
                                                        weights=[self.model_obj.word_embeddings],
                                                        name='word_embedding')(word_ids)

        # build character based word embedding
        if self.config.get_parameter("use_char_emb"):
            char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
            inputs.append(char_ids)
            char_mask = False if self.config.get_parameter("use_char_cnn") else True
            char_embeddings = Embedding(input_dim=self.model_obj.p.char_vocab_size,
                                                        output_dim=self.config.get_parameter("char_emb_size"),
                                                        mask_zero=char_mask,
                                                        name='char_embeddings')(char_ids)
            
            if self.config.get_parameter("use_char_cnn"):
                char_embeddings = TimeDistributed(Conv1D(30, 3, padding='same'), name='char_cnn')(char_embeddings)
                char_embeddings = TimeDistributed(GlobalMaxPooling1D(), name='char_pooling')(char_embeddings)
            else:
                char_embeddings = TimeDistributed(Bidirectional(LSTM(self.config.get_parameter("char_lstm_units"))))(char_embeddings)

            # ref http://arxiv.org/abs/1611.04361
            if self.config.get_parameter("use_word_self_attention"):
                char_embeddings = Dense(self.config.get_parameter("word_emb_size"))(char_embeddings)
                word_emb_proj = Dense(self.config.get_parameter("word_emb_size"))(char_embeddings)
                char_emd_proj = Dense(self.config.get_parameter("word_emb_size"))(word_embeddings)
                add_proj = Activation('tanh')(K.layers.Add()([word_emb_proj, char_emd_proj]))
                att_weight = Dense(self.config.get_parameter("word_emb_size"), activation='sigmoid')(add_proj)

                x1 = K.layers.Multiply()([att_weight, word_emb_proj])
                var_one = VarOne()(att_weight)
                x2_1 = K.layers.Subtract()([var_one, att_weight])
                x2 = K.layers.Multiply()([x2_1, char_emd_proj])

                word_embeddings = K.layers.Add()([x1, x2])
            else:
                word_embeddings = Concatenate()([word_embeddings, char_embeddings])
        
        z = Bidirectional(LSTM(units=self.config.get_parameter("word_lstm_units"), return_sequences=True, dropout=max(0,self.config.get_parameter("dropout")), recurrent_dropout=max(0,self.config.get_parameter("dropout"))))(word_embeddings) # , dropout=max(0,self._dropout), kernel_regularizer=l2(), recurrent_regularizer=l2(), recurrent_dropout=max(0,self._dropout)

        z_ = ScaledDotProduct(name='scaled_dot_product')(z)
        if self.config.get_parameter("use_word_self_attention"):
            z = z_

        z = Dropout(self.config.get_parameter("dropout"))(z)
        z = Dense(self.model_obj.p.label_size, activation=None)(z) #, kernel_regularizer=l2()

        if self.config.get_parameter("use_crf"):
            crf = CRF(self.model_obj.p.label_size, sparse_target=False)
            loss = crf_loss
            pred = crf(z)
            metrics=[crf_accuracy]
        else:
            loss = 'categorical_crossentropy'
            pred = Dense(self.model_obj.p.label_size, activation='softmax')(z)
            metrics=["accuracy"]

        model = Model(inputs=inputs, outputs=pred)

        return model, loss, metrics


class ScaledDotProduct(K.layers.Layer):
    def __init__(self,**kwargs):
        super(ScaledDotProduct, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[-1]
        return mask

    def call(self, z, mask=None):
        z_embd_dim = K.backend.shape(z)[-1]
        att_weight = K.backend.batch_dot(z,z,axes=2) #/K.backend.sqrt(K.backend.cast(z_embd_dim, dtype =K.backend.floatx()))
        softmax_weight = K.activations.softmax(att_weight)
        # TODO: masking
        z_ = K.backend.batch_dot(softmax_weight,z)

        return z_

class VarOne(K.layers.Layer):
    def __init__(self,**kwargs):
        super(VarOne, self).__init__()
    
    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[-1]
        return mask

    def call(self, x):
        return K.backend.ones_like(x, dtype='float32', name='one')

