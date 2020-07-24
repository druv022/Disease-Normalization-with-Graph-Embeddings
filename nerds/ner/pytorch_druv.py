from pathlib import Path

import numpy as np
import random
import json
import torch
import torch.nn as nn

from nerds.config.base import BiLSTMModelConfiguration, BiLSTMModelConfigurationModified
from nerds.doc.bio import transform_annotated_documents_to_bio_format, transform_bio_tags_to_annotated_document, \
    transform_annotated_document_to_bio_format, transform_bio_tags_to_annotated_documents
from nerds.ner.base import NamedEntityRecognitionModel
from nerds.util.file import mkdir
from nerds.util.logging import get_logger, log_progress
from nerds.util.nlp import tokens_to_pos_tags
from nerds.evaluate.score import annotation_precision_recall_f1score
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import os, pickle, math
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
import time
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from seqeval.metrics import f1_score, classification_report, accuracy_score
# from pytorch_CRF import CRF
from typing import List, Optional, Tuple
from sklearn.utils import shuffle as Shuffle_lists

log = get_logger()

KEY = "pytorch_ner"

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

    # TODO: Revisit
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
        word_ids = torch.tensor(self._word_vocab.doc2id(X), dtype=torch.long)

        if self._use_char:
            char_ids = [[self._char_vocab.doc2id(w) for w in X]]
            char_ids = pad_nested_sequences(char_ids)
            char_ids = torch.tensor(char_ids, dtype=torch.long)
            features = [word_ids, char_ids.squeeze(dim=0)]
        else:
            features = [word_ids]

        if y is not None:
            y = torch.tensor(self._label_vocab.doc2id(y), dtype=torch.long)
            return [features, y]
        else:
            return features



    # used for minibatch method
    def transform2(self, X, y=None):
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
        word_ids = [torch.tensor(self._word_vocab.doc2id(doc), dtype=torch.double) for doc in X ]
        # word_idx = sorted(range(len(word_ids)),key=lambda idx: len(word_ids[idx]), reverse=True)
        # word_ids = [word_ids[i] for i in word_idx]        
        word_ids = nn.utils.rnn.pad_sequence(word_ids, batch_first=True)

        if self._use_char:
            char_ids = [[self._char_vocab.doc2id(w) for w in doc] for doc in X] 
            char_ids = pad_nested_sequences(char_ids)
            char_ids = torch.tensor(char_ids)
            features = [word_ids, char_ids]
        else:
            features = word_ids

        if y is not None:
            y = [torch.tensor(self._label_vocab.doc2id(doc)) for doc in y]
            # y = [y[i] for i in word_idx]
            y = nn.utils.rnn.pad_sequence(y, batch_first=True)
            return [features, y]
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
        # y = np.argmax(y, -1)
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
    def word_vocab(self):
        return self._word_vocab.vocab

    @property
    def char_vocab(self):
        return self._char_vocab.vocab

    @property
    def label_vocab(self):
        return self._label_vocab.vocab

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)

        return p

class NERSequence(Dataset):

    def __init__(self, x, y=None, preprocess=None):
        self.x = x
        self.y = y
        self.preprocess = preprocess

    def __getitem__(self, idx):
        if self.y is not None:
            x_data = self.preprocess(self.x[idx], self.y[idx])
        else:
            x_data = self.preprocess(self.x[idx], self.y)
        return x_data

    def __len__(self):
        return len(self.x)


def pad_nested_sequences(sequences, dtype='int64'):
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

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size, dtype=torch.long)], dim=dim)

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, use_char=True):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = -1
        self.use_char = use_char

    def pad_collate(self, batch):
        """
        args:
            batch - list of (list of tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # print(batch)

        def lambda_func1(data):
            x1_data = [] # words
            x2_data = [] # character
            y_data = [] # labels
            for x in data:
                # pad alond word dim
                x1_data.append(pad_tensor(x[0][0],pad=max_word_len,dim=-1))
                # pad along word dim
                temp = pad_tensor(x[0][1],pad=max_word_len,dim=-2)
                # pad along char dim
                x2_data.append(pad_tensor(temp,pad=max_char_length,dim=-1))
                # pad along word dim
                y_data.append(pad_tensor(x[1], pad=max_word_len, dim=-1))

            return [torch.stack(x1_data, dim=0), torch.stack(x2_data, dim=0)], torch.stack(y_data, dim=0)

        def lambda_func2(data):
            x1_data = []
            x2_data = []
            for x in data:
                # pad alond word dim
                x1_data.append(pad_tensor(x[0][0],pad=max_word_len,dim=-1))
                # pad along word dim
                temp = pad_tensor(x[0][1],pad=max_word_len,dim=-2)
                # pad along char dim
                x2_data.append(pad_tensor(temp,pad=max_char_length,dim=-1))

            return torch.stack(x1_data, dim=0), torch.stack(x2_data, dim=0)

        def lambda_func3(data):
            x1_data = []
            y_data = []
            for x in data:
                # pad alond word dim
                x1_data.append(pad_tensor(x[0][0],pad=max_word_len,dim=-1))
                # pad alond word dim
                y_data.append(pad_tensor(x[1], pad=max_word_len, dim=-1))

            return torch.stack(x1_data, dim=0), torch.stack(y_data, dim=0)

        def lambda_func4(data):
            x1_data = []
            for x in data:
                # pad alond word dim
                x1_data.append(pad_tensor(x[0][0],pad=max_word_len,dim=-1))

            return torch.stack(x1_data, dim=0)

        if self.use_char:
            # get max mord length and character length (for current thread)
            max_word_len = max(map(lambda x: len(x[0][0]), batch))
            max_char_length = max(map(lambda x: len(x[0][1][0]), batch))
            max_char_length = max(max_char_length, 10) # 3: Kernal size, 2 layer conv

            if len(batch[0]) > 1:
                return_padded = lambda_func1(batch)
            else:
                return_padded = lambda_func2(batch)
        else:
            max_word_len = max(map(lambda x: len(x[0][0]), batch))

            if len(batch[0]) > 1:
                return_padded = lambda_func3(batch)
            else:
                return_padded = lambda_func4(batch)

        return return_padded

    def __call__(self, batch):
        return self.pad_collate(batch)

class PytorchNERModel(NamedEntityRecognitionModel):
    def __init__(self, entity_labels=None, word_embeddings=None):
        super().__init__(entity_labels)
        self.key = KEY
        # TODO: select config based on model
        self.config = BiLSTMModelConfigurationModified()
        if self.entity_labels:
            self.config.set_parameter("entity_labels", self.entity_labels)
        self.word_embeddings = word_embeddings
        self.model = None
        use_cuda = torch.cuda.is_available()
        self.hparams = {'optimizer': 'adam', 'lr': 0.001, 'seq_lenght':200, 
                        'device': torch.device("cuda:0" if use_cuda else "cpu"),
                        'file_path':'tmp', 'save_best':True, 'use_GRU':False}

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
            self.hparams.update(hparams_2)

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
        self.word_embeddings = torch.from_numpy(self.word_embeddings)

        print("------------------- Training BiLSTM ---------------------------")
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        params = {'batch_size': self.config.get_parameter('batch_size'),
                    'shuffle': self.config.get_parameter('shuffle'), 'num_workers': 12}
                    
        training_set = NERSequence(X_train, Y_train, preprocess=self.p.transform)
        training_generator = DataLoader(training_set,collate_fn=PadCollate(use_char=self.config.get_parameter('use_char_emb')),**params)

        if X_valid:
            x_valid, y_valid = self._transform_to_bio(X_valid)

            params = {'batch_size': self.config.get_parameter('batch_size'),
                    'shuffle': False, 'num_workers': 12}

            validation_set = NERSequence(x_valid, y_valid, preprocess=self.p.transform)
            validation_generator = DataLoader(validation_set,collate_fn=PadCollate(use_char=self.config.get_parameter('use_char_emb')),**params)

        self.model = BiLSTMCRF(self.config, self, batch_first=True, device=self.hparams['device'])
               
        self.model.to(self.hparams['device'])

        if self.hparams['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr = self.hparams['lr'])

        with open(self.hparams['output'],'w+') as f:

            prev_f1 = 0.
            for epoch in range(self.config.get_parameter("num_epochs")):
                print("########## Epoch ", epoch, "##################")
                f.writelines('########## Epoch ", epoch, "##################\n')

                X_train, Y_train = Shuffle_lists(X_train, Y_train)
                train_loss = []
                self.model.train()
                start = time.time()
                # for x, y in self.mini_batch(X_train, Y_train, p=self.p.transform2, batch_size=self.config.get_parameter('batch_size')): # Used for debugging
                for x, y in training_generator:
                    # print('training')
                    if isinstance(x,list):
                        x_0, x_1, y = x[0].to(self.hparams['device']), x[1].to(self.hparams['device']), y.to(self.hparams['device'])
                    else:
                        x, y = x.to(self.hparams['device']), y.to(self.hparams['device'])
                    # print(x_1.shape)
                    optimizer.zero_grad()

                    mask = torch.where(y > self.p.label_vocab['<pad>'], torch.tensor([1], dtype=torch.uint8,device=self.hparams['device']), \
                            torch.tensor([0], dtype=torch.uint8, device=self.hparams['device']))

                    if self.config.get_parameter('use_char_emb'):
                        z, _ = self.model([x_0,x_1], y ,mask)
                    else:
                        z, _ = self.model([x],y, mask)

                    loss = z

                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
                
                print("Epoch: ",epoch,"\tTraining Loss: ",np.mean(train_loss), "\tTime: ", time.time()-start)
                f.writelines(f"Epoch: {epoch}\tTraining Loss: {np.mean(train_loss)}\tTime: {time.time()-start}\n")

                self.model.eval()
                valid_loss = []
                y_pred = []
                start = time.time()
                
                with torch.no_grad():
                    # for x, y in self.mini_batch(x_valid, y_valid, p=self.p.transform2, batch_size=self.config.get_parameter('batch_size')): # used for debugging
                    for x, y in validation_generator:
                        if isinstance(x,list):
                            x_0, x_1, y = x[0].to(self.hparams['device']), x[1].to(self.hparams['device']), y.to(self.hparams['device'])
                        else:
                            x, y = x.to(self.hparams['device']), y.to(self.hparams['device'])

                        mask = torch.where(y > self.p.label_vocab['<pad>'], torch.tensor([1], dtype=torch.uint8,device=self.hparams['device']), \
                                torch.tensor([0], dtype=torch.uint8, device=self.hparams['device']))
                        
                        if self.config.get_parameter('use_char_emb'):
                            z, _ = self.model([x_0,x_1], y ,mask)
                            pred, _ = self.model.decode([x_0,x_1], mask)
                        else:
                            z, _ = self.model([x],y, mask)
                            pred, _ = self.model.decode([x], mask)

                        loss = z

                        valid_loss.append(loss.item())
                        y_pred.extend(pred)

                print("Epoch: ", epoch, "\tValidation Loss: ",np.mean(valid_loss), "\tTime: ", time.time()-start)
                f.writelines(f"Epoch: {epoch}\tValidation Loss: {np.mean(valid_loss)} \tTime: {time.time()-start}\n")

                lengths = map(len, x_valid)
                tags = self.p.inverse_transform(np.asarray(y_pred), lengths)
                print('F1: ',f1_score(y_valid, tags),'\t Acc: ', accuracy_score(y_valid, tags))
                f.writelines(f'F1: {f1_score(y_valid, tags)}\t Acc: {accuracy_score(y_valid, tags)}\n')
                print(classification_report(y_valid, tags))
                f.writelines(classification_report(y_valid, tags))

                x_pred = transform_bio_tags_to_annotated_documents(x_valid, tags, X_valid)

                p, r, f1 = annotation_precision_recall_f1score(x_pred, X_valid)
                print("Disease:\tPrecision: ", p, "\tRecall: " ,r, "\tF-score: ", f1)
                f.writelines(f"Disease:\tPrecision: {p}\tRecall: {r}\tF-score: {f1}\n")

                if self.hparams['save_best'] and f1 > prev_f1:
                    self.save(self.hparams['file_path'])
                    prev_f1 = f1
                    print('New best: ', prev_f1, '\tSaving model...')
                    f.writelines(f'New best: {prev_f1}\tSaving model...\n')

                print("Best so far: ", prev_f1)
                f.writelines(f"Best so far: {prev_f1}\n")

        return self

    def mini_batch(self, x, y=None, p=None, batch_size=1):
        """Generate minibatch
        
        Arguments:
            x {list} -- list of documents
        
        Keyword Arguments:
            y {list} -- Targets (default: {None})
            p {fuction} -- Preprocess fuction (default: {None})
            batch_size {int} -- Size of Batch (default: {1})
        """
        length_x = len(x)
        for idx in range(0, math.ceil(length_x/batch_size)):
            if length_x - batch_size < 0:
                updated_size = length_x
            else:
                updated_size = batch_size
            length_x = length_x - batch_size
            batch_x = x[idx*batch_size : idx*batch_size + updated_size]
            if y is not None:
                batch_y = y[idx*batch_size : idx*batch_size + updated_size]
            else:
                batch_y = y
            yield p(batch_x, batch_y)


    def _transform_to_bio(self, X):
        train_data = transform_annotated_documents_to_bio_format(X, entity_labels=self.entity_labels)

        X_train = [x_i for x_i in train_data[0]]
        y_train = [y_i for y_i in train_data[1]]

        # check sizes
        if len(X_train) != len(y_train):
            log.error("Got {} feature vectors but {} labels, cannot train!".format(len(X_train), len(y_train)))
            return self

        return X_train, y_train
      
    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
            input and returns a list of `AnnotatedDocument` objects.
        """
        log.info("Annotating named entities in {} documents with BiLSTM...".format(len(X)))

        predictions = []
        annotated_documents= []
        self.model.eval()
        x_test, y_test = self._transform_to_bio(X)
        att_weights = []

        # set requires
        with torch.no_grad():
            for x, y in self.mini_batch(x_test, y_test, p=self.p.transform2, batch_size=self.config.get_parameter('batch_size')):
                if isinstance(x,list):
                    x_0, x_1, y = x[0].to(self.hparams['device']), x[1].to(self.hparams['device']), y.to(self.hparams['device'])
                else:
                    x, y = x.to(self.hparams['device']), y.to(self.hparams['device'])

                mask = torch.where(y > self.p.label_vocab['<pad>'], torch.tensor([1], dtype=torch.uint8,device=self.hparams['device']), \
                        torch.tensor([0], dtype=torch.uint8, device=self.hparams['device']))
                
                if self.config.get_parameter('use_char_emb'):
                    pred, att_weight = self.model.decode([x_0,x_1], mask=mask)
                else:
                    pred, att_weight = self.model.decode([x], mask=None)

                predictions.extend(pred)
                att_weights.extend(att_weight.cpu().numpy())

        lengths = map(len, x_test)
        tags = self.p.inverse_transform(np.asarray(predictions), lengths)

        annotated_documents = transform_bio_tags_to_annotated_documents(x_test, tags, X)

        return annotated_documents, [x_test, att_weights]

    # TODO: Revisit
    def predict_for_EL(self, X, y=None):
        predictions = []
        annotated_documents= []
        att_weights_list = []
        self.model.eval()
        x_test, _ = self._transform_to_bio(X)
        with torch.no_grad():
            for x in self.mini_batch(x_test, None, p=self.p.transform2, batch_size=self.config.get_parameter('batch_size')):
                if isinstance(x,list):
                    x_0, x_1= x[0].to(self.hparams['device']), x[1].to(self.hparams['device'])
                else:
                    x = x.to(self.hparams['device'])
                
                if self.config.get_parameter('use_char_emb'):
                    pred, att_weights = self.model.decode([x_0,x_1], mask=None)
                else:
                    pred, att_weights = self.model.decode([x], mask=None)

                predictions.extend(pred)
                if att_weights is not None:
                    att_weights_list.extend(att_weights.detach().cpu())

        lengths = map(len, x_test)
        tags = self.p.inverse_transform(np.asarray(predictions), lengths)

        annotated_documents = transform_bio_tags_to_annotated_documents(x_test, tags, X)

        return annotated_documents,tags, att_weights_list
    
    # TODO: save/load model dict instead of the full model
    # TODO: change model name for NERDS full functionality
    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        save_path = Path(file_path)
        model_save_path = save_path.joinpath('trained_model_'+self.timestr+'.pkl')
        config_save_path = save_path.joinpath("PytorchNER.config")
        encoder_save_path = save_path.joinpath("encoder")
        self.config.save(config_save_path)
        torch.save(self.model, model_save_path)
        # torch.save(self.model.state_dict(), model_save_path) # need to debug
        self.p.save(encoder_save_path)

    def load(self, file_path, file_name):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model_load_path = load_path.joinpath(file_name)
        config_load_path = load_path.joinpath("PytorchNER.config")
        encoder_load_path = load_path.joinpath("encoder")
        self.config.load(config_load_path)
        self.p = IndexTransformer.load(encoder_load_path)
        # if not self.model: # need debugging
        #     self.word_embeddings = filter_embeddings(self.word_embeddings, self.p._word_vocab.vocab, self.config.get_parameter("word_emb_size"))
        #     self.word_embeddings = torch.from_numpy(self.word_embeddings)
        #     self.model = BiLSTMCRF(self.config, self, batch_first=True, device=self.hparams['device'])
        #     self.model.to(self.hparams['device'])
        # self.model.load_state_dict(torch.load(model_load_path))
        self.model = torch.load(model_load_path)
        self.model.eval()
        
        return self


class BiLSTMCRF(nn.Module):
    """
    A Pytorch implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360

    Ref: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    """

    def __init__(self, config, NERmodel, batch_first=True, device='cpu'):
        """Build a Bi-LSTM CRF model.

        Args:
            config: Model configuration obj
            model: Object of KerasNERModel
        """
        super(BiLSTMCRF,self).__init__()

        self.rnn = BiLSTM(config,NERmodel, batch_first=batch_first, device=device)
        self.crf = CRF(NERmodel.p.label_size, batch_first=batch_first)

    def forward(self, X, y, mask):
        h, z_att = self.rnn(X)
        z = self.crf(h,y,mask=mask, reduction='mean')
        return z, z_att

    def decode(self, X, mask):
        h, z_att = self.rnn(X, mask)
        z = self.crf.decode(h, mask)
        return z, z_att


class BiLSTM(nn.Module):

    def __init__(self, config, NERmodel, batch_first=True, device='cpu'):
        """Build a Bi-LSTM CRF model.

        Args:
            config: Model configuration obj
            model: Object of KerasNERModel
        """
        super(BiLSTM,self).__init__()
        self.char_emb_size = config.get_parameter('char_emb_size')
        self.word_emb_size = config.get_parameter('word_emb_size')
        self.char_lstm_unit = config.get_parameter('char_lstm_units')
        self.word_lstm_unit = config.get_parameter('word_lstm_units')
        self.batch_size = config.get_parameter('batch_size')
        self.batch_first = batch_first
        self.use_crf = config.get_parameter('use_crf')
        self.use_char_emb = config.get_parameter('use_char_emb')
        self.use_char_cnn= config.get_parameter('use_char_cnn')
        self.use_char_attention = config.get_parameter('use_char_attention')
        self.use_word_self_attention= config.get_parameter('use_word_self_attention')
        self.dropout = config.get_parameter('dropout')
        self.num_tags = NERmodel.p.label_size
        self.device = device
        self.use_GRU = NERmodel.hparams['use_GRU']
        
        self.word_embedding = nn.Embedding(NERmodel.p.word_vocab_size, self.word_emb_size, padding_idx=NERmodel.p.word_vocab['<pad>'])
        self.word_embedding.from_pretrained(NERmodel.word_embeddings)
        self.word_embedding.weight.requires_grad = False
        self.char_embedding = nn.Embedding(NERmodel.p.char_vocab_size, self.char_emb_size, padding_idx=NERmodel.p.char_vocab['<pad>'])
        self.char_embedding.weight.requires_grad = False
        self.num_layers = 1

        if self.use_GRU:
            self.char_rnn= nn.GRU(input_size = self.char_emb_size, hidden_size=self.char_lstm_unit, 
                                    batch_first=self.batch_first, dropout=self.dropout, bidirectional= True)
            if self.use_char_attention:
                self.word_rnn = nn.GRU(input_size=self.word_emb_size, hidden_size=self.word_lstm_unit, 
                                        batch_first=self.batch_first, dropout=self.dropout, bidirectional=True)
            else:
                if self.use_char_emb:
                    self.word_rnn = nn.GRU(input_size=self.word_emb_size+2*self.char_lstm_unit, hidden_size=self.word_lstm_unit, num_layers=self.num_layers,
                                        batch_first=self.batch_first,  bidirectional=True)
                else:
                    self.word_rnn = nn.GRU(input_size=self.word_emb_size, hidden_size=self.word_lstm_unit, 
                                            batch_first=self.batch_first, dropout=self.dropout, bidirectional=True)
        else:
            self.char_rnn= nn.LSTM(input_size = self.char_emb_size, hidden_size=self.char_lstm_unit, 
                                        batch_first=self.batch_first, dropout=self.dropout, bidirectional= True)
            if self.use_char_attention:
                self.word_rnn = nn.LSTM(input_size=self.word_emb_size, hidden_size=self.word_lstm_unit, 
                                        batch_first=self.batch_first, dropout=self.dropout, bidirectional=True)
            else:
                if self.use_char_emb:
                    self.word_rnn = nn.LSTM(input_size=self.word_emb_size+2*self.char_lstm_unit, hidden_size=self.word_lstm_unit, num_layers=self.num_layers,
                                        batch_first=self.batch_first,  bidirectional=True)
                else:
                    self.word_rnn = nn.LSTM(input_size=self.word_emb_size, hidden_size=self.word_lstm_unit, 
                                            batch_first=self.batch_first, dropout=self.dropout, bidirectional=True)
        
        self.char_cov = CharCNN(self.char_emb_size, 2*self.char_emb_size)

        self.fc1 = nn.Linear(2*self.char_lstm_unit, self.word_emb_size, bias=False)
        self.fc2 = nn.Linear(self.word_emb_size, self.word_emb_size, bias=False)
        self.fc3 = nn.Linear(self.word_emb_size, self.word_emb_size, bias=False)
        self.fc4 = nn.Linear(self.word_emb_size, self.word_emb_size, bias=False)
        self.fc5 = nn.Linear(2*self.word_lstm_unit, NERmodel.p.label_size)
        self.fc6 = nn.Linear(NERmodel.p.label_size, NERmodel.p.label_size)

        self.dropout_layer1 = nn.Dropout(p=self.dropout)
        self.dropout_layer2 = nn.Dropout(p=self.dropout)
        self.dropout_layer3 = nn.Dropout(p=self.dropout)


        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)


    def _init_word_hidden(self, batch_size):
        # default bidirectional
        direction_dim = 2
        if self.use_GRU:
            word_hidden = torch.rand([direction_dim*self.num_layers, batch_size, self.word_lstm_unit]).to(self.device)
        else:
            word_hidden = torch.rand([direction_dim*self.num_layers, batch_size, self.word_lstm_unit]).to(self.device),\
                            torch.rand([direction_dim*self.num_layers, batch_size, self.word_lstm_unit]).to(self.device)
        return word_hidden

    
    def _init_char_hidden(self, batch_size):
        # default bidirectional
        direction_dim = 2
        if self.use_GRU:
            char_hidden = torch.rand([direction_dim, batch_size, self.char_lstm_unit]).to(self.device)
        else:
            char_hidden = torch.rand([direction_dim, batch_size, self.char_lstm_unit]).to(self.device),\
                            torch.rand([direction_dim, batch_size, self.char_lstm_unit]).to(self.device)
        
        return char_hidden

    def forward(self, X, mask=None):
        # Process word

        word_emb = self.word_embedding(X[0].long())

        if self.use_char_emb:
            self.batch_size, w_seq_length, c_seq_length = X[1].shape
            char_emb = self.char_embedding(X[1])
            w_emb = []
            if self.use_char_cnn:
                for i in range(w_seq_length):
                    if c_seq_length != 1:
                        c_emb = torch.squeeze(char_emb[:,i,:,:],dim=1)
                    else:
                        c_emb = char_emb[:,i,:,:]
                    c_emb = self.dropout_layer3(c_emb)
                    c_emb = self.char_cov(c_emb.transpose(-1,-2))
                    w_emb.append(c_emb)
            else:
                for i in range(w_seq_length):
                    self.char_hidden = self._init_char_hidden(self.batch_size)
                    if c_seq_length != 1:
                        c_emb = torch.squeeze(char_emb[:,i,:,:],dim=1)
                    else:
                        c_emb = char_emb[:,i,:,:]
                    c_emb, _ = self.char_rnn(c_emb, self.char_hidden)
                    c_emb = c_emb[:,-1,:]
                    w_emb.append(c_emb)
            
            char_emb = torch.stack(w_emb, dim=1)

            # use char attention
            if self.use_char_attention:
                char_emb = self.fc1(char_emb)
                char_emb_proj = self.fc2(char_emb)
                word_emb_proj = self.fc3(word_emb)
                emb_comb = self.tanh(char_emb_proj + word_emb_proj)
                att_weight = self.sigmoid(self.fc4(emb_comb))

                x1 = att_weight * char_emb
                x2 = (1 - att_weight) * word_emb
                word_emb = x1 + x2
            else:
                word_emb = torch.cat((word_emb, char_emb),-1)
        else:
            self.batch_size, w_seq_length = X[0].shape

        self.word_hidden = self._init_word_hidden(self.batch_size)
        z = self.dropout_layer1(word_emb)
        z, _ = self.word_rnn(word_emb, self.word_hidden)

        if mask is not None:
            mask = mask.unsqueeze(2).expand(-1,-1,z.shape[2]).float()
            z = mask * z

        # self attention
        att_weight = torch.matmul(z, z.transpose(-1,-2))
        softmax_weight = self.softmax(att_weight)
        z_att = torch.matmul(softmax_weight, z)
        if self.use_word_self_attention:
            z = z_att
        
        z = self.dropout_layer2(z)
        z = self.fc5(z)
        z = self.fc6(z)
        return z, softmax_weight

class CharCNN(nn.Module):

    def __init__(self, num_features, out_feature):
        super(CharCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, out_feature, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x, _ = torch.max(x,-1)

        return x

# from https://github.com/kmkurn/pytorch-crf/blob/master/torchcrf/__init__.py
class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = denominator - numerator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
