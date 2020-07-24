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
from allennlp.modules.elmo import Elmo, batch_to_ids
from nerds.util.nlp import text_to_tokens, text_to_sentences

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

# TODO: Scope to optimize maybe!
class NERSequence(Dataset):

    def __init__(self, x, y=None, preprocess=None):
        self.x = x
        self.y = y
        self.preprocess = preprocess

    def __getitem__(self, idx):
        if self.y is not None:
            x_data = [self.x[idx], torch.tensor(self.preprocess(self.y[idx]))]
        else:
            x_data = self.x[idx]
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

def pad_tensor(vec, pad, dim, dtype):
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
    return torch.cat([vec, torch.zeros(*pad_size, dtype=dtype)], dim=dim)

def padding(batch):
    """Pad batch
    
    Arguments:
        batch {list} -- List of feature
    
    Returns:
        List -- List 
    """
    max_len = max(map(lambda x: len(x[0]), batch))
    sorted_idx = sorted(range(len(batch)),key=lambda idx: len(batch[idx][0]), reverse=True)

    x, y = [], []
    for idx in sorted_idx:
        item = batch[idx]
        x.append(pad_tensor(item[0].float(),pad=max_len,dim=0, dtype=torch.float))
        y.append(pad_tensor(item[1],pad=max_len,dim=0, dtype=torch.long))

    return [torch.stack(x, dim=0), torch.stack(y, dim=0), sorted_idx]

def annotated_docs_to_tokens(docs, sentence_pad=False):
    """Align tokenized docs
    
    """
    text_list = []
    label_list = []
    tokens_list = []
    for i,doc in enumerate(docs):
        if sentence_pad:
            text = [[r'<s>']+ text_to_tokens(sent)+[r'<\s>'] for sent in text_to_sentences(doc.plain_text_)[0] if len(sent.split()) > 0]
        else:
            text = [text_to_tokens(sent) for sent in text_to_sentences(doc.plain_text_)[0] if len(sent.split()) > 0]

        text_list.append(text)

        count = 0
        pad_index = []
        for line in text:
            for idx,word in enumerate(line):
                if word==r'<s>' or word==r'<\s>':
                    pad_index.append(count+idx)
            count+= len(line)

        tokens, labels = transform_annotated_document_to_bio_format(doc)

        count = 0
        for i,line in enumerate(text_list[-1]):
            start_count = 0
            for j,word in enumerate(line):
                if word not in [r'<s>',r'<\s>'] and word != tokens[count]:
                    k=0
                    start_count = count
                    if tokens[count] in word:
                        text_list[-1][i][j] = tokens[count+k]
                        k += 1
                    while count+k < len(tokens) and tokens[count+k] in word:
                        text_list[-1][i].insert(j+k,tokens[count+k])
                        k += 1
                    # print(f'Error: split text= {word}, token{tokens[start_count:count+k]}')
                    count += 1
                elif word not in [r'<s>',r'<\s>']:
                    count += 1

        [labels.insert(i, 'O') for i in pad_index]
        [tokens.insert(i,r'<s>') for i in pad_index]

        label_list.append(labels)
        tokens_list.append(tokens)

    return text_list, label_list, tokens_list

def inverse_transform(y, label_vocab, lengths=None):
    # y = np.argmax(y, -1)
    inverse_y = [label_vocab.id2doc(ids) for ids in y]
    if lengths is not None:
        inverse_y = [iy[:l] for iy, l in zip(inverse_y, lengths)]

    return inverse_y

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
        self.hparams = {'optimizer': 'adam', 'lr': 0.001, 'seq_length':200, 
                        'device': torch.device("cuda:0" if use_cuda else "cpu"),
                        'file_path':'tmp', 'save_best':True, 'use_GRU':False, 'elmo_dim' : 1024}

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

        x_train_text, ner_train_labels, x_train_tokens = annotated_docs_to_tokens(X)

        elmo = Elmo(self.hparams['options_file'], self.hparams['weight_file'], 2,dropout=0)
        elmo.to(self.hparams['device'])

        text_train = []
        for idx, t in enumerate(x_train_text):
            char_id = batch_to_ids(t).to(self.hparams['device'])
            with torch.no_grad():
                elmo_emb = elmo(char_id)
            t_emb = elmo_emb['elmo_representations'][0].view(-1, self.hparams['elmo_dim']).detach().cpu()
            t_emb = torch.stack([tensor for tensor in t_emb if len(np.nonzero(tensor.numpy())[0])!=0],dim=0)
            text_train.append(t_emb)

        self.ner_labels_vocab = Vocabulary(lower=False)
        self.ner_labels_vocab.add_documents(ner_train_labels)
        self.ner_labels_vocab.build()

        print("------------------- Training BiLSTM ---------------------------")
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        params = {'batch_size': self.config.get_parameter('batch_size'),
                    'shuffle': self.config.get_parameter('shuffle'), 'num_workers': 1}
                    
        training_set = NERSequence(text_train, ner_train_labels, preprocess=self.ner_labels_vocab.doc2id)
        training_generator = DataLoader(training_set,collate_fn=padding,**params)

        if X_valid:
            x_val_text, ner_val_labels, x_val_tokens = annotated_docs_to_tokens(X_valid)

            text_val = []
            for idx, t in enumerate(x_val_text):
                char_id = batch_to_ids(t).to(self.hparams['device'])
                with torch.no_grad():
                    elmo_emb = elmo(char_id)
                t_emb = elmo_emb['elmo_representations'][0].view(-1, self.hparams['elmo_dim']).detach().cpu()
                t_emb = torch.stack([tensor for tensor in t_emb if len(np.nonzero(tensor.numpy())[0])!=0],dim=0)
                text_val.append(t_emb)

            params = {'batch_size': self.config.get_parameter('batch_size'),
                    'shuffle': False, 'num_workers': 12}

            validation_set = NERSequence(text_val, ner_val_labels, preprocess=self.ner_labels_vocab.doc2id)
            validation_generator = DataLoader(validation_set,collate_fn=padding,**params)

        self.model = BiLSTMCRF(self.config, self, batch_first=True, device=self.hparams['device'])
        self.model.to(self.hparams['device'])

        if self.hparams['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr = self.hparams['lr'])

        with open(self.hparams['output'],'w+') as f:

            prev_f1 = 0.
            for epoch in range(self.config.get_parameter("num_epochs")):
                print("########## Epoch ", epoch, "##################")
                f.writelines('########## Epoch ", epoch, "##################\n')

                train_loss = []
                self.model.train()
                start = time.time()
                # text_train, ner_train_labels= Shuffle_lists(text_train, ner_train_labels) # Used for debugging
                # for batch in self.mini_batch(text_train, ner_train_labels, p=self.ner_labels_vocab.doc2id, batch_size=self.config.get_parameter('batch_size')): # Used for debugging
                #     x, y, _ = padding(batch) # used for debugging
                for x, y, _ in training_generator:
                    x, y = x.to(self.hparams['device']), y.to(self.hparams['device'])

                    optimizer.zero_grad()

                    mask = torch.where(y != self.ner_labels_vocab.vocab['<pad>'], torch.tensor([1], dtype=torch.uint8,device=self.hparams['device']), \
                            torch.tensor([0], dtype=torch.uint8, device=self.hparams['device']))

                    z, _ = self.model(x, y, mask)

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
                    # for batch in self.mini_batch(text_val,  ner_val_labels, p=self.ner_labels_vocab.doc2id, batch_size=self.config.get_parameter('batch_size')): # used for debugging
                    #     x, y, sorted_idx = padding(batch) # used for debugging
                    for x, y, sorted_idx in validation_generator:
                        x, y = x.to(self.hparams['device']), y.to(self.hparams['device'])

                        mask = torch.where(y != self.ner_labels_vocab.vocab['<pad>'], torch.tensor([1], dtype=torch.uint8,device=self.hparams['device']), \
                                torch.tensor([0], dtype=torch.uint8, device=self.hparams['device']))
                        
                        z, _ = self.model(x,y, mask)
                        pred, _ = self.model.decode(x, mask)
                        
                        loss = z

                        valid_loss.append(loss.item())

                        pred_unsort = [0]*x.shape[0]
                        for i,j in zip(sorted_idx, pred):
                            pred_unsort[i]=j 
                        y_pred.extend(pred_unsort)

                print("Epoch: ", epoch, "\tValidation Loss: ",np.mean(valid_loss), "\tTime: ", time.time()-start)
                f.writelines(f"Epoch: {epoch}\tValidation Loss: {np.mean(valid_loss)} \tTime: {time.time()-start}\n")

                lengths = map(len, x_val_tokens)
                tags = inverse_transform(np.asarray(y_pred), self.ner_labels_vocab , lengths)
                print('F1: ',f1_score(ner_val_labels, tags),'\t Acc: ', accuracy_score(ner_val_labels, tags))
                f.writelines(f'F1: {f1_score(ner_val_labels, tags)}\t Acc: {accuracy_score(ner_val_labels, tags)}\n')
                print(classification_report(ner_val_labels, tags))
                f.writelines(classification_report(ner_val_labels, tags))

                x_pred = transform_bio_tags_to_annotated_documents(x_val_tokens, tags, X_valid)

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
                batch_y = [torch.tensor(p(i)) for i in y[idx*batch_size : idx*batch_size + updated_size]]
            else:
                batch_y = y
            batch = [[i[0],i[1]] for i in zip(batch_x, batch_y)]
            yield batch


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

        self.model.eval()

        x_test_text, ner_test_labels, x_test_tokens = annotated_docs_to_tokens(X)

        elmo = Elmo(self.hparams['options_file'], self.hparams['weight_file'], 2,dropout=0)
        elmo.to(self.hparams['device'])

        att_weights = []
        text_test = []
        for idx, t in enumerate(x_test_text):
            char_id = batch_to_ids(t).to(self.hparams['device'])
            with torch.no_grad():
                elmo_emb = elmo(char_id)
            t_emb = elmo_emb['elmo_representations'][0].view(-1, self.hparams['elmo_dim']).detach().cpu()
            t_emb = torch.stack([tensor for tensor in t_emb if len(np.nonzero(tensor.numpy())[0])!=0],dim=0)
            text_test.append(t_emb)

        y_pred = []
        with torch.no_grad():
            for batch in self.mini_batch(text_test,  ner_test_labels, p=self.ner_labels_vocab.doc2id, batch_size=self.config.get_parameter('batch_size')): # used for debugging
                x, y, sorted_idx = padding(batch) # used for debugging
            # for x, y, sorted_index in validation_generator:
                x, y = x.to(self.hparams['device']), y.to(self.hparams['device'])

                mask = torch.where(y != self.ner_labels_vocab.vocab['<pad>'], torch.tensor([1], dtype=torch.uint8,device=self.hparams['device']), \
                        torch.tensor([0], dtype=torch.uint8, device=self.hparams['device']))
                
                z, _ = self.model(x,y, mask)
                pred, att_weight = self.model.decode(x, mask)

                pred_unsort = [0]*x.shape[0]
                for i,j in zip(sorted_idx, pred):
                    pred_unsort[i]=j 
                y_pred.extend(pred_unsort)
                att_weights.extend(att_weight.cpu().numpy())

        lengths = map(len, x_test_tokens)
        tags = inverse_transform(np.asarray(y_pred), self.ner_labels_vocab , lengths)
        print('F1: ',f1_score(ner_test_labels, tags),'\t Acc: ', accuracy_score(ner_test_labels, tags))
        print(classification_report(ner_test_labels, tags))

        x_pred = transform_bio_tags_to_annotated_documents(x_test_tokens, tags, X)

        p, r, f1 = annotation_precision_recall_f1score(x_pred, X)

        return x_pred, [x_test_tokens, att_weights]

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
        params_save_path = save_path.joinpath("params")
        self.config.save(config_save_path)
        torch.save(self.model, model_save_path)
        # torch.save(self.model.state_dict(), model_save_path) # need to debug
        with open(params_save_path, 'wb+') as f:
            pickle.dump([self.hparams, self.ner_labels_vocab], f)

    def load(self, file_path, file_name):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model_load_path = load_path.joinpath(file_name)
        config_load_path = load_path.joinpath("PytorchNER.config")
        params_load_path = load_path.joinpath("params")
        self.config.load(config_load_path)
        # if not self.model: # need debugging
        #     self.word_embeddings = filter_embeddings(self.word_embeddings, self.p._word_vocab.vocab, self.config.get_parameter("word_emb_size"))
        #     self.word_embeddings = torch.from_numpy(self.word_embeddings)
        #     self.model = BiLSTMCRF(self.config, self, batch_first=True, device=self.hparams['device'])
        #     self.model.to(self.hparams['device'])
        # self.model.load_state_dict(torch.load(model_load_path))
        self.model = torch.load(model_load_path)
        self.model.eval()
        with open(params_load_path, 'rb') as f:
            self.hparams, self.ner_labels_vocab = pickle.load(f)
        
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
        self.crf = CRF(len(NERmodel.ner_labels_vocab), batch_first=batch_first)

    def forward(self, X, y, mask):
        h, z_att = self.rnn(X, mask)
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
        self.word_emb_size = config.get_parameter('word_emb_size')
        self.word_lstm_unit = config.get_parameter('word_lstm_units')
        self.batch_size = config.get_parameter('batch_size')
        self.batch_first = batch_first
        self.use_crf = config.get_parameter('use_crf')
        self.use_word_self_attention= config.get_parameter('use_word_self_attention')
        self.dropout = config.get_parameter('dropout')
        self.num_tags = len(NERmodel.ner_labels_vocab)
        self.device = device
        self.use_GRU = NERmodel.hparams['use_GRU']

        self.num_layers = 1

        if self.use_GRU:
            self.word_rnn = nn.GRU(input_size=self.word_emb_size, hidden_size=self.word_lstm_unit, 
                                    batch_first=self.batch_first, dropout=self.dropout, bidirectional=True)
        else:
            self.word_rnn = nn.LSTM(input_size=self.word_emb_size, hidden_size=self.word_lstm_unit, 
                                            batch_first=self.batch_first, dropout=self.dropout, bidirectional=True)
        
        self.fc = nn.Linear(2*self.word_lstm_unit, self.num_tags)

        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, mask):
        z, _ = self.word_rnn(X)
        mask = mask.unsqueeze(2).expand(-1,-1,z.shape[2]).float()

        # self attention
        att_weight = torch.matmul(z, z.transpose(-1,-2))
        softmax_weight = self.softmax(att_weight)
        z_att = torch.matmul(softmax_weight, z)
        if self.use_word_self_attention:
            z = z_att

        z = z * mask
        z = self.dropout_layer(z)
        z = self.fc(z)

        return z, softmax_weight


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
