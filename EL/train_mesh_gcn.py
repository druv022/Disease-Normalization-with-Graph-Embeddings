# This is an experiment file

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
import networkx as nx
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from embeddings import load_embedding_pkl
import pickle
import os
from scipy import sparse
from models.summery import SummeryCNN, SummeryRNN
from models.gcn import GCN, FC
import math
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import time
from tensorboardX import SummaryWriter
from allennlp.modules.elmo import Elmo, batch_to_ids
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

class MESH(object):

    def __init__(self,mesh_h, scope_note, entry_terms, unique_id, other=None):
        self._scope_note = scope_note
        self._entry_terms = entry_terms
        self._unique_id = unique_id
        self._mesh_h = mesh_h
        self._other =other

    def __str__(self):
        return self._mesh_h

    def __eq__(self, other):
        return self._mesh_h == other.mesh_h

    def object_to_string(self):
        return str(self.__dict__)
    
    @property
    def scope_note(self):
        return self._scope_note

    @property
    def entry_terms(self):
        return self._entry_terms
    
    @property
    def unique_id(self):
        return self._unique_id

    @property
    def mesh_h(self):
        return self._mesh_h

    @property
    def other(self):
        return self._other

def read_mesh_file(filepath):

    with open(filepath, 'r', encoding='utf8') as f:
        data = f.readlines()

    mesh_dict = {}
    add_recod = False

    scope_note = ''
    entry_terms = []
    unique_id = ''
    mesh_h = ''
    other = []

    for line in data:
        if line.replace('\n','') == '*NEWRECORD':
            if add_recod:
                mesh_dict[unique_id] = MESH(mesh_h, scope_note, entry_terms, unique_id, other=other)
                add_recod = False
            scope_note = ''
            entry_terms = []
            unique_id = ''
            mesh_h = ''
            other = []
            continue

        if line == '\n':
            continue

        eq_indx = line.find('=')

        label = line[:eq_indx]
        label = label.strip(' ')

        value = line[eq_indx+2:].strip('\n')

        if label == 'MH':
            mesh_h = value
            add_recod = True
            value_split = value.split(',')
            value_split.reverse()
            value_split[0] = value_split[0].strip()
            value_split = ' '.join(value_split)
            mesh_h = value_split
            entry_terms.append(value_split.split())
        elif label == 'ENTRY' or label == 'PRINT ENTRY':
            value = value.split('|')[0]
            value_split = value.split(',')
            value_split.reverse()
            value_split[0] = value_split[0].strip()
            value_split = ' '.join(value_split)
            value_split = value_split.split()
            entry_terms.append(value_split)
        elif label == 'FX':
            other.append(value)
        elif label == 'MS':
            # TODO: use proper tokenizer
            scope_note_sentences = value.split('.')
            scope_note = [[r'<s>']+ sent.split()+[r'<\s>'] for sent in scope_note_sentences]
        elif label == 'UI':
            unique_id = value
    
    # add last entry
    if add_recod:
        mesh_dict[unique_id] = MESH(mesh_h, scope_note, entry_terms, unique_id, other=other)

    return mesh_dict

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
            use_char: boolean. Whether to use char feature for scope note.
            num_norm: boolean. Whether to normalize text.
            initial_vocab: Iterable. Initial vocabulary for expanding word_vocab.
        """
        self._num_norm = num_norm
        self._use_char = use_char
        
        # TODO: check how to use this
        self._node_vocab = Vocabulary(lower=False)
        self._word_vocab = Vocabulary(lower=lower)
        self._char_vocab = Vocabulary(lower=False)
        #TODO: check usability
        self._label_vocab = Vocabulary(lower=False, unk_token=False)

        if initial_vocab:
            self._word_vocab.add_documents([initial_vocab])
            self._char_vocab.add_documents(initial_vocab)

    def fit(self, X, y=None):
        """Learn vocabulary from training set.

        Args:
            X : iterable. An iterable which yields either str, unicode or file objects.

        Returns:
            self : IndexTransformer.
        """
        for input_data in X:
            self._node_vocab.add_node(input_data[0])
            self._word_vocab.add_document(input_data[1])
            if self._use_char:
                self._char_vocab.add_documents(input_data[1])
            for data in input_data[2]:
                self._word_vocab.add_document(data)
                if self._use_char:
                    self._char_vocab.add_documents(data)
                # self._label_vocab.add_node(' '.join(data)) # this results in a very big lable space (90K) 
                self._label_vocab.add_document(data) # Use word indexing instead, drawbacks: BOW

        self._node_vocab.build()
        self._word_vocab.build()
        self._char_vocab.build()
        self._label_vocab.build()

        return self

    # TODO: revisit
    def transform(self, nodelist, word_list, labels):
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
        word_ids = []
        node_ids = []
        entry_ids = []
        # first entry is pad so skipping index 0
        for index in range(1,len(nodelist)):
            node_ids.append(self._node_vocab.token_to_id(nodelist[index]))
            word_ids.append(torch.tensor(self._word_vocab.doc2id(word_list[index])))
            # if self._use_char:
            #     char_ids.append([self._char_vocab.doc2id(w) for w in input_data[1]])

            entry_ = [self._label_vocab.doc2id(label) for label in labels[index]]
            unique_entry = []
            for i in entry_:
                unique_entry.extend(i)
            entry_ids.append(list(set(unique_entry)))

        word_ids = nn.utils.rnn.pad_sequence(word_ids, batch_first=True)
        return [node_ids, word_ids, entry_ids]

    # def fit_transform(self, X, y=None, **params):
    #     """Learn vocabulary and return document id matrix.

    #     This is equivalent to fit followed by transform.

    #     Args:
    #         X : iterable
    #         an iterable which yields either str, unicode or file objects.

    #     Returns:
    #         list : document id matrix.
    #         list: label id matrix.
    #     """
    #     return self.fit(X, y).transform(X, y)

    # def inverse_transform(self, y, lengths=None):
    #     """Return label strings.

    #     Args:
    #         y: label id matrix.
    #         lengths: sentences length.

    #     Returns:
    #         list: list of list of strings.
    #     """
    #     y = np.argmax(y, -1)
    #     inverse_y = [self._label_vocab.id2doc(ids) for ids in y]
    #     if lengths is not None:
    #         inverse_y = [iy[:l] for iy, l in zip(inverse_y, lengths)]

    #     return inverse_y

    @property
    def word_vocab_size(self):
        return len(self._word_vocab)

    @property
    def char_vocab_size(self):
        return len(self._char_vocab)

    @property
    def node_vocab_size(self):
        return len(self._node_vocab)

    @property
    def label_size(self):
        return len(self._label_vocab)

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)

        return p


def pad_nested_sequences_2(sequences, dtype='int32'):
    """Pads nested sequences to the same length.

    This function transforms a list of list sequences
    into a 3D Numpy array of shape `(num_samples, max_sent_len, max_word_len)`.

    Args:
        sequences: List of lists of lists.
        dtype: Type of the output sequences.

    # Returns
        x: Numpy array.
    """
    max_item_len = 0
    max_sent_len = 0
    max_word_len = 0
    for item in sequences:
        max_item_len = max(len(item), max_item_len)
        for sent in item:
            max_sent_len = max(len(sent), max_sent_len)
            for word in sent:
                max_word_len = max(len(word), max_word_len)

    x = np.zeros((len(sequences), max_item_len, max_sent_len, max_word_len)).astype(dtype)
    for i, item in enumerate(sequences):
        for j, sent in enumerate(item):
            for k, word in enumerate(sent):
                x[i, j, k, :len(word)] = word

    return x

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

    def add_document(self, doc):
        sent = map(self.process_token, doc)
        self._token_count.update(sent)

    def add_node(self, node):
        self._token_count.update([node])


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
            _embeddings[word_idx] = embeddings[word]

    return _embeddings

def get_training_formated_data(mesh_dict, diseases):
    # list [node, scope_word, entry_char]
    train_data = []
    for disease_id in diseases:
        if '\n' in disease_id:
            disease_id = disease_id.strip('\n')
        entry = []
        entry.append(mesh_dict[disease_id].unique_id)
        entry.append(mesh_dict[disease_id].scope_note)
        entry.append(mesh_dict[disease_id].entry_terms)
        train_data.append(entry)

    return train_data


def get_word_embeddings(file_name, vocab, embedding_dim):
    embeddings = load_embedding_pkl(file_name)
    word_embeddings = filter_embeddings(embeddings, vocab=vocab, dim=embedding_dim)
    return torch.from_numpy(word_embeddings)


def get_adjacancy_matrix(mesh_graph, node_list):
    # get adjancy matrix
    adj_matrix = nx.linalg.graphmatrix.adjacency_matrix(mesh_graph, nodelist=node_list)
    adj_matrix = adj_matrix.toarray() + np.identity(len(node_list))

    laplacian_matrix = nx.linalg.laplacianmatrix.laplacian_matrix(mesh_graph, nodelist=node_list)
    laplacian_matrix = laplacian_matrix.toarray()

    d_matrix = laplacian_matrix + adj_matrix

    adj_matrix_compressed = sparse.csr_matrix(adj_matrix)

    del laplacian_matrix
    del adj_matrix

    d_matrix = np.linalg.inv(d_matrix)
    d_matrix = np.sqrt(d_matrix)
    prod = np.matmul(d_matrix, adj_matrix_compressed.toarray())

    return np.matmul(prod, d_matrix)


def minibatch(word_ids, batch_size=1):
    length_x = len(word_ids)
    for idx in range(0, math.ceil(length_x/batch_size)):
        if length_x - batch_size < 0:
            updated_size = length_x
        else:
            updated_size = batch_size
        length_x = length_x - batch_size
        batch_x = word_ids[idx*batch_size:idx*batch_size+updated_size]

        yield torch.stack(batch_x)

def calculate_accuracy(output, target):
    correct = 0
    false_pred = 0
    total_target = 0
    total_pred = 0
    acc = []
    fp = []
    for idx in range(len(output)):
        pred_idx = np.nonzero(output[idx])
        target_idx = np.nonzero(target[idx])
        correct_pred= sum([target_idx[0][i] in pred_idx[0] for i in range(len(target_idx[0]))])
        correct += correct_pred
        false_pred += len(pred_idx[0]) - correct_pred
        total_target += len(target_idx[0])
        total_pred += len(pred_idx[0])

        # acc.append(correct_pred/len(target_idx[0]))
        # fp.append((len(pred_idx[0]) - correct_pred/len(pred_idx[0]))/len(pred_idx[0]) if len(pred_idx[0]) > 0 else 0.0)

    return correct/total_target, false_pred/total_pred if total_pred > 0 else 0.0
    # return sum(acc)/len(output), sum(fp)/len(output)

def get_weight(labels):
    positive_labels = torch.sum(labels, dim=0)
    total_labels = labels.shape[0]

    # positive_labels = labels.sum()
    # total_labels = len(labels.view(-1))

    return (total_labels - positive_labels)/positive_labels


def train_GCN():
    labels = torch.tensor(MultiLabelBinarizer().fit_transform(entry_ids), dtype=torch.float32, device=device)
    labels_weight = get_weight(labels)

    n_class = p.label_size-1 # p.label_size-1
    # labels = torch.empty(labels.shape[0], n_class).uniform_(0,1)
    # labels = torch.bernoulli(labels).to(device)

    
    # y = torch.empty(labels.shape[0],1, dtype=torch.long, device=device).random_() % n_class
    # labels = torch.empty(labels.shape[0],n_class, dtype=torch.float32, device=device)
    # labels.zero_()
    # labels.scatter_(1,y,1)


    # check if adjacancy matrix already calculated
    if not os.path.exists(os.path.join(mesh_folder, 'a_hat_matrix')):
        a_hat = get_adjacancy_matrix(mesh_graph, node_list[1:])

        # save node_list and the calculated adjacancy matrix
        with open(os.path.join(mesh_folder, 'node_list'), 'wb') as f:
            pickle.dump(node_list, f)
        data = sparse.coo_matrix(a_hat)
        with open(os.path.join(mesh_folder, 'a_hat_matrix'), 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(os.path.join(mesh_folder, 'a_hat_matrix'), 'rb') as f:
            data = pickle.load(f)

    i = torch.tensor([data.row, data.col], dtype=torch.long, device=device)
    v = torch.tensor(data.data, dtype=torch.float32, device=device)
    a_hat = torch.sparse.FloatTensor(i,v, torch.Size([len(node_list)-1, len(node_list)-1])).cuda()

    # summery_model = SummeryCNN(p.word_vocab_size, word_embeddings, embedding_size=word_embedding_size, output_size=word_embedding_size, 
    #                             padidx=p._word_vocab.vocab['<pad>'])
    # summery_model = SummeryRNN(p.word_vocab_size, word_embeddings, embedding_size=word_embedding_size, output_size=word_embedding_size, 
    #                             padidx=p._word_vocab.vocab['<pad>'])
    gcn_model = GCN(nfeat=word_embedding_size, nhid=512, dropout=0.0)
    fc_model = FC(word_embedding_size, n_class, dropout=0.5) # p.label_size-1
    
    # summery_model.to(device)
    gcn_model.to(device)
    fc_model.to(device)

    optimizer = torch.optim.Adam( list(gcn_model.parameters()) + list(fc_model.parameters()), lr=0.001) # weight_decay=0.01; list(summery_model.parameters()) +

    # optimizer = torch.optim.Adam(list(summery_model.parameters()) + list(fc_model.parameters()), lr=0.001)

    if use_elmo:
        ELMO_folder = r'/media/druv022/Data1/Masters/Thesis/Data/ELMO'
        with open(os.path.join(ELMO_folder, 'elmo_scope_weights'), 'rb') as f:
            elmo_scope_embeddings = pickle.load(f)
        with open(os.path.join(ELMO_folder, 'id2idx_dict'), 'rb') as f:
            id2idx_dict = pickle.load(f)

        scope_embds = []
        for i in node_list:
            if i in id2idx_dict:
                scope_embds.append(torch.sum(elmo_scope_embeddings[id2idx_dict[i]], dim=0))

            if i == '<unk>':
                scope_embds.append(torch.zeros(word_embedding_size, device=device))

        scope_embds = torch.stack(scope_embds)
    
    best_z = []
    best_acc = 0

    epochs = 100
    batch_size = 256
    for epoch in range(epochs):
        start_time = time.time()
        if not use_elmo:

            optimizer.zero_grad()

            idxs = np.arange(0,len(node_ids))
            np.random.shuffle(idxs)
            word_ids_shuffled = [word_ids[i] for i in list(idxs)]

            embeddings = []
            for x in minibatch(word_ids_shuffled, batch_size=batch_size):
                x = x.to(device)
                z = summery_model(x)
                embeddings.extend(z)

            scope_embds = list(np.arange(0, len(node_ids)))
            for i,idx in enumerate(idxs):
                scope_embds[idx] = embeddings[i]

            scope_embds = torch.stack(scope_embds)

        z_, x = gcn_model(scope_embds, a_hat)

        z = fc_model(z_)
        # z = fc_model(scope_embds)

        # loss1 = nn.functional.binary_cross_entropy_with_logits(z[0:5000],labels[0:5000])
        # loss2 = nn.functional.binary_cross_entropy_with_logits(z[5000:10000],labels[5000:10000])
        # loss3 = nn.functional.binary_cross_entropy_with_logits(z[10000:],labels[10000:])
        # loss = (loss1 + loss2 + loss3)/3

        loss = nn.functional.binary_cross_entropy_with_logits(z,labels, reduction='mean', pos_weight=labels_weight) #, pos_weight=labels_weight

        # loss = nn.functional.cross_entropy(z,y.view(-1), reduction='mean')
        loss.backward()
        optimizer.step()

        writer.add_scalars('loss', {'loss': loss.item()}, epoch)

        with torch.no_grad():
            pred_labels = (torch.sigmoid(z.data) > 0.5)
            # pred_labels = nn.functional.softmax(z,dim=-1).detach()
            # _, pred_idx = torch.max(pred_labels,-1)
            # pred_labels = torch.empty(labels.shape[0],n_class, dtype=torch.float32, device=device)
            # pred_labels.zero_()
            # pred_labels.scatter_(1,pred_idx.view(-1,1),1)

            true_labels = labels.data
            if epoch % 10 == 0:
                print(classification_report(true_labels.cpu().numpy(), pred_labels.cpu().numpy()))

            acc, fp = calculate_accuracy(pred_labels.cpu().numpy(), true_labels.cpu().numpy())
            print(f'Epoch: {epoch}\t Loss: {loss.item()}\t time: {time.time() - start_time}\tAccuracy: {acc}\t False positive: {fp}')
            
            if acc > best_acc:
                best_acc = acc
                # best_z = z_
                # # torch.save(summery_model.state_dict(),os.path.join(mesh_folder,'summery_model'))
                # torch.save(gcn_model.state_dict(),os.path.join(mesh_folder,'gcn_model'))

                # with open(os.path.join(mesh_folder,'GCN_mesh_embedding'), 'wb') as f:
                #     pickle.dump(x, f)
                
                # print('Saving Best model...')

            print('Best ACC: ', best_acc)

    
    embedding_txt = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/GCN/embedding.txt'
    embedding_temp = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/GCN/embedding_temp.txt'
    embedding = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/GCN/embedding'
    
    with open(embedding_txt, 'w') as f:
        for idx,i in enumerate(node_list[1:]):
            f.write(i+' '+' '.join([str(i) for i in z_[idx].cpu().detach().numpy().tolist()]) + '\n')

    glove_file = datapath(embedding_txt)
    temp_file = get_tmpfile(embedding_temp)
    _ = glove2word2vec(glove_file, temp_file)

    wv = KeyedVectors.load_word2vec_format(temp_file)
    wv.save(embedding)

    print('Complete')

if __name__ == '__main__':
    mesh_graph_file = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/mesh_graph_disease'
    mesh_file = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/ASCIImeshd2019.bin'
    disease_file= r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/disease_list'
    all_D_file= r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/all_D_list'
    edge_node_list = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/edge_node_list'
    path_to_embeddings = r'/media/druv022/Data1/Masters/Thesis/Data/Embeddings'
    mesh_folder = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH'

    directed_graph=False

    np.random.seed(5)
    torch.manual_seed(5)
    random.seed(5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    mesh_graph = nx.read_gpickle(mesh_graph_file)
    mesh_graph = mesh_graph.to_undirected()

    # read disease file
    with open(disease_file,'r') as f:
        data = f.readlines()

    p = IndexTransformer()

    mesh_dict = read_mesh_file(mesh_file)
    # check if preprocessing is already completed
    if not os.path.exists(os.path.join(mesh_folder,'processor')):
        train_data = get_training_formated_data(mesh_dict, data)
        p.fit(train_data)
        p.save(os.path.join(mesh_folder,'processor'))
    else:
        p = p.load(os.path.join(mesh_folder,'processor'))

    word_embedding_size = 200
    word_embeddings = get_word_embeddings(path_to_embeddings, p._word_vocab.vocab, word_embedding_size)

    # Get the list of nodes (idx 0 is '<pad>')
    node_list = list(p._node_vocab.vocab)

    if not os.path.exists(os.path.join(mesh_folder, 'processed_tr_data')):
        scope_note = []
        for i in node_list:
            note = []
            if i in mesh_dict:
                for sent in mesh_dict[i].scope_note:
                    note.extend(sent[1:-2])
            else:
                note = ['<pad>']
            scope_note.append(note)
        labels = [mesh_dict[i].entry_terms if i in mesh_dict else ['<pad>'] for i in node_list]
        node_ids, word_ids, entry_ids = p.transform(node_list, scope_note, labels)
        with open(os.path.join(mesh_folder, 'processed_tr_data'), 'wb') as f:
            pickle.dump([node_ids, word_ids, entry_ids], f)
    else:
        with open(os.path.join(mesh_folder, 'processed_tr_data'), 'rb') as f:
            node_ids, word_ids, entry_ids = pickle.load(f)

    use_elmo = True
    if use_elmo:
        word_embedding_size = 1024
    # train GCN
    # TODO: use arguments
    train_GCN()


    writer.close()
    

