import numpy as np
import math
from collections import Counter
import pickle
import os
from multiprocessing import Pool, Manager
from functools import partial
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from gensim.models import KeyedVectors
from tensorboardX import SummaryWriter
import csv
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.corpus import stopwords 


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

    def token_counter(self):
        return self._token_count

# TODO: support transform to w2i; support better tokenization
def build_context(sent, pdict, plist, p2list, window=2, transform=None):
    context_dict = pdict
    p2list += [len(p2list)]
    print(f'Sentence no: {len(p2list)}')
    if '\n' in sent:
        sent.remove('\n')
    # if transformation is required from word to index, do it
    sent_ = []
    for word in sent:
        if word not in stopwords:
            sent_.append(word)

    if transform:
        sent = transform(sent_)
    else:
        sent = sent_

    for idx, word in enumerate(sent):
        context_ = []
        for w in range(-window, window+1):
            crnt_pos = w+idx

            if crnt_pos < 0 or crnt_pos == idx or crnt_pos > len(sent)-1:
                continue
            plist += [(word, sent[crnt_pos])]
            context_.append(sent[crnt_pos])

        if word in context_dict:
            value = context_dict[word]
            context_dict[word] = list(set(value + context_))
        else:
            context_dict[word] = context_            


def multiprocess(sentences, threads=100, window=2, transform=None):
    manager = Manager()
    d = manager.dict()
    l1 = manager.list()
    l2 = manager.list()

    p = Pool(processes=threads)
    func = partial(build_context, pdict=d, plist=l1, p2list=l2, window=window, transform=transform)
    p.map(func, sentences)

    return list(l1), dict(d)

def get_one_hot(indxs, vocab_size, batch_size=1):
    x = np.zeros((batch_size, vocab_size))
    for i, indx in enumerate(indxs):
        x[i,indx] = 1.0
    return x

def get_context_data(context_list, context_dict, vocab_size, neg_samples, transform=None, n_sample=5, batch_size=1, device='cpu'):
    x = [] # center word 
    y = [] # context word 
    z = [] # negative word

    count = 0
    for i, item in enumerate(context_list):
        if transform:
            key_ = transform(item[0])
        else:
            key_ = item[0]
        
        x.append([key_])#*(n_sample))
        if transform:
            y.append([transform(item[1])])#*(n_sample))
        else:
            y.append([item[1]])
        
        z.append(neg_samples[i])

        count += 1
        if count == batch_size :
            if batch_size == 1:
                count = 0
                yield torch.tensor(x, device=device).unsqueeze(0), torch.tensor(y, device=device).unsqueeze(0), torch.tensor(z, device=device).unsqueeze(0)
            else:
                count = 0
                yield torch.tensor(x, device=device), torch.tensor(y, device=device), torch.tensor(z, device=device)


def negative_sampling_table(counter_data, transform=None, num_samples=5):
    table = []
    table_size = sum(counter_data.values())

    u = np.array(list(counter_data.values()))**0.75
    z = sum(u)
    p = u/z

    count = np.round(p*table_size)

    for i, key in enumerate(counter_data):
        c = count[i]
        indx = transform(key)
        table += [indx]*np.int(c)

    random.shuffle(table)
    return table

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(SkipGram, self).__init__()
        
        self.in_embedding = nn.Embedding(vocab_size, embedding_size, sparse=True)
        self.out_embedding = nn.Embedding(vocab_size, embedding_size, sparse=True)

    def forward(self, c_word, p_word, n_word):
        c_embed = self.in_embedding(c_word) # shape: batch_size, 1, embedding_size
        p_embed = self.out_embedding(p_word) # shape: batch_size, 1, embedding_size
        n_embed = self.out_embedding(n_word) # shape: batch_size, neg_samples, embedding_size

        p_score = torch.bmm(p_embed, c_embed.transpose(-1,-2)) # shape: batch_size, 1, 1
        p_score = nn.functional.logsigmoid(p_score) # # shape: batch_size, 1, 1

        n_score = torch.bmm(n_embed, c_embed.transpose(-1,-2)) # shape: batch_size, neg_samples, 1
        n_score = nn.functional.logsigmoid(-1*n_score) # shape: batch_size, neg_samples, 1
        n_score = n_score.sum(dim=1, keepdim=True) # shape: batch_size, 1, 1

        combine_loss = torch.mean(p_score + n_score) # shape: []

        return - combine_loss

class ContextData(Dataset):
    def __init__(self, context_list, context_dict, neg_samples, n_sample = 5, transform=None):
        super(ContextData, self).__init__()
        self.keys = list(context_dict.keys())
        self.context_dict = context_dict
        self.context_list = context_list
        self.sample_table = sample_table
        self.transform = transform
        self.n_sample = n_sample
        self.total = len(self.context_list)
        self.neg_samples = neg_samples
        self.count = 0

    def __len__(self):
        return len(self.context_list)

    def __getitem__(self, idx):
        item_0, item_1 = self.context_list[idx]

        x = [] # center word 
        y = [] # context word 
        z = [] # negative word
        if self.transform:
            key_ = self.transform(item_0)
        else:
            key_ = item_0

        x.append(key_)
        if self.transform:
            y.append(self.transform(item_1))
        else:
            y.append(item_1)

        z.extend(self.neg_samples[idx])

        self.count += 1
        if self.count == self.total:
            self.count = 0
            np.random.shuffle(self.neg_samples)

        return torch.tensor(x), torch.tensor(y), torch.tensor(z)

def train(model, optimizer, context_dataloader, epochs, device, neg_samples, n_sample=5, transform=None):
    print('Training started:...')
    counter = 0
    for epoch in range(epochs):
        epoch_loss = 0
        count = 0
        start_time = time.time()
        # for x, y, z in get_context_data(l, d, len(vocab.vocab), neg_samples, transform=transform,
        #                                  n_sample=n_sample, batch_size=batch_size, device=device): # use for debugging
        for x, y, z in context_dataloader:
            x, y, z = x.to(device), y.to(device), z.to(device)
            optimizer.zero_grad()
            loss = model(x, y, z)
            loss.backward()
            optimizer.step()
            count += 1
            counter += 1
            epoch_loss += loss.item()
            print(count,'\t', time.time() - start_time)

        print(f'Epoch: {epoch}\tLoss: {epoch_loss/count}\tTime: {time.time() - start_time}')
        writer.add_scalars('loss', {'loss': epoch_loss/count}, epoch)

        torch.save(model.state_dict(), os.path.join(save_model_path,'sk_model5_5.pkl'))

def discard(word, token_counter, t=0.001):
    f = token_counter[word]
    rand_p = random.random()

    p = 1 - math.sqrt(t/f)

    if rand_p > p:
        return True
    else:
        return False

if __name__ == '__main__':
    # sentences = [['this', 'is', 'a', 'sentence', 'which', 'is', 'very', 'long'],['this', 'is', 'another', 'sentence']]
    # sentences = [[1, 2, 3, 4, 5, 2, 6, 7],[1, 2, 8, 4]]
    # context_dict = build_context(sentences)

    training_data = r'/media/druv022/Data1/Masters/Thesis/Data/Skipgram/hansards/training.en'
    dump_process_pkl = r'/media/druv022/Data1/Masters/Thesis/Data/Skipgram/hansards/processed_en_w.pkl'
    dump_context_dict = r'/media/druv022/Data1/Masters/Thesis/Data/Skipgram/hansards/context_dict_w.pkl'
    dump_context_list = r'/media/druv022/Data1/Masters/Thesis/Data/Skipgram/hansards/context_list_w.pkl'
    save_model_path = r'/media/druv022/Data1/Masters/Thesis/Data/Skipgram/hansards'
    embedding_txt = r'/media/druv022/Data1/Masters/Thesis/Data/Skipgram/hansards/embedding.txt'
    embedding_temp = r'/media/druv022/Data1/Masters/Thesis/Data/Skipgram/hansards/embedding_temp.txt'
    epochs = 20
    batch_size=2**10
    window = 5
    num_neg_sample = 5
    writer = SummaryWriter()
    stopwords = set(stopwords.words('english'))

    with open(training_data, 'r') as f:
        data = f.readlines()
        data = [line.replace('\n','').split(' ') for line in data]
        data = [[word for word in line if word not in stopwords] for line in data]

    if os.path.exists(dump_process_pkl):
        with open(dump_process_pkl, 'rb') as f:
            vocab = pickle.load(f)
    else:
        vocab = Vocabulary()
        vocab.add_documents(data)
        vocab.build()

        with open(dump_process_pkl, 'wb') as f:
            pickle.dump(vocab, f)

    # use transformation only once, i.e either during creating the context dict and list or during training
    if not os.path.exists(dump_context_dict):
        l, d = multiprocess(data, window=window, transform=vocab.doc2id)
        with open(dump_context_dict, 'wb') as f:
            pickle.dump(d, f)
        with open(dump_context_list, 'wb') as f:
            pickle.dump(l, f)
    else:
        with open(dump_context_dict, 'rb') as f:
            d = pickle.load(f)
        with open(dump_context_list, 'rb') as f:
            l = pickle.load(f)

    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # here transformation is required we will directly sample the index
    sample_table = negative_sampling_table(vocab.token_counter(), transform=vocab.token_to_id, num_samples=num_neg_sample)
    neg_sample = np.random.choice(sample_table, size=(len(l),num_neg_sample))

    
    context_data = ContextData(l, d, neg_sample, n_sample=5, transform=None)
    context_dataloader = DataLoader(context_data, batch_size=batch_size, shuffle=True, num_workers=6)

    model_embedding = SkipGram(len(vocab.vocab), embedding_size=200)
    model_embedding.load_state_dict(torch.load(os.path.join(save_model_path,'sk_model5_5.pkl')))
    model_embedding.to(device)
    optimizer_embedding = torch.optim.SparseAdam(model_embedding.parameters(), lr=0.005)

    # train(model_embedding, optimizer_embedding, context_dataloader, epochs, device, neg_sample,n_sample=num_neg_sample)
    word_embeddings = (model_embedding.out_embedding.weight.data + model_embedding.in_embedding.weight.data)/2
    word_embeddings = word_embeddings.cpu().numpy()

    sorted_vocab_tuple = sorted(vocab.vocab.items(), key=lambda kv: kv[1])

    with open(embedding_txt, 'w') as f:
        for idx,item in enumerate(sorted_vocab_tuple):
            if item[0] == '\n':
                continue
            f.write(item[0]+' '+' '.join([str(i) for i in word_embeddings[idx]]) + '\n')

    glove_file = datapath(embedding_txt)
    temp_file = get_tmpfile(embedding_temp)
    _ = glove2word2vec(glove_file, temp_file)

    wv = KeyedVectors.load_word2vec_format(temp_file)

    result = wv.most_similar(positive=['woman', 'king'], negative=['man'])
    print("{}: {:.4f}".format(*result[0]))

    writer.close()
    print("Boom")




    
