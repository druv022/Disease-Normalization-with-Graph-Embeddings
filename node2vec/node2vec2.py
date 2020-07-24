import networkx as nx
import numpy as np 
import random
from tensorboardX import SummaryWriter
import pickle
from node2vec.skipgram import *
import os
import torch
import torch.nn as nn
from node2vec.node2vec1 import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
import networkx as nx
from allennlp.modules.elmo import Elmo, batch_to_ids
from utils.vocab import Vocabulary
from utils.mesh import *


class FC(nn.Module):
    def __init__(self, in_features, out_features):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


class SkipGramModified(nn.Module):
    def __init__(self, vocab_size, embedding_size, weight=None):
        super(SkipGramModified, self).__init__()
        
        self.in_embedding = nn.Embedding(vocab_size, embedding_size, sparse=False)
        self.in_embedding.from_pretrained(weight, freeze=False)
        self.out_embedding = nn.Embedding(vocab_size, embedding_size, sparse=False)
        self.out_embedding.from_pretrained(weight, freeze=False)

        # self.fc = nn.Linear(embedding_size, embedding_size)

    def forward(self, c_word, p_word, n_word):
        c_embed = self.in_embedding(c_word.long()) # shape: batch_size, 1, embedding_size
        # c_embed = self.fc(c_embed)

        p_embed = self.out_embedding(p_word.long()) # shape: batch_size, 1, embedding_size
        # p_embed = self.fc(p_embed)

        n_embed = self.out_embedding(n_word.long()) # shape: batch_size, neg_samples, embedding_size
        # n_embed = self.fc(n_embed)

        return c_embed, p_embed, n_embed

def loss_fn(c_embed, p_embed, n_embed):
    p_score = torch.bmm(p_embed, c_embed.transpose(-1,-2)) # shape: batch_size, 1, 1
    p_score = nn.functional.logsigmoid(p_score) # # shape: batch_size, 1, 1

    n_score = torch.bmm(n_embed, c_embed.transpose(-1,-2)) # shape: batch_size, neg_samples, 1
    n_score = nn.functional.logsigmoid(-1*n_score) # shape: batch_size, neg_samples, 1
    n_score = n_score.sum(dim=1, keepdim=True) # shape: batch_size, 1, 1

    combine_loss = torch.mean(p_score + n_score) # shape: []
    return -combine_loss


def train(model1, optimizer1, context_dataloader, epochs, device, neg_samples, n_sample=5, transform=None, writer=None, save_path = None, l=None, d=None, vocab=None, batch_size=None):
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
            optimizer1.zero_grad()
            # optimizer2.zero_grad()
            x, y, z = model1(x, y, z)
            # x, y, z = model2(x), model2(y), model2(z)

            loss = loss_fn(x, y, z)
            loss.backward()
            # optimizer2.step()
            optimizer1.step()
            count += 1
            counter += 1
            epoch_loss += loss.item()

        print(f'Epoch: {epoch}\tLoss: {epoch_loss/count}\tTime: {time.time() - start_time}')
        writer.add_scalars('loss', {'loss': epoch_loss/count}, epoch)

        torch.save(model1.state_dict(), os.path.join(save_path,'model_emb.pkl'))
        # torch.save(model2.state_dict(), os.path.join(save_path,'model_fc.pkl'))


def train_node2vec(paths, params):
    dump_process_pkl = paths.dump_process
    dump_context_dict = paths.dump_context_dict
    dump_context_list = paths.dump_context_list
    dump_walks = paths.dump_walks
    save_model_path = paths.node2vec_base
    embedding_txt = paths.embedding_text
    embedding_temp = paths.embedding_temp
    embedding = paths.embedding
    mesh_graph_file = paths.MeSH_graph_disease

    if not params.randomize:
        np.random.seed(5)
        torch.manual_seed(5)
        random.seed(5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    # ----------- Random walk --------------------
    directed_graph=False

    if not os.path.exists(dump_walks):
        num_walks = 30
        walk_length = 10
        nx_G = read_graph(mesh_graph_file, directed_graph)
        G = Graph(nx_G, is_directed=directed_graph, p=params.p, q=params.q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(num_walks, walk_length)
        with open(dump_walks, 'wb') as f:
            pickle.dump(walks, f)
    else:
        with open(dump_walks, 'rb') as f:
            walks = pickle.load(f)

    if os.path.exists(dump_process_pkl):
        with open(dump_process_pkl, 'rb') as f:
            vocab = pickle.load(f)
    else:
        vocab = Vocabulary(lower=False)
        vocab.add_documents(walks)
        vocab.build()

        with open(dump_process_pkl, 'wb') as f:
            pickle.dump(vocab, f)

    # ---------- build embedding model ----------
    mesh_file = paths.MeSH_file
    ELMO_folder = paths.elmo_folder
    options_file = paths.elmo_options
    weight_file = paths.elmo_weights
    
    elmo = Elmo(options_file, weight_file, 2,dropout=0)
    elmo.to(device)

    mesh_graph = nx.read_gpickle(mesh_graph_file)
    mesh_graph = mesh_graph.to_undirected()

    mesh_dict = read_mesh_file(mesh_file)

    # Get the list of nodes (idx 0 is '<pad>')
    node_list = list(vocab.vocab.keys())

    # create weight matrix by using node_list order(which correspond to original vocab index order)
    elmo_embedding_dim = 1024
    if not os.path.exists(os.path.join(ELMO_folder, 'elmo_weights')):
        weight_list = []
        for idx,i in enumerate(node_list):
            if i in mesh_dict:
                node_idx = vocab.token_to_id(i)
                scope_note = mesh_dict[i].scope_note
                character_ids = batch_to_ids(scope_note).to(device)
                elmo_embeddings = elmo(character_ids)
                embeddings = elmo_embeddings['elmo_representations'][0]
                mask = elmo_embeddings['mask']
                embeddings = embeddings * mask.unsqueeze(2).expand(mask.shape[0], mask.shape[1], embeddings.shape[2]).float()
                embeddings = embeddings.mean(dim=0).mean(dim=0) # average 
                weight_list.append(embeddings.cpu())
            else:
                weight_list.append(torch.zeros(elmo_embedding_dim))

        with open(os.path.join(ELMO_folder, 'elmo_weights'), 'wb') as f:
            pickle.dump(weight_list, f)
    else:
        with open(os.path.join(ELMO_folder, 'elmo_weights'), 'rb') as f:
            weight_list = pickle.load(f)
    
    weight = torch.stack(weight_list, dim=0)

    # ---------- train SkipGram -----------------
    epochs = params.epochs
    batch_size= params.batch_size
    window = params.window
    num_neg_sample = params.num_neg_sample
    writer = SummaryWriter()

    # use transformation only once, i.e either during creating the context dict and list or during training
    if not os.path.exists(dump_context_dict):
        l, d = multiprocess(walks, window=window, transform=vocab.doc2id)
        with open(dump_context_dict, 'wb') as f:
            pickle.dump(d, f)
        with open(dump_context_list, 'wb') as f:
            pickle.dump(l, f)
    else:
        with open(dump_context_dict, 'rb') as f:
            d = pickle.load(f)
        with open(dump_context_list, 'rb') as f:
            l = pickle.load(f)

    # here transformation is required we will directly sample the index
    sample_table = negative_sampling_table(vocab.token_counter(), transform=vocab.token_to_id)
    neg_sample = np.random.choice(sample_table, size=(len(l),num_neg_sample))

    context_data = ContextData(l, d, neg_sample, n_sample=5, transform=None)
    context_dataloader = DataLoader(context_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=6)

    model_embedding = SkipGramModified(len(vocab.vocab), embedding_size=elmo_embedding_dim, weight=weight)
    model_embedding.to(device)
    optimizer_FC = torch.optim.Adam(list(model_embedding.parameters()), lr=0.005) #+list(model_fc.parameters()

    train(model_embedding, optimizer_FC, context_dataloader, epochs, device, neg_sample,n_sample=num_neg_sample, writer = writer, save_path=save_model_path,
            l=l, d=d, vocab=vocab, batch_size=batch_size)

    node_idx = []
    for item in node_list:
        node_idx.append(vocab.token_to_id(item))

    x = torch.tensor(node_idx, device=device)
    y = torch.zeros(x.shape, device=device)
    z = torch.zeros(x.shape, device=device)

    x, y, z = model_embedding(x, y, z)

    word_embeddings = x.cpu().detach().numpy()

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
    wv.save(embedding)

    writer.close()

if __name__ == '__main__':

    main()



