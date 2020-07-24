import networkx as nx
import numpy as np 
import random
from tensorboardX import SummaryWriter
import pickle
from node2vec.skipgram import *
import os
import torch
from utils.vocab import Vocabulary
from config.paths import *


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next_ = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next_)
            else:
                break
        
        return walk


    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return           


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
        
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def read_graph(file_path, directed):
    G = nx.read_gpickle(file_path)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    return G


def get_node2vec(vocab_path, embedding_path):

    with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

    wv = KeyedVectors.load(embedding_path)

    return vocab, wv


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

    # ----------- Random walk --------------------
    directed_graph=False

    if not os.path.exists(dump_walks):
        num_walks = 30
        walk_length = 8
        nx_G = read_graph(mesh_graph_file, directed_graph)
        G = Graph(nx_G, is_directed=directed_graph, p=params.p, q=params.q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(num_walks, walk_length)
        with open(dump_walks, 'wb') as f:
            pickle.dump(walks, f)
    else:
        with open(dump_walks, 'rb') as f:
            walks = pickle.load(f)

    # ---------- train SkipGram -----------------
    epochs = params.epochs
    batch_size = params.batch_size
    window = params.window
    num_neg_sample = params.num_neg_sample
    writer = SummaryWriter()

    if os.path.exists(dump_process_pkl):
        with open(dump_process_pkl, 'rb') as f:
            vocab = pickle.load(f)
    else:
        vocab = Vocabulary(lower=False)
        vocab.add_documents(walks)
        vocab.build()

        with open(dump_process_pkl, 'wb') as f:
            pickle.dump(vocab, f)

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

    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # here transformation is required we will directly sample the index
    sample_table = negative_sampling_table(vocab.token_counter(), transform=vocab.token_to_id)
    neg_sample = np.random.choice(sample_table, size=(len(l),num_neg_sample))

    context_data = ContextData(l, d, neg_sample, n_sample=5, transform=None)
    context_dataloader = DataLoader(context_data, batch_size=batch_size, shuffle=True, num_workers=6)

    model_embedding = SkipGram(len(vocab.vocab), embedding_size=1024)
    model_embedding.to(device)
    optimizer_embedding = torch.optim.SparseAdam(model_embedding.parameters(), lr=0.005)

    train(model_embedding, optimizer_embedding, context_dataloader, epochs, device, neg_sample,n_sample=num_neg_sample, transform=None, writer = writer, save_path=save_model_path, l=l, d=d, vocab=vocab, batch_size=batch_size)
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
    wv.save(embedding)

    writer.close()


# if __name__ == '__main__':
#     base_path = '/media/druv022/Data2/Final'
#     paths = Paths(base_path, node2vec_type='1')

#     train_node2vec(paths)