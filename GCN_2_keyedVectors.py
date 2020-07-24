import os
import pickle
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Update the path
embedding_txt = '--------/embedding_text.txt'
embedding_temp = '-------/embedding_temp'
embedding_path = '-------/GCN_embedding.pkl'

with open('--------------/GCN_emb.pkl', 'rb') as f:
    word_embeddings = pickle.load(f)
    
with open('--------------/GCN_emb_idx2id_dict.pkl', 'rb') as f:
    idx2id_dict = pickle.load(f)

if not os.path.exists(embedding_txt):
    with open(embedding_txt, 'w') as f:
        for item in idx2id_dict:
            f.write(idx2id_dict[item]+' '+' '.join([str(i.item()) for i in word_embeddings[item]]) + '\n')

glove_file = datapath(embedding_txt)
temp_file = get_tmpfile(embedding_temp)
_ = glove2word2vec(glove_file, temp_file)

wv = KeyedVectors.load_word2vec_format(temp_file)
wv.save(embedding_path)
