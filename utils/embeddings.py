import wvlib_master.wvlib as wvlib
import pickle
import os
import numpy as np 

def save_embedding_pkl(path):

    data = wvlib.load(os.path.join(path,'PubMed-and-PMC-w2v.bin'))
    data_save = data.word_to_vector_mapping()

    with open(os.path.join(path,'embedding_dump.pkl'),'wb') as f:
        pickle.dump(data_save,f)

def load_embedding_pkl(path):
    file_path = os.path.join(path,'embedding_KeyedVectors.pkl')
    if not os.path.exists(file_path):
        file_path = os.path.join(path,'embedding_dump.pkl')

    with open(file_path,'rb') as f:
        return pickle.load(f)

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
