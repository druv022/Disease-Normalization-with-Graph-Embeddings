from pathlib import Path

import numpy as np

from nerds.features.base import VectorFeatureExtractor, RelationFeatureExtractor
from nerds.features.char2vec import Char2VecFeatureExtractor
from nerds.features.pos2vec import PoS2VecFeatureExtractor
from nerds.features.word2vec import Word2VecFeatureExtractor
from nerds.util.file import mkdir
from nerds.util.logging import get_logger, log_progress
from nerds.util.nlp import document_to_tokens

log = get_logger()

KEY = "rel2vec"


class VectorRelationFeatureExtractor(VectorFeatureExtractor, RelationFeatureExtractor):
    def __init__(self,
                 word_vectorizer=Word2VecFeatureExtractor(),
                 char_vectorizer=Char2VecFeatureExtractor(),
                 pos_vectorizer=PoS2VecFeatureExtractor()):
        super().__init__()
        self.key = KEY
        self.word_vectorizer = word_vectorizer
        self.char_vectorizer = char_vectorizer
        self.pos_vectorizer = pos_vectorizer

    def fit(self, X, y=None, size=100, min_count=5, workers=1, window=5, sample=1e-3, skipgram=False, min_n=3, max_n=6):
        """ Trains word, character, and part-of-speech embeddings
            (see Char2VecFeatureExtractor for the description of arguments).
        """
        # Get sentences as lists of tokens
        log.info("Tokenizing {} documents...".format(len(X)))
        sentences = []
        for idx, doc in enumerate(X):
            sentences.append(document_to_tokens(doc))
            log_progress(log, idx, len(X))
        self.word_vectorizer.fit(
            sentences,
            y,
            size=size,
            min_count=min_count,
            workers=workers,
            window=window,
            sample=sample,
            skipgram=skipgram)
        self.pos_vectorizer.fit(
            sentences,
            y,
            size=size,
            min_count=min_count,
            workers=workers,
            window=window,
            sample=sample,
            skipgram=skipgram)
        self.char_vectorizer.fit(
            sentences,
            y,
            size=size,
            min_count=min_count,
            workers=workers,
            window=window,
            sample=sample,
            skipgram=skipgram,
            min_n=min_n,
            max_n=max_n)
        return self

    def transform(self, X, y=None, relation_labels=None):
        log.info("Generating features for {} documents...".format(len(X)))
        self.docs_examples = list(self.annotated_documents_to_examples(X, relation_labels=relation_labels))
        rel_labels = []
        features = []
        for doc, examples in self.docs_examples:
            for ex in examples:
                rel_labels.append(ex.label)
                features.append(
                    np.concatenate((self.word_vectorizer.sum_vector(ex.context["source.text"]),
                                    self.word_vectorizer.sum_vector(ex.context["target.text"]),
                                    self.char_vectorizer.sum_vector(ex.context["source.text"]),
                                    self.char_vectorizer.sum_vector(ex.context["target.text"]),
                                    self.pos_vectorizer.sum_vector(ex.context["source.pos"]),
                                    self.pos_vectorizer.sum_vector(ex.context["target.pos"]))))
        return features, rel_labels

    def save(self, file_path):
        save_path = Path(file_path)
        mkdir(save_path)
        self.word_vectorizer.save(save_path)
        self.pos_vectorizer.save(save_path)
        self.char_vectorizer.save(save_path)

    def load(self, file_path):
        load_path = Path(file_path)
        self.word_vectorizer.load(load_path)
        self.pos_vectorizer.load(load_path)
        self.char_vectorizer.load(load_path)
        return self
