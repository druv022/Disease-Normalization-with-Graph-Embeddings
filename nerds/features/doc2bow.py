from pathlib import Path

from scipy.sparse import csc_matrix
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

from nerds.features.base import BOWFeatureExtractor, UNKNOWN_WORD
from nerds.util.file import mkdir
from nerds.util.logging import get_logger, log_progress
from nerds.util.nlp import remove_stop_words_and_lemmatize

log = get_logger()

KEY = "doc2bow"


class BOWDocumentFeatureExtractor(BOWFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.key = KEY
        self.word_vectorizer = None

    def transform(self, X, y=None):
        log.info("Generating features for {} documents...".format(len(X)))
        doc_snippets = []
        for idx, doc in enumerate(X):
            doc_snippets += [remove_stop_words_and_lemmatize(doc.plain_text_)]
            # info
            log_progress(log, idx, len(X))

        doc_snippets += [UNKNOWN_WORD]

        if not self.word_vectorizer:
            # first time run
            self.word_vectorizer = CountVectorizer(binary=True)
        else:
            # use vocabularies
            self.word_vectorizer = CountVectorizer(binary=True, vocabulary=self.word_vectorizer.vocabulary_)

            # substitute unknown values
            doc_snippets = self._process_unknown_values(
                doc_snippets, self.word_vectorizer.vocabulary, UNKNOWN_WORD)

        # vectorize
        word_vectors = self.word_vectorizer.fit_transform(doc_snippets)

        # get shapes
        n_wor, m_wor = word_vectors.get_shape()

        # create indices
        rows, cols, vals = [], [], []
        # ignore the last auxiliary value
        for row in range(n_wor - 1):
            for col in word_vectors.getrow(row).nonzero()[1]:
                rows += [row]
                cols += [col]
                vals += [1]

        # create a sparse matrix of features
        feature_matrix = csc_matrix((vals, (rows, cols)), shape=(n_wor - 1, m_wor))
        return feature_matrix

    def _process_unknown_values(self, entries, vocabulary, unknown_label):
        entries_ref = []
        for entry in entries:
            known_tokens = []
            for token in entry.split():
                if token.lower() in vocabulary:
                    known_tokens += [token]
                else:
                    known_tokens += [unknown_label]
            entries_ref += [" ".join(known_tokens)]
        return entries_ref

    def save(self, file_path):
        save_path = Path(file_path)
        mkdir(save_path)
        words_path = save_path.joinpath("words.dict")

        # save dictionaries
        # we don't save examples for now
        joblib.dump(self.word_vectorizer, words_path)

    def load(self, file_path):
        load_path = Path(file_path)
        words_path = load_path.joinpath("words.dict")

        # load dictionaries
        # we don't load examples for now
        self.word_vectorizer = joblib.load(words_path)
        return self
