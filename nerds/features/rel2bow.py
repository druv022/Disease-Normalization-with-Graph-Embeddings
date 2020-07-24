from pathlib import Path

from scipy.sparse import csc_matrix
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

from nerds.features.base import RelationFeatureExtractor, UNKNOWN_WORD, UNKNOWN_LABEL, UNKNOWN_POS_TAG, \
    UNKNOWN_DEPENDENCY, BOWFeatureExtractor
from nerds.util.file import mkdir
from nerds.util.logging import get_logger

log = get_logger()

KEY = "rel2bow"


class BOWRelationFeatureExtractor(BOWFeatureExtractor, RelationFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.key = KEY
        self.word_vectorizer = None
        self.label_vectorizer = None
        self.pos_vectorizer = None
        self.dep_vectorizer = None

    def transform(self, X, y=None, relation_labels=None):
        log.info("Generating features for {} documents...".format(len(X)))
        self.docs_examples = list(self.annotated_documents_to_examples(X, relation_labels=relation_labels))
        ent_words = []
        ent_labels = []
        ent_pos_tags = []
        ent_deps = []
        rel_labels = []
        for doc, examples in self.docs_examples:
            for ex in examples:
                ent_words += [ex.context["source.text"], ex.context["target.text"]]
                ent_labels += [ex.context["source.label"], ex.context["target.label"]]
                ent_pos_tags += [ex.context["source.pos"], ex.context["target.pos"]]
                ent_deps += [ex.context["dependency"]]
                rel_labels += [ex.label]

        # add unknown values
        ent_words += [UNKNOWN_WORD, UNKNOWN_WORD]
        ent_labels += [UNKNOWN_LABEL, UNKNOWN_LABEL]
        ent_pos_tags += [UNKNOWN_POS_TAG, UNKNOWN_POS_TAG]
        ent_deps += [UNKNOWN_DEPENDENCY]

        if not self.word_vectorizer:
            # first time run
            self.word_vectorizer = CountVectorizer(binary=True)
            self.label_vectorizer = CountVectorizer(binary=True)
            self.pos_vectorizer = CountVectorizer(binary=True)
            self.dep_vectorizer = CountVectorizer(binary=True)
        else:
            # use vocabularies
            self.word_vectorizer = CountVectorizer(binary=True, vocabulary=self.word_vectorizer.vocabulary_)
            self.label_vectorizer = CountVectorizer(binary=True, vocabulary=self.label_vectorizer.vocabulary_)
            self.pos_vectorizer = CountVectorizer(binary=True, vocabulary=self.pos_vectorizer.vocabulary_)
            self.dep_vectorizer = CountVectorizer(binary=True, vocabulary=self.dep_vectorizer.vocabulary_)

            ent_words = self._process_unknown_values(
                ent_words, self.word_vectorizer.vocabulary, UNKNOWN_WORD)
            ent_labels = self._process_unknown_values(
                ent_labels, self.label_vectorizer.vocabulary, UNKNOWN_LABEL)
            ent_pos_tags = self._process_unknown_values(
                ent_pos_tags, self.pos_vectorizer.vocabulary, UNKNOWN_POS_TAG)
            ent_deps = self._process_unknown_values(
                ent_deps, self.dep_vectorizer.vocabulary, UNKNOWN_DEPENDENCY)

        # vectorize
        log.info("Vectorizing {} textual entries (words)...".format(len(ent_words)))
        word_vectors = self.word_vectorizer.fit_transform(ent_words)
        log.info("Vectorizing {} textual entries (labels)...".format(len(ent_labels)))
        label_vectors = self.label_vectorizer.fit_transform(ent_labels)
        log.info("Vectorizing {} textual entries (POS tags)...".format(len(ent_pos_tags)))
        pos_vectors = self.pos_vectorizer.fit_transform(ent_pos_tags)
        log.info("Vectorizing {} textual entries (dependency types)...".format(len(ent_deps)))
        dep_vectors = self.dep_vectorizer.fit_transform(ent_deps)

        # get shapes
        n_wor, m_wor = word_vectors.get_shape()
        n_lab, m_lab = label_vectors.get_shape()
        n_pos, m_pos = pos_vectors.get_shape()
        n_dep, m_dep = dep_vectors.get_shape()

        # create indices
        rows, cols, vals = [], [], []
        # ignore the last auxiliary value
        for row in range(n_dep - 1):
            for col in word_vectors.getrow(2 * row).nonzero()[1]:
                rows += [row]
                cols += [col]
                vals += [1]
            for col in word_vectors.getrow(2 * row + 1).nonzero()[1]:
                rows += [row]
                cols += [col + m_wor]
                vals += [1]
            for col in label_vectors.getrow(2 * row).nonzero()[1]:
                rows += [row]
                cols += [col + 2 * m_wor]
                vals += [1]
            for col in label_vectors.getrow(2 * row + 1).nonzero()[1]:
                rows += [row]
                cols += [col + 2 * m_wor + m_lab]
                vals += [1]
            for col in pos_vectors.getrow(2 * row).nonzero()[1]:
                rows += [row]
                cols += [col + 2 * m_wor + 2 * m_lab]
                vals += [1]
            for col in pos_vectors.getrow(2 * row + 1).nonzero()[1]:
                rows += [row]
                cols += [col + 2 * m_wor + 2 * m_lab + m_pos]
                vals += [1]
            for col in dep_vectors.getrow(row).nonzero()[1]:
                rows += [row]
                cols += [col + 2 * m_wor + 2 * m_lab + 2 * m_pos]
                vals += [1]

        # create a sparse matrix of features
        log.info("Creating a feature matrix...")
        feature_matrix = csc_matrix((vals, (rows, cols)), shape=(n_dep - 1, 2 * m_wor + 2 * m_lab + 2 * m_pos + m_dep))
        return feature_matrix, rel_labels

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
        labels_path = save_path.joinpath("labels.dict")
        pos_path = save_path.joinpath("pos.dict")
        dep_path = save_path.joinpath("dep.dict")

        # save dictionaries
        # we don't save examples for now
        joblib.dump(self.word_vectorizer, words_path)
        joblib.dump(self.label_vectorizer, labels_path)
        joblib.dump(self.pos_vectorizer, pos_path)
        joblib.dump(self.dep_vectorizer, dep_path)

    def load(self, file_path):
        load_path = Path(file_path)
        words_path = load_path.joinpath("words.dict")
        labels_path = load_path.joinpath("labels.dict")
        pos_path = load_path.joinpath("pos.dict")
        dep_path = load_path.joinpath("dep.dict")

        # load dictionaries
        # we don't load examples for now
        self.word_vectorizer = joblib.load(words_path)
        self.label_vectorizer = joblib.load(labels_path)
        self.pos_vectorizer = joblib.load(pos_path)
        self.dep_vectorizer = joblib.load(dep_path)
        return self
