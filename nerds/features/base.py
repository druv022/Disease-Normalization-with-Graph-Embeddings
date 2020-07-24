import collections

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

from nerds.doc.annotation import Relation
from nerds.doc.bio import best
from nerds.doc.document import AnnotatedDocument
from nerds.util.logging import get_logger, log_progress
from nerds.util.nlp import text_to_spacy_document

log = get_logger()

NO_RELATION = "NO_RELATION"

UNKNOWN_WORD = "UNKNOWN_WORD"
UNKNOWN_LABEL = "UNKNOWN_LABEL"
UNKNOWN_POS_TAG = "UNKNOWN_POS_TAG"
UNKNOWN_DEPENDENCY = "UNKNOWN_DEPENDENCY"


class FeatureExtractor(BaseEstimator, ClassifierMixin):
    """ Provides a basic interface to extract features from documents.
        This includes document classification, named entity recognition, and relation extraction.
    """

    def __init__(self):
        """	Initializes the model.
        """
        self.key = ""  # To be added in subclass.

    def fit(self, X, y=None):
        """ Fits a feature extractor. The input X is a list of
            `Document` instances. The input y is a list of labels.
            The basic implementation of this method performs no training and
            should be overridden by offspring if required.
        """
        return self

    def transform(self, X, y=None):
        """ Transforms the list of `Document` objects that are provided as
            input and returns an array-like matrix of features.
            The basic implementation of this method should be overridden by offspring.
        """
        return X

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. Should be overridden. """

    def load(self, file_path):
        """ Loads a model saved locally. Should be overridden. """

    @property
    def name(self):
        """ Get the name of a model: it is currently model's key
        """
        return self.key


class BOWFeatureExtractor(FeatureExtractor):
    """ Provides a basic interface to extract features from documents as dense vectors (embeddings).
    """

    def __init__(self):
        """	Initializes the model.
        """
        super().__init__()


class VectorFeatureExtractor(FeatureExtractor):
    """ Provides a basic interface to extract features from documents as dense vectors (embeddings).
    """

    def __init__(self):
        """	Initializes the model.
        """
        super().__init__()
        self.model = None

    def vector(self, word):
        if word in self.model.wv:
            return self.model.wv[word]
        return None

    def sum_vector(self, text):
        if text in self.model.wv:
            return self.model.wv[text]
        sum_vec = np.zeros(self.model.vector_size)
        for word in text.split():
            if word in self.model.wv:
                sum_vec += self.model.wv[word]
        return sum_vec

    @property
    def dimension_(self):
        return self.model.vector_size


class DocumentExample:
    def __init__(self, context, label):
        self.context = context
        self.label = label


class RelationExample:
    def __init__(self, context, label, source, target):
        self.context = context
        self.label = label
        self.source = source
        self.target = target


class RelationFeatureExtractor(FeatureExtractor):
    """ Provides a basic interface to extract features from documents for relation extraction.
    """

    def __init__(self):
        """	Initializes the model.
        """
        super().__init__()
        self.docs_examples = None

    def predictions_to_annotated_documents(self, predicted_labels):
        # predicted_labels must correspond to examples (the same length and order)
        ann_docs = []
        idx = 0
        for doc, examples in self.docs_examples:
            rel_id = 0
            relations = []
            for ex in examples:
                label = best(predicted_labels[idx])
                if label != NO_RELATION:
                    rel_id += 1
                    relations += [
                        Relation(
                            label,
                            ex.source.identifier,
                            ex.target.identifier,
                            'R{}'.format(rel_id),
                            score=predicted_labels[idx][label])
                    ]
                idx += 1

            ann_docs += [
                AnnotatedDocument(
                    content=doc.content,
                    annotations=doc.annotations,
                    relations=relations,
                    normalizations=doc.normalizations,
                    encoding=doc.encoding,
                    identifier=doc.identifier,
                    uuid=doc.uuid)
            ]

        return ann_docs

    def annotated_documents_to_examples(self, annotated_documents, relation_labels=None):
        """ Given annotated documents, generates features and labels for relation extraction.
                data: list(AnnonatedDocument)
            """
        log.info("Generating examples for {} documents...".format(len(annotated_documents)))
        for idx, ann_doc in enumerate(annotated_documents):
            entities = ann_doc.annotations
            relations = ann_doc.relations

            examples = []

            # basic NLP
            doc = text_to_spacy_document(ann_doc.plain_text_)

            # get positive examples
            relation_args = set()
            for relation in relations:
                # all other labels are ignored
                if relation_labels and relation.label not in relation_labels:
                    continue
                # otherwise, process all labels
                # this should work fine for short documents
                rel_source = None
                for source in entities:
                    if source.identifier == relation.source_id:
                        rel_source = source
                        break
                rel_target = None
                for target in entities:
                    if target.identifier == relation.target_id:
                        rel_target = target
                        break

                if not rel_source:
                    log.error("Source with ID '{}' is not found in document '{}', skipping...".format(
                        relation.source_id, doc.text))
                    continue
                if not rel_target:
                    log.error("Target with ID '{}' is not found in document '{}', skipping...".format(
                        relation.target_id, doc.text))
                    continue

                relation_args.add((rel_source, rel_target))

                examples += [
                    RelationExample(self._relation_args_to_features(rel_source, rel_target, doc),
                                    relation.label, rel_source, rel_target)
                ]

            # get negative examples
            for ent1 in entities:
                for ent2 in entities:
                    if ent1 == ent2 or (ent1, ent2) in relation_args:
                        continue
                    examples += [RelationExample(self._relation_args_to_features(ent1, ent2, doc),
                                                 NO_RELATION, ent1, ent2)]
            # info
            log_progress(log, idx, len(annotated_documents))
            yield (ann_doc, examples)

    def _relation_args_to_features(self, source, target, spacy_document):
        """ Given a piece of text processed by Spacy,
                    the function extracts handcrafted features for the source and
                    target of the given relation.
            :param source: subject of a relation
            :param target: object of a relation
            :param spacy_document: a piece of text processed by Spacy
            :return: extracted features, a dictionary of feature names and values
            """
        # a dictionary of feature names and values
        context = dict()

        # entity text
        context["source.text"] = source.text.lower()
        context["target.text"] = target.text.lower()

        # entity type
        context["source.label"] = source.label
        context["target.label"] = target.label

        # locate tokens
        def _to_tokens(entity):
            ent_tokens = [token for token in spacy_document if entity.offset[0] <= token.idx <= entity.offset[1]]
            # search for tokens that overlap with the entity, e.g. "fFatal", "(ILD)/pneumonitis"
            if not ent_tokens:
                ent_tokens = [
                    token for token in spacy_document if token.idx < entity.offset[0] < token.idx + len(token.text)
                ]
            return ent_tokens

        source_tokens = _to_tokens(source)
        target_tokens = _to_tokens(target)

        # pos tags
        context["source.pos"] = " ".join([t.pos_ for t in source_tokens])
        context["target.pos"] = " ".join([t.pos_ for t in target_tokens])

        # get dependencies
        def _dependency():
            deps = []
            for t in target_tokens:
                for s in source_tokens:
                    if s == t.head:
                        deps += [t.dep_]
            return deps

        # dependency
        deps = _dependency()
        context["dependency"] = " ".join(deps) if deps else "NO_DEPENDENCY"

        return context


def to_weights(labels):
    """ Calculates weights inversely proportional to class frequencies
    :param labels (list)
    :return: weights (list), the same size as labels
    """
    label_weights = []
    label_hist = dict()
    label_set = set()
    for label in labels:
        label_set.add(label)
        if label in label_hist:
            label_hist[label] += 1.
        else:
            label_hist[label] = 1.
    for label in labels:
        label_weights.append(len(labels) / (len(label_set) * label_hist[label]))
    return label_weights


def is_empty(features):
    if isinstance(features, csc_matrix):
        return features.getnnz() == 0
    elif isinstance(features, collections.Sequence):
        return len(features) == 0
    else:
        raise TypeError("Feature matrix is of type '{}' which is not array-like".format(type(features)))
