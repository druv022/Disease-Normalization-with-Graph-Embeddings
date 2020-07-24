from collections import Iterable

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import clone

from nerds.doc.document import AnnotatedDocument
from nerds.features.rel2bow import BOWRelationFeatureExtractor
from nerds.util.logging import get_logger

log = get_logger()


class RelationExtractionModel(BaseEstimator, ClassifierMixin):
    """ Provides a basic interface to train Relation Extraction models and annotate documents.
        This is the core class responsible for training models that perform
        relation extraction, and retrieving relations between named entities from documents.
    """

    def __init__(self, relation_labels=None, feature_extractor=BOWRelationFeatureExtractor()):
        """	Initializes the model.
            relation_labels: if None, then all labels will be used; otherwise, only given ones."""
        # labels must be a string or collection of strings
        self.key = ""  # To be added in subclass.
        self.feature_extractor = feature_extractor
        self.relation_labels = None
        if relation_labels:
            if type(relation_labels) == str:
                self.relation_labels = [relation_labels]
            elif isinstance(relation_labels, Iterable):
                self.relation_labels = list(relation_labels)
            else:
                raise TypeError("Relation labels must be a string or collection of strings but '{}' is given"
                                .format(type(relation_labels)))

    def fit(self, X, y=None):
        """ Trains the Relation Extraction model. The input is a list of
                        `AnnotatedDocument` instances.

                        The basic implementation of this method performs no training and
                        should be overridden by offspring.
                """
        return self

    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
                        input and returns a list of `AnnotatedDocument` objects.

                        The basic implementation of this method does not annotate any
                        entities and should be overridden by offspring.
                """
        annotated_documents = []
        for document in X:
            annotated_documents.append(
                AnnotatedDocument(
                    content=document.content,
                    encoding=document.encoding,
                    identifier=document.identifier,
                    uuid=document.uuid))
        return annotated_documents

    def extract(self, X, y=None):
        """ Returns a list of relations, extracted from annotated documents. """
        annotated_documents = self.transform(X, y)
        relations = []
        for annotated_document in annotated_documents:
            relations.append(annotated_document.relations)
        return relations

    def clone(self):
        return clone(self)

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path.
                        Should be overridden.
                """
        raise NotImplementedError

    def load(self, file_path):
        """ Loads a model saved locally. Should be overridden. """
        raise NotImplementedError

    @property
    def name(self):
        """ Get the name of a model: it includes model's key and target relations labels (if any)
        """
        return self.key + ("_" + "_".join(self.relation_labels) if self.relation_labels else "")
