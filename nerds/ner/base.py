from collections import Iterable

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import clone

from nerds.doc.document import AnnotatedDocument
from nerds.util.logging import get_logger

log = get_logger()


class NamedEntityRecognitionModel(BaseEstimator, ClassifierMixin):
    """ Provides a basic interface to train NER models and annotate documents.
        This is the core class responsible for training models that perform
        named entity recognition, and retrieving named entities from documents.
    """

    def __init__(self, entity_labels=None):
        """	Initializes the model.
            entity_labels: if None, then all labels will be used; otherwise, only given ones."""
        # labels must be a string or collection of strings
        self.key = ""  # To be added in subclass.
        self.entity_labels = None
        if entity_labels:
            if type(entity_labels) == str:
                self.entity_labels = [entity_labels]
            elif isinstance(entity_labels, Iterable):
                self.entity_labels = list(entity_labels)
            else:
                raise TypeError("Entity labels must be a string or collection of strings but '{}' is given"
                                .format(type(entity_labels)))

    def fit(self, X, y=None):
        """ Trains the NER model. The input is a list of
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
        """ Returns a list of entities, extracted from annotated documents. """
        annotated_documents = self.transform(X, y)
        entities = []
        for annotated_document in annotated_documents:
            entities.append(annotated_document.annotations)
        return entities

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
        """ Get the name of a model: it includes model's key and target entity labels (if any)
        """
        return self.key + ("_" + "_".join(self.entity_labels) if self.entity_labels else "")
