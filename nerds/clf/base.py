from sklearn.base import BaseEstimator, ClassifierMixin

from nerds.doc.document import AnnotatedDocument
from nerds.features.doc2bow import BOWDocumentFeatureExtractor


class DocumentClassifier(BaseEstimator, ClassifierMixin):
    """ Provides a basic interface to train models for document classification.
        """

    def __init__(self, feature_extractor=BOWDocumentFeatureExtractor()):
        """	Initializes the model.
                """
        self.key = ""  # To be added in subclass.
        self.feature_extractor = feature_extractor

    def fit(self, X, y):
        """ Trains a document classifier. The input X is a list of
                        `AnnotatedDocument` instances. The input y is a list of labels.

                        The basic implementation of this method performs no training and
                        should be overridden by offspring.
                """
        return self

    def transform(self, X):
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
        """ Get the name of a model: it is currently model's key
        """
        return self.key
