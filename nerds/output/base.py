from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin

from nerds.doc.document import AnnotatedDocument
from nerds.util.file import mkdir
from nerds.util.logging import get_logger

log = get_logger()


class DataOutput(BaseEstimator, TransformerMixin):
    """ Wrapper class that initializes a Converter object with a configuration.

        It accepts as input brat files, or path to brat files and parses them into
        AnnotatedDocument objects.

        Attributes:
            file_path (str): The path to the files that will be converted.
    """

    def __init__(self, path_to_files):
        path_to_files = Path(path_to_files)
        mkdir(path_to_files)
        self.path = path_to_files

    def fit(self, X=None, y=None):
        # Do nothing, just return the piped object.
        return self

    def transform(self, X, y=None):
        """ Uses the Converter this wrapper contains to convert AnnotatedDocuments
            into a desired format. The output of our pipeline is of type AnnotatedDocument,
            thus the base convert should simply read and return the Annotated Document Objects.

            To actually apply the conversion and return the desired result, one should
            extend this class to support specific output format ( e.g. Brat, RDF, JSON)
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
        """ Saves the converted input to the local disk, provided a file path.
            Should be overridden.
        """
        raise NotImplementedError

    def load(self, file_path):
        """ Loads a converted result saved locally. Should be overridden. """
        raise NotImplementedError
