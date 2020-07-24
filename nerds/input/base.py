from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin

from nerds.doc.document import Document
from nerds.util.logging import log_progress, get_logger

log = get_logger()


class DataInput(BaseEstimator, TransformerMixin):
    """ Provides the data input of a NER pipeline.

        This class provides the input to the rest of the pipeline,
        by transforming a collection of files (in the provided path) into a
        collection of documents. The `annotated` parameter differentiates
        between the already annotated input (required for training/evaluation)
        and the non-annotated input (required for entity extraction).

        Attributes:
            path_to_files (str): The path containing the input files,
                annotated or not.
            annotated (bool): If `False`, then the returned collection will
                consist of Document objects. If `True`, it will consist of
                AnnotatedDocument objects.
            encoding (str, optional): Specifies the encoding of the plain
                text. Defaults to 'utf-8'.

        Raises:
            IOError: If `path_to_files` does not exist.
    """

    def __init__(self, path_to_files, annotated=True, encoding="utf-8"):
        path_to_files = Path(path_to_files)
        if not path_to_files.is_dir():
            raise IOError("Invalid path for parameter 'path_to_file'")

        self.path = path_to_files
        self.annotated = annotated
        self.encoding = encoding

    @property
    def file_count_(self):
        """ Method that returns the number of files that the input found within a specific folder

            Yields:
                int: File count
        """
        return len(list(self.path.glob('*.txt')))

    def fit(self, X=None, y=None):
        # Do nothing, just return the piped object.
        return self

    def transform(self, X=None, y=None):
        """ Transforms the available documents into the appropriate objects,
            differentiating on the `annotated` parameter.
        """
        docs = []
        if self.annotated:
            raise NotImplementedError
        else:
            txt_files = list(self.path.glob('*.txt'))
            txt_files.sort()
            for idx, txt_file in enumerate(txt_files):
                with open(txt_file, 'rb') as doc_file:
                    docs.append(
                        Document(
                            content=doc_file.read(),
                            encoding=self.encoding,
                            identifier=txt_file.name.replace('.txt', '')))
                # info
                log_progress(log, idx, len(txt_files))
            return docs
