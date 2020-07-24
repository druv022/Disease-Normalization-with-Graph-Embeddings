from pathlib import Path

from nerds.clf.base import DocumentClassifier
from nerds.clf.svm import SupportVectorClassifier
from nerds.features.doc2bow import BOWDocumentFeatureExtractor
from nerds.util.file import mkdir
from nerds.util.logging import get_logger

log = get_logger()

KEY = "svm_doc_filter"


class SupportVectorFilter(DocumentClassifier):
    def __init__(self, feature_extractor=BOWDocumentFeatureExtractor()):
        super().__init__(feature_extractor)
        self.svm = None
        self.key = KEY

    def fit(self, target_data, other_data, max_iterations=100, C=1):
        # create a model
        self.svm = SupportVectorClassifier()
        # create examples and train
        self.svm.fit(
            target_data + other_data, [1] * len(target_data) + [0] * len(other_data),
            max_iterations=max_iterations,
            C=C)
        return self

    def transform(self, data):
        """ Returns a list of 0/1 labels where 0 means exclusion and 1 means inclusion
        Args:
            data: list(Document)
        Returns: list of 0/1 labels
        """
        return self.svm.transform(data)

    def filter(self, data):
        """ Filters documents based on 0/1 labels where 0 means exclusion and 1 means inclusion
        Args:
            data: list(Document)
        Returns: list(Document), indices of retained documents
        """
        return [idx for idx, label in enumerate(self.transform(data)) if label == 1]

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        mkdir(file_path)
        self.svm.save(file_path.joinpath(self.svm.name))

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model = SupportVectorClassifier()
        self.svm = model.load(load_path.joinpath(model.name))
        return self
