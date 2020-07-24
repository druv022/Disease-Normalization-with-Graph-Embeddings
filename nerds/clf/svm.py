from pathlib import Path

import numpy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

from nerds.clf.base import DocumentClassifier
from nerds.config.base import SupportVectorModelConfiguration
from nerds.features.doc2bow import BOWDocumentFeatureExtractor
from nerds.util.file import mkdir
from nerds.util.logging import get_logger

log = get_logger()

KEY = "svm_doc_clf"


class SupportVectorClassifier(DocumentClassifier):
    def __init__(self, feature_extractor=BOWDocumentFeatureExtractor()):
        super().__init__(feature_extractor)
        self.svm = None
        self.key = KEY
        self.config = SupportVectorModelConfiguration()

    def fit(self, X, y, max_iterations=100, C=1):
        log.info("Checking parameters...")
        self.config.set_parameters({"max_iterations": max_iterations, "C": C})
        self.config.validate()

        # create a model
        log.info("Training SVM...")
        self.svm = CalibratedClassifierCV(LinearSVC(max_iter=max_iterations, C=C))

        # get features
        features = self.feature_extractor.transform(X)
        if features.getnnz() == 0:
            log.error("No examples to train, quiting...")
            return self

        self.svm.fit(features, y)
        return self

    def transform(self, X):
        # get features (labels are ignored)
        features = self.feature_extractor.transform(X)
        if features.getnnz() == 0:
            return X

        # make predictions
        log.info("Classifying {} documents with SVM...".format(len(X)))
        probs = self.svm.predict_proba(features)
        # the order of labels corresponds to the order of probabilities
        labels = self.svm.classes_
        predicted_labels = []
        for prob in probs:
            score = numpy.amax(prob)
            label = labels[numpy.where(prob == score)[0][0]]
            predicted_labels += [label]

        # could return scores if required
        return predicted_labels

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        save_path = Path(file_path)
        mkdir(save_path)
        model_save_path = save_path.joinpath("SVM.model")
        config_save_path = save_path.joinpath("SVM.config")
        joblib.dump(self.svm, model_save_path)
        self.feature_extractor.save(save_path.joinpath(self.feature_extractor.name))
        self.config.save(config_save_path)

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model_load_path = load_path.joinpath("SVM.model")
        config_load_path = load_path.joinpath("SVM.config")
        self.svm = joblib.load(model_load_path)
        self.feature_extractor.load(load_path.joinpath(self.feature_extractor.name))
        self.config.load(config_load_path)
        return self
