from pathlib import Path

import numpy

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from nerds.config.base import LogisticRegressionModelConfiguration
from nerds.features.base import is_empty, to_weights
from nerds.features.rel2bow import BOWRelationFeatureExtractor
from nerds.relext.base import RelationExtractionModel
from nerds.util.file import mkdir
from nerds.util.logging import get_logger

log = get_logger()

KEY = "log_reg_re"


class LogisticRegressionRE(RelationExtractionModel):
    def __init__(self, relation_labels=None, feature_extractor=BOWRelationFeatureExtractor()):
        super().__init__(relation_labels, feature_extractor)
        self.logreg = None
        self.key = KEY
        self.config = LogisticRegressionModelConfiguration()
        if self.relation_labels:
            self.config.set_parameter("relation_labels", self.relation_labels)

    def fit(self, X, y=None, max_iterations=100, C=1, sample_weight=False):
        log.info("Checking parameters...")
        self.config.set_parameters({"max_iterations": max_iterations, "C": C})
        self.config.validate()

        # create a model
        self.logreg = LogisticRegression(
            max_iter=self.config.get_parameter("max_iterations"), C=self.config.get_parameter("C"))

        # get features and labels
        features, labels = self.feature_extractor.transform(X, relation_labels=self.relation_labels)
        if is_empty(features):
            log.error("No examples to train, quiting...")
            return self

        log.info("Training Logistic Regression...")
        weights = to_weights(labels) if sample_weight else None
        self.logreg.fit(features, labels, sample_weight=weights)
        return self

    def transform(self, X, y=None):
        # get features (labels are ignored)
        features, _ = self.feature_extractor.transform(X, relation_labels=self.relation_labels)
        if is_empty(features):
            return X

        # make predictions
        log.info("Predicting relations in {} documents with Logistic Regression...".format(len(X)))
        probs = self.logreg.predict_proba(features)
        # the order of labels corresponds to the order of probabilities
        labels = self.logreg.classes_
        predicted_labels = []
        for prob in probs:
            score = numpy.amax(prob)
            label = labels[numpy.where(prob == score)[0][0]]
            predicted_labels += [{label: score}]

        # make annotated documents
        return self.feature_extractor.predictions_to_annotated_documents(predicted_labels)

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        save_path = Path(file_path)
        mkdir(save_path)
        model_save_path = save_path.joinpath("LogReg.model")
        config_save_path = save_path.joinpath("LogReg.config")
        joblib.dump(self.logreg, model_save_path)
        self.feature_extractor.save(save_path.joinpath(self.feature_extractor.name))
        self.config.save(config_save_path)

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model_load_path = load_path.joinpath("LogReg.model")
        config_load_path = load_path.joinpath("LogReg.config")
        self.logreg = joblib.load(model_load_path)
        self.feature_extractor.load(load_path.joinpath(self.feature_extractor.name))
        self.config.load(config_load_path)
        return self
