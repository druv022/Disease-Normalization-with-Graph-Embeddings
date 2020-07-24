import collections
from math import sqrt, log2
from pathlib import Path

import numpy
from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from nerds.config.base import RandomForestModelConfiguration
from nerds.features.base import is_empty, to_weights
from nerds.features.rel2bow import BOWRelationFeatureExtractor
from nerds.relext.base import RelationExtractionModel
from nerds.util.file import mkdir
from nerds.util.logging import get_logger

log = get_logger()

KEY = "rf_re"


class RandomForestRE(RelationExtractionModel):
    def __init__(self, relation_labels=None, feature_extractor=BOWRelationFeatureExtractor()):
        super().__init__(relation_labels, feature_extractor)
        self.rf = None
        self.key = KEY
        self.config = RandomForestModelConfiguration()
        if self.relation_labels:
            self.config.set_parameter("relation_labels", self.relation_labels)

    def fit(self, X, y=None, n_estimators=9, max_features="auto", sample_weight=False, random_state=None):
        # get features and labels
        features, labels = self.feature_extractor.transform(X, relation_labels=self.relation_labels)
        if is_empty(features):
            log.error("No examples to train, quiting...")
            return self

        log.info("Checking parameters...")
        if type(max_features) == str:
            if max_features == "auto":
                max_features = self._get_max_features(features)
            elif max_features == "log":
                max_features = self._get_max_features(features, method=log2)
            elif max_features == "sqrt":
                max_features = self._get_max_features(features, method=sqrt)
            else:
                raise ValueError("Unknown method '{}' for feature selection in Random Forest".format(max_features))
        if type(max_features) != int:
            raise TypeError("The parameter 'max_features' must be either a string or integer.")

        self.config.set_parameters({"n_estimators": n_estimators,
                                    "max_features": max_features,
                                    "random_state": random_state})

        # create a model
        self.rf = RandomForestClassifier(
            n_estimators=self.config.get_parameter("n_estimators"),
            max_features=self.config.get_parameter("max_features"),
            random_state=self.config.get_parameter("random_state"))

        log.info("Training Random Forest...")
        weights = to_weights(labels) if sample_weight else None
        self.rf.fit(features, labels, sample_weight=weights)
        return self

    def _get_max_features(self, features, method=sqrt):
        if isinstance(features, csc_matrix):
            return int(method(features.shape[1]))
        elif isinstance(features, collections.Sequence):
            for vector in features:
                return int(method(len(vector)))
        else:
            raise TypeError("Feature matrix is of type '{}' which is not array-like".format(type(features)))

    def transform(self, X, y=None):
        # get features (labels are ignored)
        features, _ = self.feature_extractor.transform(X, relation_labels=self.relation_labels)
        if is_empty(features):
            return X

        # make predictions
        log.info("Predicting relations in {} documents with Random Forest...".format(len(X)))
        probs = self.rf.predict_proba(features)
        # the order of labels corresponds to the order of probabilities
        labels = self.rf.classes_
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
        model_save_path = save_path.joinpath("RF.model")
        config_save_path = save_path.joinpath("RF.config")
        joblib.dump(self.rf, model_save_path)
        self.feature_extractor.save(save_path.joinpath(self.feature_extractor.name))
        self.config.save(config_save_path)

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model_load_path = load_path.joinpath("RF.model")
        config_load_path = load_path.joinpath("RF.config")
        self.rf = joblib.load(model_load_path)
        self.feature_extractor.load(load_path.joinpath(self.feature_extractor.name))
        self.config.load(config_load_path)
        return self
