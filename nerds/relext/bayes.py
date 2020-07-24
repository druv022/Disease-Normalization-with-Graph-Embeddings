from pathlib import Path

import numpy
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB

from nerds.config.base import NaiveBayesModelConfiguration
from nerds.features.base import is_empty, to_weights
from nerds.features.rel2bow import BOWRelationFeatureExtractor
from nerds.relext.base import RelationExtractionModel
from nerds.util.file import mkdir
from nerds.util.logging import get_logger

log = get_logger()

KEY = "naive_bayes_re"


class NaiveBayesRE(RelationExtractionModel):
    def __init__(self, relation_labels=None, feature_extractor=BOWRelationFeatureExtractor()):
        super().__init__(relation_labels, feature_extractor)
        self.nb = None
        self.key = KEY
        self.config = NaiveBayesModelConfiguration()
        if self.relation_labels:
            self.config.set_parameter("relation_labels", self.relation_labels)

    def fit(self, X, y=None, sample_weight=False):
        log.info("Checking parameters...")
        self.config.validate()

        # create a model
        self.nb = MultinomialNB()

        # get features and labels
        features, labels = self.feature_extractor.transform(X, relation_labels=self.relation_labels)
        if is_empty(features):
            log.error("No examples to train, quiting...")
            return self

        log.info("Training Naive Bayes...")
        weights = to_weights(labels) if sample_weight else None
        self.nb.fit(features, labels, sample_weight=weights)
        return self

    def transform(self, X, y=None):
        # get features (labels are ignored)
        features, _ = self.feature_extractor.transform(X, relation_labels=self.relation_labels)
        if is_empty(features):
            return X

        # make predictions
        log.info("Predicting relations in {} documents with Naive Bayes...".format(len(X)))
        probs = self.nb.predict_proba(features)
        # the order of labels corresponds to the order of probabilities
        labels = self.nb.classes_
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
        model_save_path = save_path.joinpath("NaiveBayes.model")
        config_save_path = save_path.joinpath("NaiveBayes.config")
        joblib.dump(self.nb, model_save_path)
        self.feature_extractor.save(save_path.joinpath(self.feature_extractor.name))
        self.config.save(config_save_path)

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model_load_path = load_path.joinpath("NaiveBayes.model")
        config_load_path = load_path.joinpath("NaiveBayes.config")
        self.nb = joblib.load(model_load_path)
        self.feature_extractor.load(load_path.joinpath(self.feature_extractor.name))
        self.config.load(config_load_path)
        return self
