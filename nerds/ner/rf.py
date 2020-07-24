import collections
from math import sqrt, log2
from pathlib import Path

from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from nerds.config.base import RandomForestModelConfiguration
from nerds.doc.bio import transform_bio_tags_to_annotated_documents
from nerds.features.base import to_weights
from nerds.features.fcontext import WordContextFeatureExtractor
from nerds.ner import NamedEntityRecognitionModel
from nerds.util.file import mkdir
from nerds.util.logging import get_logger, log_progress

log = get_logger()

KEY = "rf_ner"


class RandomForestNER(NamedEntityRecognitionModel):
    def __init__(self, entity_labels=None, ngram_slice=2, window=1):
        super().__init__(entity_labels)
        self.rf = None
        self.key = KEY
        self.feature_extractor = WordContextFeatureExtractor(ngram_slice=ngram_slice, window=window)
        self.config = RandomForestModelConfiguration()
        if self.entity_labels:
            self.config.set_parameter("entity_labels", self.entity_labels)

    def fit(self, X, y=None, n_estimators=9, max_features="auto",
            min_samples_leaf=1, sample_weight=False, random_state=None):
        log.info("Generating features for {} samples...".format(len(X)))
        # Features and labels are useful for training.
        features, tokens, labels = self.feature_extractor.transform(X, entity_labels=self.entity_labels)

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
                                    "random_state": random_state,
                                    "min_samples_leaf": min_samples_leaf})

        # create a model
        self.rf = RandomForestClassifier(
            n_estimators=self.config.get_parameter("n_estimators"),
            max_features=self.config.get_parameter("max_features"),
            min_samples_leaf=self.config.get_parameter("min_samples_leaf"),
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
        log.info("Annotating named entities in {} documents with Decision Tree...".format(len(X)))
        tokens_per_doc, predicted_labels_per_doc = [], []
        for idx, document in enumerate(X):
            # Labels, of course, doesn't contain anything here.
            features, tokens, _ = self.feature_extractor.transform([document], entity_labels=self.entity_labels)
            # Make predictions.
            predicted_labels = self.rf.predict(features)
            tokens_per_doc.append(tokens)
            predicted_labels_per_doc.append(predicted_labels)
            log_progress(log, idx, len(X))

        # Also need to make annotated documents.
        return transform_bio_tags_to_annotated_documents(tokens_per_doc, predicted_labels_per_doc, X)

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
