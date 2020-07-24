from pathlib import Path

import sklearn_crfsuite
from sklearn.externals import joblib

from nerds.config.base import CRFModelConfiguration
from nerds.features.word2seq import WordSequenceFeatureExtractor
from nerds.ner.base import NamedEntityRecognitionModel
from nerds.doc.bio import transform_bio_tags_to_annotated_documents
from nerds.util.file import mkdir
from nerds.util.logging import get_logger

log = get_logger()

KEY = "crf_ner"


class CRF(NamedEntityRecognitionModel):
    def __init__(self, entity_labels=None, window=2, prefix=2, suffix=2):
        super().__init__(entity_labels)
        self.crf = None
        self.key = KEY
        self.config = CRFModelConfiguration()
        if self.entity_labels:
            self.config.set_parameter("entity_labels", self.entity_labels)
        self.config.set_parameters({"window": window, "prefix": prefix, "suffix": suffix})
        self.window = self.config.get_parameter("window")
        self.prefix = self.config.get_parameter("prefix")
        self.suffix = self.config.get_parameter("suffix")
        self.feature_extractor = WordSequenceFeatureExtractor(
            window=self.window, prefix=self.prefix, suffix=self.suffix)

    def fit(self, X, y=None, max_iterations=50, c1=0.1, c2=0.1):
        log.info("Checking parameters...")
        self.config.set_parameters({"max_iterations": max_iterations, "c1": c1, "c2": c2})
        self.config.validate()

        log.info("Generating features for {} samples...".format(len(X)))
        # Features and labels are useful for training.
        features, tokens, labels = self.feature_extractor.transform(X, entity_labels=self.entity_labels)

        log.info("Training CRF...")
        # Configure training parameters
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=self.config.get_parameter("c1"),
            c2=self.config.get_parameter("c2"),
            max_iterations=self.config.get_parameter("max_iterations"),
            all_possible_transitions=True,
            all_possible_states=True,
            verbose=True)
        self.crf.fit(features, labels)

        return self

    def transform(self, X, y=None):
        log.info("Annotating named entities in {} documents with CRF...".format(len(X)))
        features, tokens, _ = self.feature_extractor.transform(X, entity_labels=self.entity_labels)
        # Labels, of course, doesn't contain anything here.

        # Make predictions.
        predicted_labels = self.crf.predict_marginals(features)

        # Also need to make annotated documents.
        return transform_bio_tags_to_annotated_documents(tokens, predicted_labels, X)

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        save_path = Path(file_path)
        mkdir(save_path)
        model_save_path = save_path.joinpath("CRF.model")
        config_save_path = save_path.joinpath("CRF.config")
        joblib.dump(self.crf, model_save_path)
        self.config.save(config_save_path)

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model_load_path = load_path.joinpath("CRF.model")
        config_load_path = load_path.joinpath("CRF.config")
        self.crf = joblib.load(model_load_path)
        self.config.load(config_load_path)
        return self
