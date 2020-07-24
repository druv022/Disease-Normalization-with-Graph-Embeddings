from pathlib import Path

from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier

from nerds.config.base import DecisionTreeModelConfiguration
from nerds.doc.bio import transform_bio_tags_to_annotated_documents
from nerds.features.base import to_weights
from nerds.features.fcontext import WordContextFeatureExtractor
from nerds.ner.base import NamedEntityRecognitionModel
from nerds.util.file import mkdir
from nerds.util.logging import get_logger, log_progress

log = get_logger()

KEY = "dtree_ner"


class DecisionTreeNER(NamedEntityRecognitionModel):
    def __init__(self, entity_labels=None, ngram_slice=2, window=1):
        super().__init__(entity_labels)
        self.tree = None
        self.key = KEY
        self.feature_extractor = WordContextFeatureExtractor(ngram_slice=ngram_slice, window=window)
        self.config = DecisionTreeModelConfiguration()
        if self.entity_labels:
            self.config.set_parameter("entity_labels", self.entity_labels)

    def fit(self, X, y=None, max_depth=10, min_samples_leaf=1, sample_weight=False, random_state=None):
        log.info("Checking parameters...")
        self.config.set_parameters({"max_depth": max_depth,
                                    "min_samples_leaf": min_samples_leaf,
                                    "random_state": random_state})

        # create a model
        self.tree = DecisionTreeClassifier(
            max_depth=self.config.get_parameter("max_depth"),
            min_samples_leaf=self.config.get_parameter("min_samples_leaf"),
            random_state=self.config.get_parameter("random_state"))

        log.info("Generating features for {} samples...".format(len(X)))
        # Features and labels are useful for training.
        features, tokens, labels = self.feature_extractor.transform(X, entity_labels=self.entity_labels)

        log.info("Training Decision Tree...")
        weights = to_weights(labels) if sample_weight else None
        self.tree.fit(features, labels, sample_weight=weights)
        return self

    def transform(self, X, y=None):
        log.info("Annotating named entities in {} documents with Decision Tree...".format(len(X)))
        tokens_per_doc, predicted_labels_per_doc = [], []
        for idx, document in enumerate(X):
            # Labels, of course, doesn't contain anything here.
            features, tokens, _ = self.feature_extractor.transform([document], entity_labels=self.entity_labels)
            # Make predictions.
            predicted_labels = self.tree.predict(features)
            tokens_per_doc.append(tokens)
            predicted_labels_per_doc.append(predicted_labels)
            log_progress(log, idx, len(X))

        # Also need to make annotated documents.
        return transform_bio_tags_to_annotated_documents(tokens_per_doc, predicted_labels_per_doc, X)

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        save_path = Path(file_path)
        mkdir(save_path)
        model_save_path = save_path.joinpath("DTree.model")
        config_save_path = save_path.joinpath("DTree.config")
        joblib.dump(self.tree, model_save_path)
        self.config.save(config_save_path)
        self.feature_extractor.save(save_path.joinpath(self.feature_extractor.name))

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model_load_path = load_path.joinpath("DTree.model")
        config_load_path = load_path.joinpath("DTree.config")
        self.tree = joblib.load(model_load_path)
        self.config.load(config_load_path)
        self.feature_extractor.load(load_path.joinpath(self.feature_extractor.name))
        return self
