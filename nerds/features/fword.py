from pathlib import Path

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

from nerds.config.base import WordContextConfiguration
from nerds.doc.bio import transform_annotated_documents_to_bio_format
from nerds.features.base import FeatureExtractor, UNKNOWN_WORD
from nerds.util import string
from nerds.util.file import mkdir
from nerds.util.logging import get_logger
from nerds.util.nlp import tokens_to_pos_tags

log = get_logger()

KEY = "fword"


class WordFeatureExtractor(FeatureExtractor):
    def __init__(self, ngram_slice=1):
        super().__init__()
        self.key = KEY
        self.encoders = dict()
        self.config = WordContextConfiguration()
        self.config.set_parameter("ngram_slice", ngram_slice)

    def transform(self, X, y=None, entity_labels=None):
        """ Transforms the list of `Document` objects that are provided as
            input to the BIO format and returns features, tokens, and labels per word.
            Features of a word stored as a list of values.
        """
        log.info("Generating features for {} documents...".format(len(X)))
        tokens_per_doc, labels_per_doc = \
            transform_annotated_documents_to_bio_format(X, entity_labels=entity_labels)
        tokens_flat = [token for tokens in tokens_per_doc for token in tokens]
        labels_flat = [label for labels in labels_per_doc for label in labels]
        pos_tags_flat = [pos_tag for tokens in tokens_per_doc for pos_tag in tokens_to_pos_tags(tokens)]

        features_flat = [self._word_to_features(token) for token in tokens_flat]
        for word_features, pos_tag in zip(features_flat, pos_tags_flat):
            word_features.append(pos_tag)

        if not self.encoders:
            # first time run
            for idx in range(len(features_flat[0])):
                if isinstance(features_flat[0][idx], str):
                    self.encoders[idx] = LabelEncoder()
                    column_vector = [features_flat[i][idx] for i in range(len(features_flat))]
                    column_vector.append(UNKNOWN_WORD)
                    self.encoders[idx].fit(column_vector)

        for idx in range(len(features_flat[0])):
            if idx in self.encoders:
                column_vector = [features_flat[i][idx] for i in range(len(features_flat))]
                self._process_unknown_values(column_vector, self.encoders[idx].classes_.tolist(), UNKNOWN_WORD)
                column_vector = self.encoders[idx].transform(column_vector).tolist()
                for i in range(len(features_flat)):
                    features_flat[i][idx] = column_vector[i]

        return features_flat, tokens_flat, labels_flat

    def _process_unknown_values(self, entries, vocabulary, unknown_label):
        for i in range(len(entries)):
            if entries[i] not in vocabulary:
                entries[i] = unknown_label

    def _word_to_features(self, word):
        """ Extracts features from a given word.
        """
        features = [int(word.isupper()), int(word.islower()), int(word.istitle()),
                    int(word.isdigit()), int(string.ispunct(word)), len(word)]
        ngram_slice = self.config.get_parameter("ngram_slice")
        for n in range(1, 4):
            ngrams = self._char_ngrams(word, n=n)
            if len(ngrams) < ngram_slice:
                ngrams.extend(ngrams[-1] * (ngram_slice - len(ngrams)))
            features.extend(ngrams[:ngram_slice] + ngrams[-ngram_slice:])
        return features

    def _char_ngrams(self, word, n=2):
        """ Returns character n-grams of a given word
        """
        if n <= 1:
            return [ch for ch in word]
        elif n < len(word):
            return [word[i: i + n] for i in range(len(word) - n)]
        else:
            return [word]

    def save(self, file_path):
        save_path = Path(file_path)
        mkdir(save_path)
        # save config
        config_save_path = save_path.joinpath("context.config")
        self.config.save(config_save_path)
        # save dictionaries
        for i in self.encoders:
            words_path = save_path.joinpath("f{}.dict".format(i))
            joblib.dump(self.encoders[i], words_path)

    def load(self, file_path):
        load_path = Path(file_path)
        # load config
        config_load_path = load_path.joinpath("context.config")
        self.config.load(config_load_path)
        # load dictionaries
        for dict_file in load_path.glob("*.dict"):
            idx = int(dict_file.name.split(".")[0][1:])
            self.encoders[idx] = joblib.load(dict_file)
        return self
