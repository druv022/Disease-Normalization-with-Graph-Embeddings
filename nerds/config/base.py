import json
from sklearn.base import BaseEstimator

from nerds.config.range import ValidSetRange, ValidFloatRange
from nerds.util.logging import get_logger

log = get_logger()

IGNORED_NAMES = ("entity_labels", "relation_labels", "label_map")


class ModelConfiguration(BaseEstimator):
    """ An abstract handler for model configuration."""

    def __init__(self):
        self.parameters = dict()
        self.ranges = self.ranges()

    def ranges(self):
        raise NotImplementedError

    def set_parameters(self, parameters):
        if not parameters:
            return self
        if not isinstance(parameters, dict):
            raise TypeError("Parameters should be a dictionary of key-value pairs but got '{}'.".format(parameters))
        for name in parameters:
            self.set_parameter(name, parameters[name])
        return self

    def get_parameters(self):
        return self.parameters

    def set_parameter(self, name, value):
        if not self._isvalid(name, value):
            log.debug("Parameter '{}' has value '{}' outside of the valid range, setting the default value".format(
                name, value))
            self.parameters[name] = self._default(name)
        self.parameters[name] = value

    def get_parameter(self, name):
        return self.parameters[name]

    def validate(self):
        for name in self.parameters:
            if name in IGNORED_NAMES:
                continue
            if not self._isvalid(name, self.parameters[name]):
                log.debug("Parameter '{}' has value '{}' outside of the valid range, setting the default value".format(
                    name, self.parameters[name]))
                self.parameters[name] = self._default(name)

    def _isvalid(self, name, value):
        if name in IGNORED_NAMES:
            return True
        if name not in self.ranges:
            log.error("Wrong parameter name: '{}'".format(name))
        return self.ranges[name].contains(value)

    def _default(self, name):
        return self.ranges[name].default_value

    def save(self, file_path):
        with open(file_path, "w") as fp:
            fp.write(json.dumps(self.get_parameters()))

    def load(self, file_path):
        with open(file_path, "r") as fp:
            self.set_parameters(json.loads(fp.read().strip()))
        return self


class CRFModelConfiguration(ModelConfiguration):
    """A handler for CRF model configuration."""

    def __init__(self):
        super().__init__()

    def ranges(self):
        ranges = dict()
        ranges["max_iterations"] = ValidSetRange(range(1, 1000000), 1)
        ranges["c1"] = ValidFloatRange(0, 100, 0.01)
        ranges["c2"] = ValidFloatRange(0, 100, 0.01)
        ranges["window"] = ValidSetRange(range(0, 100), 2)
        ranges["prefix"] = ValidSetRange(range(1, 10), 2)
        ranges["suffix"] = ValidSetRange(range(1, 10), 2)
        return ranges


class BiLSTMModelConfiguration(ModelConfiguration):
    """A handler for CRF model configuration."""

    def __init__(self):
        super().__init__()

    def ranges(self):
        ranges = dict()
        ranges["max_iterations"] = ValidSetRange(range(1, 1000000), 1)
        ranges["num_epochs"] = ValidSetRange(range(1, 1000), 1)
        ranges["batch_size"] = ValidSetRange(range(1, 100), 1)
        ranges["dropout"] = ValidFloatRange(0, 1, 0.5)
        ranges["char_emb_size"] = ValidSetRange(range(1, 100), 32)
        ranges["word_emb_size"] = ValidSetRange(range(1, 1000), 128)
        ranges["char_lstm_units"] = ValidSetRange(range(1, 100), 32)
        ranges["word_lstm_units"] = ValidSetRange(range(1, 1000), 128)
        ranges["use_crf"] = ValidSetRange({False, True}, False)
        ranges["use_char_emb"] = ValidSetRange({False, True}, False)
        ranges["shuffle"] = ValidSetRange({False, True}, False)
        ranges["use_pos_emb"] = ValidSetRange({False, True}, False)
        ranges["pos_emb_size"] = ValidSetRange(range(1, 100), 16)
        return ranges


class BiLSTMModelConfigurationModified(ModelConfiguration):
    """A handler for BiLSTM CRF Modified model configuration."""

    def __init__(self):
        super().__init__()

    def ranges(self):
        ranges = dict()
        ranges["max_iterations"] = ValidSetRange(range(1, 1000000), 1)
        ranges["num_epochs"] = ValidSetRange(range(1, 1000), 1)
        ranges["batch_size"] = ValidSetRange(range(1, 100), 1)
        ranges["dropout"] = ValidFloatRange(0, 1, 0.5)
        ranges["char_emb_size"] = ValidSetRange(range(1, 100), 30)
        ranges["word_emb_size"] = ValidSetRange(range(1, 2048), 200)
        ranges["char_lstm_units"] = ValidSetRange(range(1, 100), 30)
        ranges["word_lstm_units"] = ValidSetRange(range(1, 1000), 100)
        ranges["use_crf"] = ValidSetRange({False, True}, True)
        ranges["use_char_emb"] = ValidSetRange({False, True}, True)
        ranges["shuffle"] = ValidSetRange({False, True}, True)
        ranges["use_pos_emb"] = ValidSetRange({False, True}, True)
        ranges["pos_emb_size"] = ValidSetRange(range(1, 100), 16)
        ranges["use_char_cnn"] = ValidSetRange({False, True}, True)
        ranges["use_char_attention"] = ValidSetRange({False, True}, False)
        ranges["use_word_self_attention"] = ValidSetRange({False, True}, False)
        ranges["use_EL"] = ValidSetRange({False, True}, True)
        return ranges


class SpacyModelConfiguration(ModelConfiguration):
    """A handler for CRF model configuration."""

    def __init__(self):
        super().__init__()

    def ranges(self):
        ranges = dict()
        ranges["max_iterations"] = ValidSetRange(range(1, 1000000), 1)
        ranges["num_epochs"] = ValidSetRange(range(1, 1000), 1)
        ranges["batch_size"] = ValidSetRange(range(1, 100), 1)
        ranges["dropout"] = ValidFloatRange(0, 1, 0.5)
        return ranges


class LogisticRegressionModelConfiguration(ModelConfiguration):
    """A handler for CRF model configuration."""

    def __init__(self):
        super().__init__()

    def ranges(self):
        ranges = dict()
        ranges["max_iterations"] = ValidSetRange(range(1, 1000000), 1)
        ranges["C"] = ValidFloatRange(0, 10, 1)
        return ranges


class SupportVectorModelConfiguration(ModelConfiguration):
    """A handler for CRF model configuration."""

    def __init__(self):
        super().__init__()

    def ranges(self):
        ranges = dict()
        ranges["max_iterations"] = ValidSetRange(range(1, 1000000), 1)
        ranges["C"] = ValidFloatRange(0, 10, 1)
        return ranges


class RandomForestModelConfiguration(ModelConfiguration):
    """A handler for Random Forest model configuration."""

    def __init__(self):
        super().__init__()

    def ranges(self):
        ranges = dict()
        ranges["n_estimators"] = ValidSetRange(range(1, 10000), 9)
        ranges["max_features"] = ValidSetRange(range(1, 10000), 9)
        ranges["min_samples_leaf"] = ValidSetRange(range(1, 10000), 3)
        ranges["random_state"] = ValidSetRange(range(1, 1000000), 2019)
        return ranges


class DecisionTreeModelConfiguration(ModelConfiguration):
    """A handler for Decision Tree model configuration."""

    def __init__(self):
        super().__init__()

    def ranges(self):
        ranges = dict()
        ranges["max_depth"] = ValidSetRange(range(1, 100), 10)
        ranges["min_samples_leaf"] = ValidSetRange(range(1, 10000), 3)
        ranges["random_state"] = ValidSetRange(range(1, 1000000), 2019)
        return ranges


class NaiveBayesModelConfiguration(ModelConfiguration):
    """A handler for CRF model configuration."""

    def __init__(self):
        super().__init__()

    def ranges(self):
        ranges = dict()
        return ranges


class Char2VecModelConfiguration(ModelConfiguration):
    """A handler for CRF model configuration."""

    def __init__(self):
        super().__init__()

    def ranges(self):
        ranges = dict()
        ranges["size"] = ValidSetRange(range(1, 1000), 100)
        ranges["min_count"] = ValidSetRange(range(1, 1000), 5)
        ranges["workers"] = ValidSetRange(range(1, 100), 1)
        ranges["window"] = ValidSetRange(range(2, 100), 5)
        ranges["sample"] = ValidFloatRange(1e-10, 1, 1e-3)
        ranges["min_n"] = ValidSetRange(range(1, 100), 3)
        ranges["max_n"] = ValidSetRange(range(1, 100), 6)
        return ranges


class Word2VecModelConfiguration(ModelConfiguration):
    """A handler for CRF model configuration."""

    def __init__(self):
        super().__init__()

    def ranges(self):
        ranges = dict()
        ranges["size"] = ValidSetRange(range(1, 1000), 100)
        ranges["min_count"] = ValidSetRange(range(1, 1000), 5)
        ranges["workers"] = ValidSetRange(range(1, 100), 1)
        ranges["window"] = ValidSetRange(range(2, 100), 5)
        ranges["sample"] = ValidFloatRange(1e-10, 1, 1e-3)
        return ranges


class WordContextConfiguration(ModelConfiguration):
    """A handler for word context feature extractor configuration."""

    def __init__(self):
        super().__init__()

    def ranges(self):
        ranges = dict()
        ranges["window"] = ValidSetRange(range(0, 10), 1)
        ranges["ngram_slice"] = ValidSetRange(range(1, 10), 1)
        return ranges
