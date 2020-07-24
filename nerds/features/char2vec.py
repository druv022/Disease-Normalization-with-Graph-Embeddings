from pathlib import Path

from gensim.models import FastText

from nerds.config.base import Char2VecModelConfiguration
from nerds.features.base import VectorFeatureExtractor
from nerds.util.file import mkdir
from nerds.util.logging import get_logger, log_progress
from nerds.util.nlp import document_to_tokens

log = get_logger()

KEY = "char2vec"


class Char2VecFeatureExtractor(VectorFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.key = KEY
        self.config = Char2VecModelConfiguration()

    def fit(self, X, y=None, size=100, min_count=5, workers=1, window=5, sample=1e-3, skipgram=False, min_n=3, max_n=6):
        """ Trains a Word2vec model on given documents.
            Each document should represent a sentence.
        Args:
            X: list(Document | AnnotatedDocument | list(str))
            y: optional labels
            size: Size of embeddings to be learnt (Default 100), i.e. word vector dimensionality
            min_count: Minimum word count. Ignore words with number of occurrences below this (Default 5).
            workers: Number of threads to run in parallel
            window: Context window size
            sample: Threshold for downsampling higher-frequency words (Default 0.001)
            skipgram: Use skip-gram if True and CBOW otherwise
            min_n: min length of char ngrams (Default 3)
            max_n: max length of char ngrams (Default 6)
        """
        log.info("Checking parameters...")
        self.config.set_parameters({
            "size": size,
            "min_count": min_count,
            "workers": workers,
            "window": window,
            "sample": sample,
            "min_n": min_n,
            "max_n": max_n
        })
        self.config.validate()
        # Get sentences as lists of tokens
        log.info("Tokenizing {} documents...".format(len(X)))
        sentences = []
        for idx, doc in enumerate(X):
            sentences.append(document_to_tokens(doc))
            log_progress(log, idx, len(X))
        # Initialize and train the model (this will take some time)
        log.info("Training FastText on {} sentences...".format(len(X)))
        self.model = FastText(
            sentences,
            workers=self.config.get_parameter("workers"),
            size=self.config.get_parameter("size"),
            min_count=self.config.get_parameter("min_count"),
            window=self.config.get_parameter("window"),
            sample=self.config.get_parameter("sample"),
            sg=1 if skipgram else 0,
            min_n=self.config.get_parameter("min_n"),
            max_n=self.config.get_parameter("max_n"))

        # If you don't plan to train the model any further, calling
        # init_sims() will make the model much more memory-efficient.
        self.model.init_sims(replace=True)
        return self

    def transform(self, X, y=None):
        """ Transforms the list of documents and returns tokens with their features.
            Each document should represent a sentence.
        """
        log.info("Generating features for {} documents...".format(len(X)))
        features = []
        for doc in X:
            doc_features = []
            for token in document_to_tokens(doc):
                if token in self.model.wv:
                    doc_features.append((token, self.model.wv[token]))
            features.append(doc_features)
        return features

    def save(self, file_path):
        save_path = Path(file_path)
        mkdir(save_path)
        model_save_path = save_path.joinpath("char2vec.model")
        config_save_path = save_path.joinpath("char2vec.config")
        self.model.save(str(model_save_path))
        self.config.save(config_save_path)

    def load(self, file_path):
        load_path = Path(file_path)
        model_load_path = load_path.joinpath("char2vec.model")
        config_load_path = load_path.joinpath("char2vec.config")
        self.model = FastText.load(str(model_load_path))
        self.config.load(config_load_path)
        return self
