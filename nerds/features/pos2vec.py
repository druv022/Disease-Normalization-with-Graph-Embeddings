from pathlib import Path

from gensim.models.word2vec import Word2Vec

from nerds.config.base import Word2VecModelConfiguration
from nerds.features.base import VectorFeatureExtractor
from nerds.util.file import mkdir
from nerds.util.logging import get_logger, log_progress
from nerds.util.nlp import tokens_to_pos_tags, document_to_tokens

log = get_logger()

KEY = "pos2vec"


class PoS2VecFeatureExtractor(VectorFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.key = KEY
        self.config = Word2VecModelConfiguration()

    def fit(self, X, y=None, size=100, min_count=5, workers=1, window=5, sample=1e-3, skipgram=False):
        """ Trains a Word2vec model on given documents.
            Each document should represent a sentence.
        Args:
            X: list(Document | AnnotatedDocument | list(str))
            y: optional labels
            size: Word vector dimensionality
            min_count: Minimum word count
            workers: Number of threads to run in parallel
            window: Context window size
            sample: Downsample setting for frequent words
            skipgram: Use skip-gram if True and CBOW otherwise
        """
        log.info("Checking parameters...")
        self.config.set_parameters({
            "size": size,
            "min_count": min_count,
            "workers": workers,
            "window": window,
            "sample": sample
        })
        # Get sentences as lists of tokens
        log.info("Tokenizing {} documents...".format(len(X)))
        sentences = []
        for idx, doc in enumerate(X):
            sentences.append(tokens_to_pos_tags(document_to_tokens(doc)))
            log_progress(log, idx, len(X))
        # Initialize and train the model (this will take some time)
        log.info("Training Word2vec on {} sentences...".format(len(X)))
        self.model = Word2Vec(
            sentences,
            workers=self.config.get_parameter("workers"),
            size=self.config.get_parameter("size"),
            min_count=self.config.get_parameter("min_count"),
            window=self.config.get_parameter("window"),
            sample=self.config.get_parameter("sample"),
            sg=1 if skipgram else 0)

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
            for pos_tag in tokens_to_pos_tags(document_to_tokens(doc)):
                if pos_tag in self.model.wv:
                    doc_features.append((pos_tag, self.model.wv[pos_tag]))
            features.append(doc_features)
        return features

    def save(self, file_path):
        save_path = Path(file_path)
        mkdir(save_path)
        model_save_path = save_path.joinpath("pos2vec.model")
        config_save_path = save_path.joinpath("pos2vec.config")
        self.model.save(str(model_save_path))
        self.config.save(config_save_path)

    def load(self, file_path):
        load_path = Path(file_path)
        model_load_path = load_path.joinpath("pos2vec.model")
        config_load_path = load_path.joinpath("pos2vec.config")
        self.model = Word2Vec.load(str(model_load_path))
        self.config.load(config_load_path)
        return self
