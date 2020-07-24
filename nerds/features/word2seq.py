from nerds.features.base import FeatureExtractor
from nerds.util import string
from nerds.doc.bio import transform_annotated_documents_to_bio_format
from nerds.util.logging import get_logger
from nerds.util.nlp import tokens_to_pos_tags

log = get_logger()

KEY = "word2seq"


class WordSequenceFeatureExtractor(FeatureExtractor):
    def __init__(self, window=2, prefix=2, suffix=2):
        super().__init__()
        self.key = KEY
        self.window = window
        self.prefix = prefix
        self.suffix = suffix

    def transform(self, X, y=None, entity_labels=None):
        """ Transforms the list of `Document` objects that are provided as
            input to the BIO format and returns features, tokens, and labels per word.
            Features of a word stored as a dictionary of name-value pairs.
        """
        tokens_per_doc, labels_per_doc = \
            transform_annotated_documents_to_bio_format(X, entity_labels=entity_labels)
        pos_tags_per_doc = []
        for tokens in tokens_per_doc:
            pos_tags_per_doc.append(tokens_to_pos_tags(tokens))
        sentences = []
        for i in range(len(tokens_per_doc)):
            sentence = [(text, pos, label)
                        for text, pos, label in zip(tokens_per_doc[i], pos_tags_per_doc[i], labels_per_doc[i])]
            sentences.append(sentence)

        features_per_doc = [self._sent_to_features(s) for s in sentences]
        labels_per_doc = [self._sent_to_labels(s) for s in sentences]

        return features_per_doc, tokens_per_doc, labels_per_doc

    def _sent_to_features(self, sent):
        return [self._word_to_features(sent, i) for i in range(len(sent))]

    def _sent_to_labels(self, sent):
        return [label for token, postag, label in sent]

    def _word_to_features(self, sent, i):
        """ Given a sequence of words (sentence), extract features from it.
        """
        # As default, we have:
        # 1) A window size of 2, so 2 words before and 2 words after.
        # 2) Prefix and suffix of size 2 by default.
        # 3) The word itself, lowercase.
        # 4) Boolean flags for the word: isupper, islower, istitle, isdigit, ispunct.
        # 5) POS tags.
        features = dict()
        features['bias'] = 1.0
        features['BOS'] = True if i == 0 else False
        features['EOS'] = True if i == len(sent) - 1 else False
        # window features
        for j in range(max(i - self.window, 0), min(i + self.window, len(sent) - 1)):
            word = sent[j][0]
            postag = sent[j][1]
            features.update({
                '{}:suffix'.format(j - i): word[-self.suffix:],
                '{}:prefix'.format(j - i): word[:self.prefix],
                '{}:word.lower()'.format(j - i): word.lower(),
                '{}:word.istitle()'.format(j - i): word.istitle(),
                '{}:word.islower()'.format(j - i): word.islower(),
                '{}:word.isupper()'.format(j - i): word.isupper(),
                '{}:word.isdigit()'.format(j - i): word.isdigit(),
                '{}:word.ispunct()'.format(j - i): string.ispunct(word),
                '{}:postag'.format(j - i): postag
            })
        return features
