from nerds.dataset.clean import clean_document
from nerds.dataset.split import split_document
from nerds.util.nlp import text_to_sentences
from nerds.util.string import normalize_whitespaces


class DocumentProcessor(object):

    def process(self, documents):
        raise NotImplementedError


class DocumentSplitter(DocumentProcessor):

    def __init__(self, method='nltk'):
        self.method = method

    def process(self, document):
        return split_document(document, text_to_sentences, self.method)


class DocumentCleaner(DocumentProcessor):

    def __init__(self, normalizer=normalize_whitespaces):
        self.normalizer = normalizer

    def process(self, document):
        return clean_document(document, normalizer=self.normalizer)
