from collections import Iterable

from nerds.dataset.clean import clean_annotated_documents, remove_duplicates, remove_inconsistencies
from nerds.dataset.merge import merge_documents
from nerds.dataset.split import split_annotated_documents
from nerds.dataset.stats import count_annotated_documents, count_annotations, \
    get_distinct_labels, get_distinct_entities
from nerds.util.string import normalize_whitespaces


class Dataset(object):

    def __init__(self, documents):
        if not isinstance(documents, Iterable):
            raise TypeError("Documents are not iterable: check your input")
        self.documents = [doc for doc in documents]
        self.stats = Statistics(self.documents)

    def normalize(self, normalizer=normalize_whitespaces):
        self.documents = clean_annotated_documents(self.documents, normalizer=normalizer)
        return self

    def split(self, method="nltk"):
        self.documents = split_annotated_documents(self.documents, method=method)
        return self

    def merge(self, separator=" "):
        self.documents = merge_documents(self.documents, separator=separator)
        return self

    def remove_duplicates(self):
        self.documents = remove_duplicates(self.documents)
        return self

    def remove_inconsistencies(self):
        self.documents = remove_inconsistencies(self.documents)
        return self

    def cook(self):
        return self.split().normalize().remove_duplicates().remove_inconsistencies().documents


class Statistics(object):

    def __init__(self, documents):
        self.documents = documents

    def count_distinct_entities(self, entity_type=None):
        return len(get_distinct_entities(self.documents, entity_type=entity_type))

    @property
    def doc_count(self):
        return len(self.documents)

    @property
    def ent_doc_count(self):
        return count_annotated_documents(self.documents, annotation_type="annotation")

    @property
    def rel_doc_count(self):
        return count_annotated_documents(self.documents, annotation_type="relation")

    @property
    def ent_count(self):
        return count_annotations(self.documents, annotation_type="annotation")

    @property
    def rel_count(self):
        return count_annotations(self.documents, annotation_type="relation")

    @property
    def ent_type_count(self):
        return len(get_distinct_labels(self.documents, annotation_type="annotation"))

    @property
    def rel_type_count(self):
        return len(get_distinct_labels(self.documents, annotation_type="relation"))

    @property
    def ent_value_count(self):
        return len(get_distinct_entities(self.documents))
