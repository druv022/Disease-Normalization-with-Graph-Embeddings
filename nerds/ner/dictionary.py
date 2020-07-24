import ahocorasick as alg
import string
from pathlib import Path
from copy import deepcopy

from nerds.doc.annotation import Annotation
from nerds.doc.document import AnnotatedDocument
from nerds.ner.base import NamedEntityRecognitionModel
from nerds.util.logging import get_logger, log_progress

log = get_logger()

KEY = "dict_ner"


class ExactMatchDictionaryNER(NamedEntityRecognitionModel):
    """ Annotates the list of `Document` objects that are provided as
        input and returns a list of `AnnotatedDocument` objects.
        In a dictionary based approach, a dictionary of keywords is used
        to create a FSA which is then used to search with. See [1].
        [1]: https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm
    """
    def __init__(self, entity_labels, static=False):
        super().__init__(entity_labels)
        if len(entity_labels) != 1:
            raise TypeError("Entity labels must contain exactly one label for a dictionary but got {} instead"
                            .format(len(entity_labels)))
        self.key = KEY
        self._vocabulary = None
        self.static = static

    def _create_automaton(self):
        self._automaton = alg.Automaton()
        for idx, entity in enumerate(self._vocabulary):
            self._automaton.add_word(entity, (idx, entity))
        self._automaton.make_automaton()

    def fit(self, X, y=None):
        """ Given annotated documents, generates a dictionary for a given entity type.
            Dictionaries are "expanded" to include the following formats:
                ALL UPPERCASE
                all lowercase
                First letter uppercase
                First Letter Of Each Word Uppercase
        """
        log.info("Creating a dictionary from given data")

        if self.static:
            log.info("Dictionary is static: fitting to data will not have any effect.")
            return self

        # get all entity terms for a given type and create a vocabulary
        entity_set = set()
        for doc in X:
            for ann in doc.annotations:
                if ann.label == self.entity_labels[0]:
                    entity_set.add(ann.text)
                    entity_set.add(ann.text.lower())
                    entity_set.add(ann.text.upper())
                    entity_set.add(ann.text.capitalize())
                    entity_set.add(string.capwords(ann.text))

        # make a vocabulary
        self._vocabulary = list(entity_set)
        self._vocabulary.sort()

        # create an automaton
        self._create_automaton()
        return self

    def transform(self, X, y=None):
        log.info("Annotating named entities in {} documents with ExactMatchDictionaryNER...".format(len(X)))
        annotated_documents = []
        for idx, document in enumerate(X):
            annotations = []
            doc_content_str = document.plain_text_
            for item in self._automaton.iter(doc_content_str):
                end_position, (index, word) = item
                start_position = (end_position - len(word) + 1)

                # check if a candidate is a strict substring of some words,
                # e.g. "abc de" is a substring of "aaabc deeee"
                if start_position > 0 \
                        and doc_content_str[start_position - 1].isalpha():
                    continue
                if end_position < len(doc_content_str) - 1 \
                        and doc_content_str[end_position + 1].isalpha():
                    continue

                # add all given entity labels as types
                for label in self.entity_labels:
                    annotations.append(Annotation(text=word, label=label, offset=(start_position, end_position)))

            # remove annotations from inside other annotations
            annotations_to_remove = set()
            for ann in annotations:
                for other_ann in annotations:
                    if ann == other_ann:
                        continue
                    if other_ann.offset[0] <= ann.offset[0] and ann.offset[1] <= other_ann.offset[1]:
                        annotations_to_remove.add(ann)
            for ann in annotations_to_remove:
                annotations.remove(ann)

            # make a document
            annotated_documents.append(
                AnnotatedDocument(
                    content=document.content,
                    annotations=annotations,
                    relations=document.relations if type(document) == AnnotatedDocument else [],
                    normalizations=document.normalizations if type(document) == AnnotatedDocument else [],
                    encoding=document.encoding,
                    identifier=document.identifier,
                    uuid=document.uuid))
            log_progress(log, idx, len(X))

        return annotated_documents

    def save(self, file_path):
        """ Saves a vocabulary to the local disk, provided a file path. """
        save_path = Path(file_path)
        save_path.write_text("\n".join(self._vocabulary), encoding="utf-8")

    def load(self, file_path):
        """ Loads a vocabulary saved locally. """
        load_path = Path(file_path)
        vocab_text = load_path.read_text(encoding="utf-8")
        self._vocabulary = [line.strip() for line in vocab_text.split("\n")]
        self._create_automaton()
        return self

    def clone(self):
        return deepcopy(self)
