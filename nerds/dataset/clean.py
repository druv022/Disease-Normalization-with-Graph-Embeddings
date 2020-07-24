import string

from nerds.doc.annotation import Annotation
from nerds.doc.document import AnnotatedDocument, Document
from nerds.util.logging import get_logger, log_progress
from nerds.dataset.split import relations_of, normalizations_of
from nerds.util.string import normalize_whitespaces

log = get_logger()


def clean_documents(documents, normalizer=normalize_whitespaces, skip_empty=True):
    log.info("Cleaning {} documents".format(len(documents)))
    cleaned_documents = []
    for idx, document in enumerate(documents):
        # skip empty-text documents if required
        if skip_empty and not document.plain_text_.strip():
            continue
        cleaned_documents += [clean_document(document, normalizer)]
        log_progress(log, idx, len(documents))
    return cleaned_documents


def clean_annotated_documents(annotated_documents, normalizer=normalize_whitespaces, skip_empty=True):
    """
    Args:
        annotated_documents: documents to normalize
        normalizer (function, optional): The method to normalize found sentences.
        skip_empty (boolean): True if empty documents should be skipped

    Returns:
    """
    log.info("Cleaning {} documents".format(len(annotated_documents)))
    cleaned_documents = []
    for idx, document in enumerate(annotated_documents):
        # skip empty-text documents if required
        if skip_empty and not document.plain_text_.strip():
            continue
        cleaned_documents += [clean_annotated_document(document, normalizer, skip_empty)]
        log_progress(log, idx, len(annotated_documents))
    return cleaned_documents


def clean_document(document, normalizer=normalize_whitespaces):
    normalized_text = normalizer(document.plain_text_)
    return Document(
        content=normalized_text.encode(document.encoding),
        encoding=document.encoding,
        identifier=document.identifier)


def clean_annotated_document(document, normalizer=normalize_whitespaces, skip_empty=True):
    # normalize text
    if type(document) == Document:
        return clean_document(document, normalizer=normalizer)
    if type(document) != AnnotatedDocument:
        raise TypeError("The input does not seem to be a document")
    normalized_text = normalizer(document.plain_text_)
    cleaned_annotations = clean_annotation_text(document.annotations, normalizer=normalizer, skip_empty=skip_empty)
    adjusted_annotations = adjust_annotation_offsets(cleaned_annotations, normalized_text, document.plain_text_)
    return AnnotatedDocument(
        content=normalized_text.encode(document.encoding),
        annotations=adjusted_annotations,
        relations=relations_of(adjusted_annotations, document),
        normalizations=normalizations_of(adjusted_annotations, document),
        encoding=document.encoding,
        identifier=document.identifier)


def clean_annotation_text(annotations, normalizer=normalize_whitespaces, skip_empty=True):
    cleaned_annotations = []
    for ann in annotations:
        ann_text = normalizer(ann.text)
        if skip_empty and not ann_text.strip():
            continue
        cleaned_annotations += [
            Annotation(
                text=ann_text,
                label=ann.label,
                offset=ann.offset,
                identifier=ann.identifier,
                discontinued=ann.discontinued,
                uuid=ann.uuid,
                score=ann.score)
        ]

    return cleaned_annotations


def adjust_annotation_offsets(annotations, normalized_text, original_text):
    # map old word offsets to new offsets
    # it's ok if a word contains punctuations, e.g. "something,"
    words = normalized_text.split()
    new_from, old_from = 0, 0
    old_new_index_map = dict()
    for word in words:
        new_start = normalized_text.find(word, new_from)
        old_start = original_text.find(word, old_from)
        for i in range(len(word)):
            old_new_index_map[old_start + i] = new_start + i
        new_from = new_start + len(word)
        old_from = old_start + len(word)

    adjusted_annotations = []
    for ann in annotations:
        if ann.offset[0] not in old_new_index_map \
                or ann.offset[1] not in old_new_index_map \
                or ann.offset[0] < 0 or ann.offset[1] < 0 \
                or ann.offset[0] > ann.offset[1]:
            start, end = find_in(ann.text, original_text, discontinued=True)
            if start >= 0:
                log.debug("Incorrect offsets for entity '{}': {}. Attempting to fix... Fixed.".format(
                    ann.text, ann.offset))
                ann.offset = (start, end)
            else:
                log.debug("Incorrect offsets for entity '{}': {}. Attempting to fix... Failed. Skipping.".format(
                    ann.text, ann.offset))
                continue

        adjusted_annotations += [
            Annotation(
                text=ann.text,
                label=ann.label,
                offset=(old_new_index_map[ann.offset[0]], old_new_index_map[ann.offset[1]]),
                identifier=ann.identifier,
                discontinued=ann.discontinued,
                uuid=ann.uuid,
                score=ann.score)
        ]

    return adjusted_annotations


def whitespaces(text, start, end):
    # exlcluding end
    for i in range(start, min(end, len(text))):
        if text[i] not in string.whitespace:
            return False
    return True


def find_in(target, text, discontinued=False):
    # search the whole string
    start_first = text.find(target)
    if start_first >= 0:
        # the first and last character
        return start_first, start_first + len(target) - 1

    # search the first word
    words = target.split()
    start_first = text.find(words[0])

    # find all words in their order (with no words in between if discontinued=False)
    while start_first >= 0:
        start_prev = start_first
        for i in range(1, len(words)):
            start_curr = text.find(words[i], start_prev + len(words[i - 1]))
            if start_curr < 0:
                return -1, -1

            if not discontinued and not whitespaces(text, start_prev + len(words[i - 1]), start_curr):
                start_first = text.find(words[0], start_first + len(words[0]))
                break
            elif i == len(words) - 1:
                return start_first, start_curr + len(words[i]) - 1

            start_prev = start_curr

    return -1, -1


def remove_duplicates(annotated_documents):
    unique_doc_set = set()
    for doc in annotated_documents:
        unique_doc_set.add(doc)
    dupl_doc_list = return_inconsistencies(annotated_documents, only_ne_dupl=True)
    for doc in dupl_doc_list:
        if doc in unique_doc_set:
            unique_doc_set.remove(doc)
    return list(unique_doc_set)


def return_inconsistencies(annotated_documents, only_ne_dupl=False):
    text_doc_map = dict()
    singles = set()

    for doc in annotated_documents:
        ann_set = set(ann.offset for ann in doc.annotations)
        # ann_value = sorted([(ann.text,ann.offset) for ann in ann_set])
        ann_value = sorted(ann_set)

        if doc.plain_text_ not in text_doc_map.keys():
            text_doc_map[doc.plain_text_] = {}
            text_doc_map[doc.plain_text_][doc] = ann_value
            singles.add(doc.plain_text_)

        else:
            if doc not in text_doc_map[doc.plain_text_]:
                if doc.plain_text_ in singles:
                    singles.remove(doc.plain_text_)
                text_doc_map[doc.plain_text_][doc] = ann_value

    # Removing document-plain_text pairs which uniquely map to one another.
    for key in singles:
        del text_doc_map[key]

    # Removing sentences with duplicate NE ann, but which have different Relation annotations.
    dupl_docs = []
    for plain_doc in text_doc_map.keys():
        ann_set = set()
        for doc in text_doc_map[plain_doc]:
            ann_value = text_doc_map[plain_doc][doc]  # Recall, ann_value is a sorted list of offsets
            if tuple(ann_value) in ann_set:
                dupl_docs.append(doc)  # collects duplicate docs, but not the first instances
            else:
                ann_set.add(tuple(ann_value))

    if only_ne_dupl:
        return dupl_docs

    else:
        for doc in dupl_docs:
            del text_doc_map[doc.plain_text_]
        return text_doc_map, dupl_docs


def resolve_overlaps(annotated_documents):
    """ Takes a list of AnnotatedDocuments
        Returns a list of AnnotatedDocuments in which overlapping offsets have been resolved
        which in this case means the offsets with the largest span
    """
    result_documents = []
    for document in annotated_documents:
        annotations_to_remove = set()
        for ann in document.annotations:
            for other_ann in document.annotations:
                if ann == other_ann:
                    continue
                if other_ann.offset[0] <= ann.offset[0] and ann.offset[1] <= other_ann.offset[1]:
                    annotations_to_remove.add(ann)
                    break

        result_documents.append(
            AnnotatedDocument(content=document.content,
                              encoding=document.encoding,
                              identifier=document.identifier,
                              annotations=[ann for ann in document.annotations if ann not in annotations_to_remove],
                              normalizations=document.normalizations,
                              relations=document.relations))
    return result_documents
