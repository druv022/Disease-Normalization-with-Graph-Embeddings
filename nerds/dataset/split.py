from nerds.doc.annotation import Annotation
from nerds.doc.document import Document, AnnotatedDocument
from nerds.util.logging import get_logger, log_progress
from nerds.util.nlp import text_to_sentences

log = get_logger()


def split_annotated_documents(annotated_documents, splitter=text_to_sentences, *args):
    """ Wrapper function that applies `split_annotated_document` to a
                batch of documents.
        """
    log.info("Splitting {} documents".format(len(annotated_documents)))
    result_ann = []
    for idx, annotated_document in enumerate(annotated_documents):
        result_ann.extend(split_annotated_document(annotated_document, splitter, *args))
        # info
        log_progress(log, idx, len(annotated_documents))
    return result_ann


def split_annotated_document(annotated_document, splitter=text_to_sentences, *args):
    """ Splits an annotated document and maintains the annotation offsets.

                This function accepts an AnnotatedDocument object as parameter along
                with an optional tokenization method. It splits the document according
                to the tokenization method, and returns a list of AnnotatedDocument
                objects, where the annotation offsets have been adjusted.

                Args:
                        annotated_document (AnnotatedDocument): The document that will be
                                split into more documents.
                        splitter: (func, optional): A function that accepts a string as
                                input and returns a list of strings. Defaults to
                                `document_to_sentences`, which is the default sentence splitter
                                for this library.

                Returns:
                        list(AnnotatedDocument): A list of annotated documents.
        """
    if type(annotated_document) == Document:
        return split_document(annotated_document, splitter=splitter, *args)

    if type(annotated_document) != AnnotatedDocument:
        raise TypeError("The input does not seem to be a document")

    snippet_docs = []
    snippets, offsets = splitter(annotated_document.plain_text_, *args)
    annotated_snippets = []
    for idx, snippet in enumerate(snippets):
        annotations = []
        for ann in annotated_document.annotations:
            if ann.label == 'O':
                continue
            if ann.offset[0] >= offsets[idx][0]:
                if ann.offset[1] <= offsets[idx][1]:
                    annotations += [ann]
                else:
                    break

        annotations.sort()
        annotated_snippets += [(snippet, offsets[idx], annotations)]

    # check if all annotations are included
    count = 0
    for _, _, annotations in annotated_snippets:
        count += len(annotations)
    if len(annotated_document.annotations) != count:
        log.debug("Document has {} annotations but only {} of them are found".format(
            len(annotated_document.annotations), count))

    # adjust annotation offsets and create documents
    idx = 0
    for snippet, offset, annotations in annotated_snippets:

        # find entities in a sentence
        annotations_refined = []
        for ann in annotations:

            annotations_refined += [
                Annotation(
                    text=ann.text,
                    label=ann.label,
                    offset=(ann.offset[0] - offset[0], ann.offset[1] - offset[0]),
                    identifier=ann.identifier,
                    discontinued=ann.discontinued,
                    uuid=ann.uuid,
                    score=ann.score)
            ]

        idx += 1
        snippet_docs += [
            AnnotatedDocument(
                content=snippet.encode(annotated_document.encoding),
                annotations=annotations_refined,
                relations=relations_of(annotations_refined, annotated_document),
                normalizations=normalizations_of(annotations_refined, annotated_document),
                encoding=annotated_document.encoding,
                identifier="{}_{}".format(annotated_document.identifier, idx))
        ]

    return snippet_docs


def relations_of(annotations, document):
    relations = []
    entity_ids = set()
    for ann in annotations:
        entity_ids.add(ann.identifier)

    for relation in document.relations:
        if relation.source_id in entity_ids and relation.target_id in entity_ids:
            relations += [relation]

    return relations


def normalizations_of(annotations, document):
    normalizations = []
    entity_ids = set()
    for ann in annotations:
        entity_ids.add(ann.identifier)

    for norm in document.normalizations:
        if norm.argument_id in entity_ids:
            normalizations += [norm]

    return normalizations


def split_documents(documents, splitter=text_to_sentences, *args):
    """ Wrapper function that applies `split_document` to a
        batch of documents.
    """
    log.info("Splitting {} documents".format(len(documents)))
    result = []
    for idx, document in enumerate(documents):
        result.extend(split_document(document, splitter, *args))
        # info
        log_progress(log, idx, len(documents))
    return result


def split_document(document, splitter=text_to_sentences, *args):
    """ Splits a document and creates AnnotatedDocument objects
        to be populated with annotations.

                This function accepts a Document object as parameter along
                with an optional tokenization method. It splits the document according
                to the tokenization method, and returns a list of AnnotatedDocument
                objects, with empty annotations, relations, normalisations.

                Args:
                        document (Document): The document that will be
                                split into more documents.
                        splitter: (func, optional): A function that accepts a string as
                                input and returns a list of strings. Defaults to
                                `text_to_sentences`, which is the default sentence splitter
                                for this library.

                Returns:
                        list(Document): A list of documents.
        """
    snippet_docs = []
    snippets, _ = splitter(document.plain_text_, *args)
    # Generate AnnotatedDocuments per snippet
    for idx, snippet in enumerate(snippets):
        snippet_docs += [
            AnnotatedDocument(
                content=snippet.encode(document.encoding),
                encoding=document.encoding,
                identifier="{}_{}".format(document.identifier, idx))
        ]

    return snippet_docs
