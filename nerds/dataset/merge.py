from nerds.doc.annotation import Relation, Normalization, Annotation
from nerds.doc.document import AnnotatedDocument, Document
from nerds.util.logging import get_logger, log_progress

log = get_logger()


def merge_documents(documents, separator=" "):
    """ Merges given documents based on their identifiers, i.e.
        merging [adcetris_1, adcetris_2, adcetris_3, duavee_1, duavee_2]
        would result in [adcetris, duavee] with entity IDs being renumbered respectively.
        Relation and normalization argument IDs are adjusted accordingly.
    Args:
        separator: a separating string for merging
        documents: given documents
    Returns:
    """
    log.info("Merging {} documents".format(len(documents)))
    # build a map
    id_doc_map = dict()
    for document in documents:
        # identifier always ends with _{id}
        parts = document.identifier.split('_')
        doc_id = '_'.join(parts[:-1])
        if doc_id in id_doc_map:
            id_doc_map[doc_id].append(document)
        else:
            id_doc_map[doc_id] = [document]
    # merge documents
    merged_documents = []
    for idx, doc_id in enumerate(id_doc_map):
        merged_document = merge_document(id_doc_map[doc_id], separator)
        if merged_document:
            merged_documents.append(merged_document)
        log_progress(log, idx, len(id_doc_map))
    return merged_documents


def merge_document(documents, separator=" "):
    """ Merges documents in a single document, i.e.
        given documents must be parts of one single document
    Args:
        documents: given documents
        separator: a separating string for merging
    Returns: a single documents obtained by merging given documents
    """
    if not documents:
        return None
    if type(documents) != list:
        documents = list(documents)
    # sort documents by their identifier
    documents.sort(key=lambda document: int(document.identifier.split('_')[-1]))
    first = documents[0]
    parts = first.identifier.split('_')
    doc_id = '_'.join(parts[:-1])
    if type(first) == AnnotatedDocument:
        plain_text = ""
        annotations, relations, normalizations = [], [], []
        ent_id, rel_id, norm_id = 0, 0, 0
        for doc in documents:
            # create entities
            old_new_ent_id_map = dict()
            for ann in doc.annotations:
                ent_id += 1
                old_new_ent_id_map[ann.identifier] = "T{}".format(ent_id)
                annotations.append(
                    Annotation(
                        text=ann.text,
                        label=ann.label,
                        offset=(len(plain_text) + ann.offset[0], len(plain_text) + ann.offset[1]),
                        identifier=old_new_ent_id_map[ann.identifier],
                        discontinued=ann.discontinued,
                        uuid=ann.uuid,
                        score=ann.score))
            # create relations
            for rel in doc.relations:
                rel_id += 1
                relations.append(
                    Relation(
                        label=rel.label,
                        source_id=old_new_ent_id_map[rel.source_id],
                        target_id=old_new_ent_id_map[rel.target_id],
                        identifier="R{}".format(rel_id),
                        source_role=rel.source_role,
                        target_role=rel.target_role,
                        uuid=rel.uuid,
                        score=rel.score))
            # create normalizations
            for norm in doc.normalizations:
                norm_id += 1
                normalizations.append(
                    Normalization(
                        argument_id=old_new_ent_id_map[norm.argument_id],
                        resource_id=norm.resource_id,
                        external_id=norm.external_id,
                        identifier="N{}".format(norm_id),
                        label=norm.label,
                        preferred_term=norm.preferred_term,
                        uuid=norm.uuid,
                        score=norm.score))
            # merge content
            plain_text += doc.plain_text_ + separator
        # make a document
        end_idx = len(plain_text) if separator == "" else -len(separator)
        merged_doc = AnnotatedDocument(
            content=plain_text[:end_idx].encode(first.encoding),
            annotations=annotations,
            relations=relations,
            normalizations=normalizations,
            encoding=first.encoding,
            identifier=doc_id)
    else:
        plain_text = separator.join([doc.plain_text_ for doc in documents])
        merged_doc = Document(content=plain_text.encode(first.encoding), encoding=first.encoding, identifier=doc_id)
    return merged_doc
