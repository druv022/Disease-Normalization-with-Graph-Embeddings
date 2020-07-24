import regex as re
from nerds.doc.annotation import Annotation
from nerds.dataset.split import split_annotated_document
from nerds.util.nlp import text_to_tokens, get_partition_offsets, text_to_sentences
from nerds.util.logging import get_logger, log_progress

log = get_logger()


NATURAL_LANGUAGE = "NATURAL_LANGUAGE"
TABLE = "TABLE"
BULLET_LIST = "BULLET_LIST"

BULLET_RE = re.compile('[ \t]*\*|[ \t]*EXCERPT')


def is_table_header(line):
    words = line.split()
    if len(words) < 2:
        return False
    if words and words[0] == "Table":
        if len(words) == 2:
            return True
        if words[1][-1] == ":" and words[1][-2].isdigit():
            return True
        if words[1][-1] == "." and words[1][-2].isdigit():
            return True
        if words[1].isdigit() and (words[2] == ":" or words[2] == "."):
            return True
        if words[1].isdigit() and words[2][0].isupper() and words[2] != "Table":
            return True
    return False


def is_well_formed_sentence(line):
    text = line.strip()
    if not text:
        return False
    if text[0].islower():
        return False
    if not text[0].isalpha():
        return False
    if text[-1].isdigit():
        return False
    if text.startswith("Notes:"):
        return False
    if text[-1] == "." or text[-1] == ":":
        return True
    tokens = text_to_tokens(line)
    for i in range(len(tokens) - 1):
        if tokens[i] == "." and tokens[i + 1][0].isupper():
            return True
    return False


def is_section(line):
    words = line.split()
    if len(words) < 2:
        return False
    if not words[0][0].isdigit():
        return False
    if not words[0][-1].isdigit():
        return False
    if not words[1][0].isupper():
        return False
    joined_exc_first = "".join([words[i] for i in range(1, len(words))])
    if not joined_exc_first.isalpha():
        return False
    return True


def find_tables(text):
    tables = []
    lines = text.split("\n")
    table_lines = []
    for line in lines:
        if is_table_header(line):
            if table_lines:
                tables.append("\n".join(table_lines))
                table_lines = []
            table_lines.append(line)
        elif table_lines:
            if is_section(line) or (is_well_formed_sentence(line) and len(table_lines) > 5):
                tables.append("\n".join(table_lines))
                table_lines = []
            else:
                table_lines.append(line)
    offsets = get_partition_offsets(text, tables)
    return tables, offsets


def is_short_bullet(line):
    return ' *' in line and len(line) < 150 and \
           re.search('Warnings and Precautions|Dosage and Administration', line) and BULLET_RE.match(line)


def find_bullets(text):
    bullets = [line for line in text.split("\n") if is_short_bullet(line)]
    offsets = get_partition_offsets(text, bullets)
    return bullets, offsets


def categorise_content(text):
    anns = []
    tables, offsets = find_tables(text)
    if tables:
        for table, offset in zip(tables, offsets):
            anns.append(Annotation(text=table, label=TABLE, offset=offset))

    bullets, offsets = find_bullets(text)
    if bullets:
        for bullet, offset in zip(bullets, offsets):
            anns.append(Annotation(text=bullet, label=BULLET_LIST, offset=offset))
    anns.sort()
    # resolve overlaps between tables and bullet lists
    resolve_overlaps(anns)

    if not anns:
        return [Annotation(text=text, label=NATURAL_LANGUAGE, offset=(0, len(text) - 1))]
    # add natural language paragraphs
    anns_nl = []
    text_nl = text[: anns[0].offset[0]]
    if text_nl:
        anns_nl.append(Annotation(text=text_nl, label=NATURAL_LANGUAGE, offset=(0, anns[0].offset[0] - 1)))
    for i in range(len(anns) - 1):
        text_nl = text[anns[i].offset[1] + 1: anns[i + 1].offset[0]]
        if text_nl:
            anns_nl.append(Annotation(text=text_nl, label=NATURAL_LANGUAGE,
                                      offset=(anns[i].offset[1] + 1, anns[i + 1].offset[0] - 1)))
    text_nl = text[anns[-1].offset[1] + 1:]
    if text_nl:
        anns_nl.append(Annotation(text=text_nl, label=NATURAL_LANGUAGE,
                                  offset=(anns[-1].offset[1] + 1, len(text) - 1)))
    anns.extend(anns_nl)
    anns.sort()
    return anns


def resolve_overlaps(annotations):
    removals = []
    for i in range(len(annotations) - 1):
        if annotations[i].offset[0] >= annotations[i + 1].offset[0] and \
                annotations[i].offset[1] <= annotations[i + 1].offset[1]:
            removals.append(annotations[i])
        if annotations[i + 1].offset[0] >= annotations[i].offset[0] and \
                annotations[i + 1].offset[1] <= annotations[i].offset[1]:
            removals.append(annotations[i + 1])
    for rem in removals:
        annotations.remove(rem)


def text_to_paragraphs(text, method=None):
    anns = categorise_content(text)
    snippets = [ann.text for ann in anns]
    offsets = [ann.offset for ann in anns]
    return snippets, offsets


def table_to_sentences(text, method=None):
    sentences = text.split("\n")
    offsets = get_partition_offsets(text, sentences)
    return sentences, offsets


def split_fda_documents(annotated_documents, sent_type=NATURAL_LANGUAGE, with_bullets=False):
    """
    Functionality to retrieve different types of content from the fda labels.
    It splits annotated_documents as per the sentence_type "sent_type".
    Args:
         annotated_documents: documents to be split by content type
         sent_type: (str) "nl"|"table"|"bullet"

    Returns:
        a list of annotated documents of content type 'sent_type'
    """
    log.info("Splitting {} documents".format(len(annotated_documents)))
    result_ann = []
    for idx, annotated_document in enumerate(annotated_documents):
        split_docs, split_doc_types = split_fda_document(annotated_document, with_bullets)
        x = [split_docs[i] for i, doc_type in enumerate(split_doc_types) if doc_type == sent_type]
        result_ann.extend(x)
        log_progress(log, idx, len(annotated_documents))
    return result_ann


def split_fda_document(document, with_bullets=False):
    res_docs, res_types = [], []
    anns = categorise_content(document.plain_text_)
    snippet_docs = split_annotated_document(document, splitter=text_to_paragraphs)
    if len(anns) != len(snippet_docs):
        raise ValueError("Got {} annotations but {} documents while splitting an FDA document"
                         .format(len(anns), len(snippet_docs)))
    for idx, ann in enumerate(anns):
        if ann.label == NATURAL_LANGUAGE:
            docs = split_annotated_document(snippet_docs[idx], splitter=text_to_sentences)
            res_docs.extend(docs)
            res_types.extend([NATURAL_LANGUAGE] * len(docs))
        elif ann.label == TABLE:
            docs = split_annotated_document(snippet_docs[idx], splitter=table_to_sentences)
            res_docs.extend(docs)
            res_types.extend([TABLE] * len(docs))
        elif ann.label == BULLET_LIST:
            res_docs.append(snippet_docs[idx])
            if with_bullets:
                res_types.append(BULLET_LIST)
            else:
                res_types.append(NATURAL_LANGUAGE)
        else:
            raise ValueError("Unknown content type!")

    return res_docs, res_types
