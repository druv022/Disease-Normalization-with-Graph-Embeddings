from nerds.doc.annotation import Annotation
from nerds.doc.document import AnnotatedDocument
from nerds.util.logging import get_logger, log_progress
from nerds.util.nlp import text_to_tokens

log = get_logger()


def transform_annotated_documents_to_bio_format(annotated_documents, tokenizer=text_to_tokens, entity_labels=None):
    """ Wrapper function that applies `transform_annotated_document_to_bio_format`
                for a batch of annotated documents.

                Args:
                        annotated_documents (list(AnnotatedDocument)): The annotated
                                document objects to be converted to BIO format.
                        tokenizer (function, optional): A function that accepts string
                                as input and returns a list of strings - used in tokenization.
                                Defaults to `sentence_to_tokens`.
                        entity_labels (str): if given, all other labels are ignored

                Returns:
                        2-tuple: Both the first and the second elements of this tuple
                                contain a list of lists of string, the first representing the
                                tokens in each document and the second the BIO tags.
        """
    X = []
    y = []
    for idx, annotated_document in enumerate(annotated_documents):
        tokens, bio_tags = transform_annotated_document_to_bio_format(annotated_document, tokenizer, entity_labels)
        X.append(tokens)
        y.append(bio_tags)
        # info
        # log_progress(log, idx, len(annotated_documents))
    return X, y


def transform_annotated_document_to_bio_format(annotated_document, tokenizer=text_to_tokens, entity_labels=None):
    """ Transforms an annotated set of documents to the format that
                the model requires as input for training. That is two vectors of
                strings per document containing tokens and tags (i.e. BIO) in
                consecutive order.

                Args:
                        annotated_document (AnnotatedDocument): The annotated document
                                object to be converted to BIO format.
                        tokenizer (function, optional): A function that accepts string
                                as input and returns a list of strings - used in tokenization.
                                Defaults to `sentence_to_tokens`.
                        entity_labels (str): if given, all other labels are ignored

                Returns:
                        2-tuple: (list(str), list(str)): The tokenized document
                                and the BIO tags corresponding to each of the tokens.

                Example:
                        ['Barack', 'Obama', 'lives', 'in', 'the', 'White', 'House']
                        ['B_Person', 'I_Person', '0', '0', 'B_Institution',
                        'I_Institution','I_Institution']

        """
    content = annotated_document.plain_text_
    # If they're not annotated documents or no annotations are available.
    if not (isinstance(annotated_document, AnnotatedDocument) and annotated_document.annotations):
        tokens = tokenizer(content)
        labels = ["O" for _ in tokens]
        return tokens, labels
    # get tokens and their BIO tags
    return _annotations_to_bio_tags(annotated_document.annotations, content, tokenizer, entity_labels)


def _annotations_to_bio_tags(annotations, content, tokenizer, entity_labels):
    annotations = list(annotations)
    annotations.sort()
    # check if annotations overlap
    # entity recognition generally assumes that entities don't overlap
    # because this makes model training harder and can cause confusions when predicting
    if _overlap(annotations):
        log.debug("Document contains overlapping entities: consider model training per entity type")

    tokens, labels = [], []
    substring_index = 0
    overlaps_to_process, overlaps_processed = set(), set()
    for idx, annotation in enumerate(annotations):
        # all other labels are ignored
        if entity_labels and annotation.label not in entity_labels:
            continue

        # check if entities overlap
        # save overlapping entities for later processing
        if idx > 0:
            previous = annotations[idx - 1]
            if previous.offset[0] <= annotation.offset[0] <= previous.offset[1]:
                if previous not in overlaps_to_process:
                    overlaps_to_process.add(annotation)
                    overlaps_processed.add(previous)
                    continue
                else:
                    overlaps_processed.add(annotation)

        # Tokens from the end of the previous annotation or the start of the
        # sentence, to the beginning of this one.
        non_tagged_tokens = tokenizer(content[substring_index:annotation.offset[0]])

        # Tokens corresponding to the annotation itself.
        tagged_tokens = tokenizer(content[annotation.offset[0]:annotation.offset[1] + 1])

        # Adjust the index to reflect the next starting point.
        substring_index = annotation.offset[1] + 1

        # Fill in the labels.
        non_tagged_labels = ["O" for _ in non_tagged_tokens]
        # B_tag for the first token then I_tag for residual tokens.
        tagged_labels = ["B_" + annotation.label] + \
                        ["I_" + annotation.label for _ in range(len(tagged_tokens) - 1)]

        tokens += non_tagged_tokens + tagged_tokens
        labels += non_tagged_labels + tagged_labels

    # Also take into account the substring from the last token to
    # the end of the sentence.
    if substring_index < len(content):
        non_tagged_tokens = tokenizer(content[substring_index:len(content)])
        non_tagged_labels = ["O" for _ in non_tagged_tokens]

        tokens += non_tagged_tokens
        labels += non_tagged_labels

    # process overlapping entities if there are any
    # we assume here at most two layers of overlapping entities
    if overlaps_to_process:
        substring_index = 0
        for idx, annotation in enumerate(annotations):
            # all other labels are ignored
            if entity_labels and annotation.label not in entity_labels:
                continue
            # skip already processed entities
            if annotation in overlaps_processed:
                continue
            # Tokens from the end of the previous annotation or the start of the
            # sentence, to the beginning of this one.
            non_tagged_tokens = tokenizer(content[substring_index:annotation.offset[0]])

            # Tokens corresponding to the annotation itself.
            tagged_tokens = tokenizer(content[annotation.offset[0]:annotation.offset[1] + 1])

            # Adjust the index to reflect the next starting point.
            substring_index = annotation.offset[1] + 1

            # Fill in the labels.
            non_tagged_labels = ["O" for _ in non_tagged_tokens]
            # B_tag for the first token then I_tag for residual tokens.
            tagged_labels = ["B_" + annotation.label] + \
                            ["I_" + annotation.label for _ in range(len(tagged_tokens) - 1)]

            tokens += non_tagged_tokens + tagged_tokens
            labels += non_tagged_labels + tagged_labels

        # Also take into account the substring from the last token to
        # the end of the sentence.
        if substring_index < len(content):
            non_tagged_tokens = tokenizer(content[substring_index:len(content)])
            non_tagged_labels = ["O" for _ in non_tagged_tokens]

            tokens += non_tagged_tokens
            labels += non_tagged_labels

    # check tokens and labels
    if len(tokens) != len(labels):
        raise ValueError("Tokens do not match their BIO tags: got {} tokens but {} BIO tags".format(
            len(tokens), len(labels)))
    return tokens, labels


def _overlap(annotations):
    for idx, current in enumerate(annotations):
        if idx == 0:
            continue
        previous = annotations[idx - 1]
        if previous.offset[0] <= current.offset[0] <= previous.offset[1]:
            return True
    return False


def transform_bio_tags_to_annotated_documents(tokens, bio_tags, documents):
    """ Wrapper function that applies `transform_bio_tags_to_annotated_document`
                for a batch of BIO tag - token lists. It's the inverse transformation
                of `transform_annotated_documents_to_bio_format`.

                Args:
                        tokens (list(list(str))): The tokens for each document in
                                the input.
                        bio_tags (list(rel2bow)): The BIO tags for each list
                                of tokens. For each token, there is a dictionary {tag: score}.
                        documents (list(Document)): The original input documents.

                Returns:
                        list(AnnotatedDocument)
        """
    annotated_documents = []
    for idx, document in enumerate(documents):
        annotated_documents.append(transform_bio_tags_to_annotated_document(tokens[idx], bio_tags[idx], document))
        # info
        # log_progress(log, idx, len(documents))
    return annotated_documents


def transform_bio_tags_to_annotated_document(tokens, bio_tags, document):
    """ Given a list of tokens, a list of BIO tags, and a document object,
                this function returns annotated documents formed from this information.
                For each token, there is a dictionary {tag: score}.

                Example:
                        doc -> "Barack Obama lives in the White House"
                        tokens ->
                        [['Barack', 'Obama', 'lives', 'in', 'the', 'White', 'House']]
                        bio ->
                        [['B_Person', 'I_Person', '0', '0', 'B_Institution',
                        'I_Institution','I_Institution']]

                It returns:
                AnnotatedDocument(
                content = "Barack Obama lives in the White House"
                annotations = (
                        (Barack Obama, Person, (0, 11))
                        (White House, Person, (26, 36))
                        )
                )
        """
    content = document.plain_text_

    cur_token_idx = 0
    cur_substring_idx = 0

    annotations = []
    ent_id = 0
    while cur_token_idx < len(bio_tags):
        cur_token = tokens[cur_token_idx]
        # special case of "
        if cur_token == '``' or cur_token == "''":
            cur_token = '"'
            tokens[cur_token_idx] = cur_token
        cur_tag = best(bio_tags[cur_token_idx])

        # if tokens[0] == 'Study' and cur_token == 'generalized':
        #     print('Here')

        if not cur_tag.startswith("B"):
            cur_substring_idx += len(cur_token)
            # add 1 if next character is space
            while cur_substring_idx < len(content) and content[cur_substring_idx] == ' ':
                cur_substring_idx += 1
            cur_token_idx += 1
            if cur_token_idx < len(bio_tags) and tokens[cur_token_idx] not in ['``', "''"]:
                assert(cur_substring_idx < len(content)), 'Substring index overflow'
                assert(content[cur_substring_idx:cur_substring_idx+len(tokens[cur_token_idx])]== tokens[cur_token_idx]), 'Substring index mismatch!'
            continue

        # this should work for labels with multiple "_" inside, e.g. "B_Adverse_drug_reaction"
        # BIO format always starts with "B_" or "I_", otherwise it is "O"
        cur_label = cur_tag[2:]
        cur_score = bio_tags[cur_token_idx][cur_tag] if type(bio_tags[cur_token_idx]) == dict else 1.0
        cur_token_count = 1

        # Get the absolute start of the entity, given the index
        # which stores information about the previously detected
        # entity offset.
        start_idx = content.find(cur_token, cur_substring_idx)
        end_idx = start_idx + len(cur_token)

        if cur_token_idx + 1 < len(bio_tags):
            next_tag = best(bio_tags[cur_token_idx + 1])
            # If last word skip the following
            if next_tag.startswith("I"):
                while next_tag.startswith("I"):
                    # update substring index
                    cur_substring_idx += len(cur_token)
                    # add 1 if next character is space
                    while cur_substring_idx < len(content) and content[cur_substring_idx] == ' ':
                        cur_substring_idx += 1
                    cur_token_idx += 1
                    cur_token = tokens[cur_token_idx]
                    # special case of "
                    if cur_token == '``' or cur_token == "''":
                        cur_token = '"'
                        tokens[cur_token_idx] = cur_token
                    cur_score += bio_tags[cur_token_idx][next_tag] if type(bio_tags[cur_token_idx]) == dict else 1.0
                    cur_token_count += 1
                    try:
                        next_tag = best(bio_tags[cur_token_idx + 1])
                    except IndexError:
                        break

                tmp_idx = content.find(cur_token, cur_substring_idx)
                # This line overwrites end_idx, in case there is a
                # multi-term annotation.
                end_idx = tmp_idx + len(cur_token)

        # Ends at the last character, and not after!
        idx_tuple = (start_idx, end_idx - 1)
        cur_substring_idx = end_idx
        # add 1 if next character is space
        while cur_substring_idx < len(content) and content[cur_substring_idx] == ' ':
            cur_substring_idx += 1

        if cur_token_idx+1 < len(bio_tags) and tokens[cur_token_idx+1]  not in ['``', "''"]:
            assert(cur_substring_idx < len(content)), 'Substring index overflow'
            assert(content[cur_substring_idx:cur_substring_idx+len(tokens[cur_token_idx+1])]== tokens[cur_token_idx+1]), 'Substring index mismatch!'

        if start_idx >= 0 and end_idx <= len(content) and content[start_idx:end_idx].strip():
            ent_id += 1
            annotation = Annotation(
                text=content[start_idx:end_idx],
                label=cur_label,
                offset=idx_tuple,
                identifier="T{}".format(ent_id),
                score=cur_score / cur_token_count)

            # if there are annotations available, try to adjust the ID
            if type(document) == AnnotatedDocument:
                for ann in document.annotations:
                    if ann == annotation:
                        annotation.identifier = ann.identifier
                        break

            annotations.append(annotation)

        cur_token_idx += 1

    return AnnotatedDocument(
        content=document.content,
        annotations=annotations,
        relations=[],
        normalizations=[],
        encoding=document.encoding,
        identifier=document.identifier,
        uuid=document.uuid)


def best(tag_scores):
    if type(tag_scores) != dict:
        return tag_scores
    best_score = max(tag_scores.values())
    for tag in tag_scores:
        if tag_scores[tag] == best_score:
            return tag
