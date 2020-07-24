import random
from collections import defaultdict


def get_distinct_labels(annotated_docs, annotation_type="annotation"):
    """
        Function to retrieve distinct entity types from a set of annotated docs

        Input:
            Annotated document list
            annotation_type: 'annotation', 'relation', 'norm'
        Output:
            Unique entity types sorted alphabetically
    """
    entity_types = set()

    for ann_doc in annotated_docs:
        for ann in ann_doc.annotations_by_type(annotation_type):
            entity_types.add(ann.label)
    entity_types = list(entity_types)
    entity_types.sort()
    return entity_types


def get_distinct_entities(annotated_docs, entity_type=None):
    """
        Function to retrieve distinct entity values from a set of annotated docs

        Input:
            Annotated document list
        Output:
            Unique entity values sorted alphabetically
    """
    entity_values = set()
    for ann_doc in annotated_docs:
        for ann in ann_doc.annotations:
            if entity_type and ann.label == entity_type:
                entity_values.add(ann.text)
            else:
                entity_values.add(ann.text)
    entity_values = list(entity_values)
    entity_values.sort()
    return entity_values


def count_annotations(annotated_docs, annotation_type="annotation"):
    """
        Function to count annotations of documents

        Input:
            Annotated document list
        Output:
            Count: an integer number
    """
    total = 0
    for ann_doc in annotated_docs:
        total += len(ann_doc.annotations_by_type(annotation_type))
    return total


def count_annotated_documents(annotated_docs, annotation_type="annotation"):
    total = 0
    for ann_doc in annotated_docs:
        if ann_doc.annotations_by_type(annotation_type):
            total += 1
    return total


def count_words_per_sentence(annotated_docs):
    """
        Function that counts words per sentence and populates the appropriate sentence bin.
        Example : "The fat cat" sentence will add +1 to the [0-3] range bin

        Args:
            A list of annotated_docs
        Returns:
            Dictionary of {range(0,30,3) : counts}
    """

    word_count_dict = dict([(range(i, i + 3), 0) for i in range(1, 37, 3) if i < 36])

    for ann_doc in annotated_docs:
        words = [word.strip() for word in ann_doc.plain_text_.split(' ')]
        word_count = len(words)

        for key in word_count_dict:
            if word_count in key:
                word_count_dict[key] += 1

    #         print("For sentence : \n {} \n Count : {}".format(ann_doc.plain_text_, word_count))

    return word_count_dict


def get_annotations(annotated_docs, annotation_type="annotation"):
    """
        Function that returns all annotations over a list of annotated documents ( or sentences )

        Input: List of annotated docs
        Output: list(all annotations)
    """
    annotations = []
    for ann_doc in annotated_docs:
        anns = ann_doc.annotations_by_type(annotation_type)
        if len(anns) >= 1:
            annotations.extend(anns)
    return annotations


def get_ann_sentences(annotated_docs, annotation_type="annotation"):
    """
        Function that retrieves annotated documents (sentences) that have more than 1 annotation
        and the ones without any annotations.

        Input: List of annotated docs
        Output: list(sentences_with_annotation)
    """

    sentences_with_annotation = []
    for ann_doc in annotated_docs:
        anns = ann_doc.annotations_by_type(annotation_type)
        # If the sentence has at least 1 annotation...
        if len(anns) >= 1:
            sentences_with_annotation.append(ann_doc)
    return sentences_with_annotation


def get_non_ann_sentences(annotated_docs, annotation_type="annotation"):
    """
        Function that retrieves annotated documents (sentences) that have more than 1 annotation
        and the ones without any annotations.

        Input: List of annotated docs
        Output: list(sentence_without_annotation)
    """
    sentences_without_annotation = []
    for ann_doc in annotated_docs:
        anns = ann_doc.annotations_by_type(annotation_type)
        # If the sentence has at least 1 annotation...
        if len(anns) < 1:
            sentences_without_annotation.append(ann_doc)
    return sentences_without_annotation


def get_entity_count_per_type(annotated_docs, annotation_type="annotation"):
    """
        Function that retrieves the number of annotated entities, per type
        Input: A list of annotated docs
        Output: Dictionary of {Entity_type: Counts}
    """
    # Get our distinct entity types
    ent_types = get_distinct_labels(annotated_docs, annotation_type)

    # Create rel2bow for {type: counts}, default is 0
    entity_type_count = dict.fromkeys(ent_types, 0)

    for ann_sent in annotated_docs:
        for ann in ann_sent.annotations_by_type(annotation_type):
            # Add +1 to the appropriate key if you encounter that label
            if ann.label in entity_type_count.keys():
                entity_type_count[ann.label] += 1

    return entity_type_count


def get_entity_values_per_type(annotated_docs, sort=False):
    """
           Function that retrieves the set of unique entities, per type
           Input: A list of annotated docs
           Output: Dictionary of {Entity_type: set(str)}
    """
    # Get our distinct entity types
    entity_types = get_distinct_labels(annotated_docs, annotation_type="annotation")
    entity_type_values = dict()
    for entity_type in entity_types:
        entity_type_values[entity_type] = set()
    for doc in annotated_docs:
        for ann in doc.annotations:
            entity_type_values[ann.label].add(ann.text)

    if sort:
        for entity_type in entity_types:
            entity_type_values[entity_type] = sorted(entity_type_values[entity_type], key=str.lower)
    return entity_type_values


def get_entity_value_count_per_type(annotated_docs):
    """
        Function that retrieves the number of unique entities, per type
        Input: A list of annotated docs
        Output: Dictionary of {Entity_type: Counts}
    """
    entity_type_values = get_entity_values_per_type(annotated_docs)
    return {key: len(entity_type_values[key]) for key in entity_type_values}


def get_entity_value_overlaps(annotated_docs_1, annotated_docs_2):
    entity_types = get_distinct_labels(annotated_docs_1 + annotated_docs_2)
    entity_type_values_1 = get_entity_values_per_type(annotated_docs_1)
    entity_type_values_2 = get_entity_values_per_type(annotated_docs_2)
    overlaps = dict()

    for entity_type in entity_types:
        # Added double if clause because of occasional KeyError in random test sets
        # And a try-except clause seems to inconveniently raising Error E722: bare 'except'
        if entity_type in entity_type_values_1.keys():
            values_1 = entity_type_values_1[entity_type]
        if entity_type in entity_type_values_2.keys():
            values_2 = entity_type_values_2[entity_type]
        if not values_1 or not values_2:
            overlaps[entity_type] = 0.0
        else:
            overlaps[entity_type] = 1.0 * len(values_1.intersection(values_2)) / len(values_1.union(values_2))
    return overlaps


def get_entity_value_subset_size(annotated_docs_1, annotated_docs_2):
    entity_types = get_distinct_labels(annotated_docs_1 + annotated_docs_2)
    entity_type_values_1 = get_entity_values_per_type(annotated_docs_1)
    entity_type_values_2 = get_entity_values_per_type(annotated_docs_2)
    overlaps = dict()
    for entity_type in entity_types:
        values_1 = entity_type_values_1[entity_type]
        values_2 = entity_type_values_2[entity_type]
        if not values_1 or not values_2:
            overlaps[entity_type] = 0.0
        else:
            overlaps[entity_type] = 1.0 * len(values_1.intersection(values_2)) / len(values_1)
    return overlaps


def get_sentence_per_type(annotated_docs, annotation_type="annotation"):
    """
        Function that retrieves sentences containing a specific entity type. Can also be used to count unique
        sentences containing a specific entity type.

        Input: A list of annotated docs

        Output:
            A dictionary of {entity_type: set(sentences of that type)}
    """
    # A list for all our annotated sentences
    total_ann_sentences = get_ann_sentences(annotated_docs, annotation_type)

    # Create rel2bow for {entity_type: set()}, default is empty set
    sentence_type_count = defaultdict(set)

    for ann_sent in total_ann_sentences:
        for ann in ann_sent.annotations_by_type(annotation_type):
            # Add the sentence to the appropriate key if you encounter that label
            sentence_type_count[ann.label].add(ann_sent)

    return sentence_type_count


def get_total_unique_ann_sent(annotated_docs, annotation_type="annotation"):
    stc = get_sentence_per_type(annotated_docs, annotation_type)

    total_unique_sent = 0
    for key, value in stc.items():
        print(key, len(value))
        total_unique_sent += len(value)

    return total_unique_sent


def get_sample(pool, size):
    pool = list(pool)
    if size >= len(pool):
        return pool
    return [pool[i] for i in sorted(random.sample(range(len(pool)), size))]


def get_sent_count_per_unique_ent_value(entity_type, annotated_documents):
    """
        Function to retrieve count of sentences in which an entity value occurs.

        Args:
            entity_type: Str; the entity type (class) on which filtering is to occur.
            annotated_documents: iterable; documents to be searched through.

        Returns:
            Dictionary of form {entity_value: #sentences}
    """
    sent_by_type = get_sentence_per_type(annotated_documents)
    ent_values = get_entity_values_per_type(sent_by_type[entity_type])
    num_sent = {ent_value: 0 for ent_value in ent_values[entity_type]}
    for doc in sent_by_type[entity_type]:
        for ann in doc.annotations:
            if ann.text in num_sent.keys():
                num_sent[ann.text] += 1
    return num_sent


def get_entity_values_with_same_sent_count(entity_type, annotated_documents, only_count=True):
    """
        Function which gives a cumulative count of the number of entity values per type,
        which have the same number of sentences mentioning them.

        Args:
            entity_type: Str; the entity type (class) on which filtering is to occur.
            annotated_documents: iteratable; the set of documents to search through.
            only_count: Bool; If True, returns the count of entity values with
                        same number of sentence mentions. Else, returns a list of
                        strings denoting the names of those entity values. Defaults
                        to True.
        Returns:
            Dictionary of form:
            (a) if only_count==True: {# of sentences mentioning an entity value:
                       count of entity values with same number of sentence mentions}
            (b) if only_count==False: {# of sentences mentioning an entity value:
                       list of entity values with same number of sentence mentions}
    """

    ent_values = defaultdict(list)
    num_sent = get_sent_count_per_unique_ent_value(entity_type, annotated_documents)

    for k, v in num_sent.items():
        ent_values[v].append(k)

    if only_count:
        freq_ent = {k: len(v) for k, v in sorted(ent_values.items(), key=lambda x: x[0])}
        return freq_ent
    else:
        return ent_values


def get_ent_count_per_sent(entity_type, annotated_documents):
    """
       Function to retrieve the number of sentences with a given number
       of annotations of a particular entity_type.
        Returns:
            Dictionary of the form {# of annotations in sentence of entity_type:
            count of sentences}
    """
    sent_w_type = get_sentence_per_type(annotated_documents)
    num_ent_per_sent = defaultdict(int)
    for doc in sent_w_type[entity_type]:
        ann_list = [ann for ann in doc.annotations if ann.label == entity_type]
        ann_count = len(ann_list)
        num_ent_per_sent[ann_count] += 1

    num_ent_per_sent = {k: v for k, v in sorted(
        num_ent_per_sent.items(), key=lambda x: x[0])}

    return num_ent_per_sent
