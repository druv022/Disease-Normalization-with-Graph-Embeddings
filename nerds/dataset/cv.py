import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from nerds.util.logging import get_logger

log = get_logger()

RANDOM_STRATEGY = "random"
BINNED_STRATEGY = "strat_entity_num"
MIN_OVERLAP_STRATEGY = "min_entity_overlap"
BINNED_MIN_OVERLAP_STRATEGY = "strat_min_entity_overlap"
DOCUMENT_SCOPE_STRATEGY = "separate_document_scope"


class CVSplit(object):
    """ This represents a handler for data partitioning to support
        evaluation and cross validation efforts.

        Attributes:
            entity_type: (str) the entity_type (such as "Adverse_drug_reaction")
                         based on which the binned splits would be obtained.
            n_folds:     (int) the number of folds required in the data partitioning.
            strategy:    (string) splitting strategy, one from
                        ('random', 'strat_entity_num', 'min_entity_overlap',
                        'strat_min_entity_overlap', 'separate_document_scope')
            test_ratio: (float) The ratio of a test set to be created
            valid_ratio: (float) The ratio of a validation set to be created
    """

    def __init__(self,
                 entity_type,
                 n_folds=5,
                 strategy=RANDOM_STRATEGY,
                 test_ratio=0.2,
                 valid_ratio=0.2):

        self.entity_type = entity_type
        self.k = n_folds
        self.strategy = strategy
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.train = None
        self.test = None
        self.valid = None
        self.kfolds = None
        log.debug("The entity type being considered is {}".format(entity_type))

    def make_test_train(self, annotated_documents, random_state=2019):
        """
        This method obtains and returns the test-train split of the
        given data according to the chosen strategy.

        Returns:
            list: A list of annotated documents comprising test set
            list: A list of annotated documents comprising train set
        """
        unique_annotated_documents = list(set(annotated_documents))

        if self.strategy == RANDOM_STRATEGY:
            self.train, self.test = split_to_train_test_random(
                unique_annotated_documents,
                test_ratio=self.test_ratio,
                random_state=random_state)

        elif self.strategy == BINNED_STRATEGY:
            self.test, self.train = _create_test_train_by_strat_entity_num(
                entity_type=self.entity_type,
                annotated_documents=unique_annotated_documents,
                test_ratio=self.test_ratio,
                random_state=random_state)

        elif self.strategy == MIN_OVERLAP_STRATEGY:
            self.test, self.train = _create_test_train_by_min_entity_overlap(
                entity_type=self.entity_type,
                annotated_documents=unique_annotated_documents,
                test_ratio=self.test_ratio,
                random_state=random_state)

        elif self.strategy == BINNED_MIN_OVERLAP_STRATEGY:
            self.test, self.train = _create_test_train_by_strat_min_entity_overlap(
                entity_type=self.entity_type,
                annotated_documents=unique_annotated_documents,
                test_ratio=self.test_ratio,
                random_state=random_state)

        elif self.strategy == DOCUMENT_SCOPE_STRATEGY:
            self.test, self.train = _create_test_train_by_document_scope(
                annotated_documents=unique_annotated_documents,
                test_ratio=self.test_ratio,
                random_state=random_state)

        else:
            raise ValueError("Unknown split strategy '{}': use one from {}"
                             .format(self.strategy, (RANDOM_STRATEGY, BINNED_STRATEGY, MIN_OVERLAP_STRATEGY,
                                                     BINNED_MIN_OVERLAP_STRATEGY, DOCUMENT_SCOPE_STRATEGY)))

        return self.test, self.train

    def make_test_train_validation(self, annotated_documents, random_state=2019):
        """
        This method obtains and returns the test-train-validation split of the
        given data according to the chosen strategy.

        Returns:
            list: A list of annotated documents comprising test set
            list: A list of annotated documents comprising train set
            list: A list of annotated documents comprising validation set
        """
        unique_annotated_documents = list(set(annotated_documents))

        if self.strategy == RANDOM_STRATEGY:
            self.train, test = split_to_train_test_random(
                unique_annotated_documents,
                test_ratio=self.test_ratio + self.valid_ratio,
                random_state=random_state)

            self.valid, self.test = split_to_train_test_random(
                test,
                test_ratio=self.test_ratio / (self.valid_ratio + self.test_ratio),
                random_state=random_state)

        elif self.strategy == BINNED_STRATEGY:
            test, self.train = _create_test_train_by_strat_entity_num(
                entity_type=self.entity_type,
                annotated_documents=unique_annotated_documents,
                test_ratio=self.test_ratio + self.valid_ratio,
                random_state=random_state)

            self.test, self.valid = _create_test_train_by_strat_entity_num(
                entity_type=self.entity_type,
                annotated_documents=test,
                test_ratio=self.test_ratio / (self.valid_ratio + self.test_ratio),
                random_state=random_state)

        elif self.strategy == MIN_OVERLAP_STRATEGY:
            test, self.train = _create_test_train_by_min_entity_overlap(
                entity_type=self.entity_type,
                annotated_documents=unique_annotated_documents,
                test_ratio=self.test_ratio + self.valid_ratio,
                random_state=random_state)

            self.valid, self.test = split_to_train_test_random(
                test,
                test_ratio=self.test_ratio / (self.valid_ratio + self.test_ratio),
                random_state=random_state)

        elif self.strategy == BINNED_MIN_OVERLAP_STRATEGY:
            test, self.train = _create_test_train_by_strat_min_entity_overlap(
                entity_type=self.entity_type,
                annotated_documents=unique_annotated_documents,
                test_ratio=self.test_ratio + self.valid_ratio,
                random_state=random_state)

            self.test, self.valid = _create_test_train_by_strat_entity_num(
                entity_type=self.entity_type,
                annotated_documents=test,
                test_ratio=self.test_ratio / (self.valid_ratio + self.test_ratio),
                random_state=random_state)

        elif self.strategy == DOCUMENT_SCOPE_STRATEGY:
            test, self.train = _create_test_train_by_document_scope(
                annotated_documents=unique_annotated_documents,
                test_ratio=self.test_ratio + self.valid_ratio,
                random_state=random_state)

            self.test, self.valid = _create_test_train_by_document_scope(
                annotated_documents=test,
                test_ratio=self.test_ratio / (self.valid_ratio + self.test_ratio),
                random_state=random_state)

        else:
            raise ValueError("Unknown split strategy '{}': use one from {}"
                             .format(self.strategy, (RANDOM_STRATEGY, BINNED_STRATEGY, MIN_OVERLAP_STRATEGY,
                                                     BINNED_MIN_OVERLAP_STRATEGY, DOCUMENT_SCOPE_STRATEGY)))

        return self.test, self.train, self.valid

    def make_cv_folds(self, annotated_documents, random_state=2019):
        """
        This method obtains and returns k-fold data partition according to the chosen strategy.
        """
        unique_annotated_documents = list(set(annotated_documents))

        if self.strategy == RANDOM_STRATEGY:
            self.kfolds = split_to_cross_validation_folds(
                X=unique_annotated_documents,
                n_folds=self.k,
                random_state=random_state)

        elif self.strategy == BINNED_STRATEGY:
            self.kfolds = _create_cv_folds_by_strat_entity_num(
                entity_type=self.entity_type,
                n_folds=self.k,
                annotated_documents=unique_annotated_documents,
                random_state=random_state)

        elif self.strategy == MIN_OVERLAP_STRATEGY:
            self.kfolds = _create_cv_folds_min_entity_overlap(
                entity_type=self.entity_type,
                n_folds=self.k,
                annotated_documents=unique_annotated_documents,
                random_state=random_state)

        elif self.strategy == BINNED_MIN_OVERLAP_STRATEGY:
            self.kfolds = _create_cv_folds_strat_min_entity_overlap(
                entity_type=self.entity_type,
                n_folds=self.k,
                annotated_documents=unique_annotated_documents,
                random_state=random_state)

        elif self.strategy == DOCUMENT_SCOPE_STRATEGY:
            self.kfolds = _create_cv_folds_by_document_scope(
                n_folds=self.k,
                annotated_documents=unique_annotated_documents,
                random_state=random_state)

        else:
            raise ValueError("Unknown split strategy '{}': use one from {}"
                             .format(self.strategy, (RANDOM_STRATEGY, BINNED_STRATEGY, MIN_OVERLAP_STRATEGY,
                                                     BINNED_MIN_OVERLAP_STRATEGY, DOCUMENT_SCOPE_STRATEGY)))

        return self.kfolds

    def make_test_and_cv_folds(self, annotated_documents, random_state=2019):
        """
        This method obtains and returns the test and k-fold data partition
        according to the chosen strategy.
        """
        unique_annotated_documents = list(set(annotated_documents))

        self.test, self.train = self.make_test_train(
            annotated_documents=unique_annotated_documents,
            random_state=random_state)

        self.kfolds = self.make_cv_folds(
            annotated_documents=self.train, random_state=random_state)

        return self.test, self.kfolds


def bin_documents_by_entity_number(entity_type, annotated_documents, min_count):
    # get counts
    ent_counts = [len(doc.annotations_by_label(entity_type)) for doc in annotated_documents]
    # resolve insufficient counts
    _resolve_insufficient_counts(ent_counts, min_count)

    # make bins
    binned_documents = dict()
    for count, doc in zip(ent_counts, annotated_documents):
        if count in binned_documents:
            binned_documents[count].append(doc)
        else:
            binned_documents[count] = [doc]
    return binned_documents


def bin_documents_by_identifier(annotated_documents):
    # bin documents by identifier
    idx_docs_map = dict()
    for doc in annotated_documents:
        idx = doc.identifier.split("_")[0]
        if idx in idx_docs_map:
            idx_docs_map[idx].append(doc)
        else:
            idx_docs_map[idx] = [doc]
    return idx_docs_map


def split_to_cross_validation_folds(X, n_folds=5, random_state=None, strata=None):
    if strata:
        kfold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    else:
        kfold = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    folds = []
    X_np = np.asarray(X)
    for train_index, test_index in kfold.split(X=X, y=strata):
        folds.append(list(X_np[test_index]))
    return folds


def split_to_train_test_random(*arrays, test_ratio=0.2, random_state=None, strata=None):
    return train_test_split(*arrays, test_size=test_ratio, random_state=random_state, shuffle=True, stratify=strata)


def split_to_train_validation_test_random(X, test_ratio=0.2, validation_ratio=0.2,
                                          random_state=None, strata=None):
    if strata:
        train, test_valid, train_s, test_valid_s = split_to_train_test_random(
            X, strata, test_ratio=test_ratio + validation_ratio,
            random_state=random_state, strata=strata)
        valid, test = split_to_train_test_random(
            test_valid, test_ratio=test_ratio / (test_ratio + validation_ratio),
            random_state=random_state, strata=test_valid_s)
    else:
        train, test_valid = split_to_train_test_random(
            X, test_ratio=test_ratio + validation_ratio,
            random_state=random_state)
        valid, test = split_to_train_test_random(
            test_valid, test_ratio=test_ratio / (test_ratio + validation_ratio),
            random_state=random_state)
    return train, valid, test


def _create_test_train_by_strat_entity_num(entity_type, annotated_documents, test_ratio, random_state):
    # get counts
    ent_counts = [len(doc.annotations_by_label(entity_type)) for doc in annotated_documents]
    # resolve insufficient counts
    _resolve_insufficient_counts(ent_counts, 1.0 / test_ratio)
    # split
    train, test = split_to_train_test_random(
        annotated_documents, test_ratio=test_ratio, random_state=random_state, strata=ent_counts)
    return test, train


def _create_cv_folds_by_strat_entity_num(entity_type, annotated_documents, n_folds, random_state):
    # get counts
    ent_counts = [len(doc.annotations_by_label(entity_type)) for doc in annotated_documents]
    # resolve insufficient counts
    _resolve_insufficient_counts(ent_counts, n_folds)
    # split
    folds = split_to_cross_validation_folds(
        X=annotated_documents, n_folds=n_folds, random_state=random_state, strata=ent_counts)
    return folds


def baseline_score(test, train, entity_type=None):
    train_entities = set()
    for doc in train:
        for ann in doc.annotations:
            if not entity_type or ann.label == entity_type:
                train_entities.add(ann.text)
    count, total = 1.0, 1.0
    for doc in test:
        for ann in doc.annotations:
            if not entity_type or ann.label == entity_type:
                total += 1.0
                if ann.text in train_entities:
                    count += 1.0
    return count / total


def _resolve_insufficient_counts(counts, min_count):
    # make histogram
    hist = dict()
    for c in counts:
        if c in hist:
            hist[c] += 1
        else:
            hist[c] = 1

    # find small counts
    tiny_counts = {c for c in hist if hist[c] < min_count}

    # find nearest normal counts
    min_val = min(hist.keys())
    max_val = max(hist.keys())
    tiny_normal_map = dict()
    for c in tiny_counts:
        for j in range(1, max(c - min_val, max_val - c)):
            if c + j not in tiny_counts and c + j in hist:
                tiny_normal_map[c] = c + j
                break
            if c - j not in tiny_counts and c - j in hist:
                tiny_normal_map[c] = c - j
                break

    # reassign values to satisfy min_count
    for i in range(len(counts)):
        c = counts[i]
        if c in tiny_counts:
            counts[i] = tiny_normal_map[c]


def _create_test_train_by_min_entity_overlap(entity_type, annotated_documents,
                                             test_ratio, random_state):
    # bin documents by number of entities
    binned_docs = bin_documents_by_entity_number(
        entity_type=entity_type, annotated_documents=annotated_documents, min_count=1.0 / test_ratio)

    # split documents with no entities
    train, test = split_to_train_test_random(
        binned_docs[0], test_ratio=test_ratio, random_state=random_state)

    # split documents with entities
    docs_with_entities = []
    for b in binned_docs:
        if b == 0:
            continue
        docs_with_entities.extend(binned_docs[b])
    train_b, test_b = _create_train_test_min_entity_overlap_ann(
        docs_with_entities, test_ratio=test_ratio, entity_type=entity_type)
    train.extend(train_b)
    test.extend(test_b)

    return test, train


def _create_test_train_by_strat_min_entity_overlap(entity_type, annotated_documents,
                                                   test_ratio, random_state):
    # bin documents by number of entities
    binned_docs = bin_documents_by_entity_number(
        entity_type=entity_type, annotated_documents=annotated_documents, min_count=1.0 / test_ratio)

    # split documents with no entities
    train, test = split_to_train_test_random(
        binned_docs[0], test_ratio=test_ratio, random_state=random_state)

    # split documents with entities
    for b, docs in binned_docs.items():
        if b == 0:
            continue
        train_b, test_b = _create_train_test_min_entity_overlap_ann(
            X=docs, test_ratio=test_ratio, entity_type=entity_type)
        train.extend(train_b)
        test.extend(test_b)

    return test, train


def _score_document_by_entity_frequencies(index, entities_per_doc):
    # if empty
    if not entities_per_doc[index]:
        count = 0.0
        for i in range(len(entities_per_doc)):
            if not entities_per_doc[i]:
                count += 1.0
        return count / len(entities_per_doc)
    # count occurrences per entity
    score = 0.0
    for ent in entities_per_doc[index]:
        count = 0.0
        for i in range(len(entities_per_doc)):
            if ent in entities_per_doc[i]:
                count += 1.0
        score += count
    return score / len(entities_per_doc)


def _create_train_test_min_entity_overlap_ann(X, test_ratio, entity_type):
    if test_ratio >= 1.0:
        return [], X
    entities_per_doc = [doc.annotation_values_by_label(ann_label=entity_type) for doc in X]
    scored_docs = [(X[i], _score_document_by_entity_frequencies(i, entities_per_doc)) for i in range(len(X))]
    scored_docs.sort(key=lambda x: x[1], reverse=False)
    test, train = [], []
    test_size = int(test_ratio * len(X))
    for doc, _ in scored_docs:
        if len(test) < test_size:
            test.append(doc)
        else:
            train.append(doc)
    return train, test


def _create_cv_folds_min_entity_overlap_ann(X, n_folds, entity_type):
    if n_folds <= 1:
        return [X]
    folds = []
    train_i = X
    for i in range(n_folds):
        test_ratio = 1.0 / (n_folds - i)
        train_i, test_i = _create_train_test_min_entity_overlap_ann(
            X=train_i, test_ratio=test_ratio, entity_type=entity_type)
        folds.append(test_i)
    return folds


def _create_cv_folds_min_entity_overlap(entity_type, n_folds, annotated_documents, random_state):
    # bin documents by number of entities
    binned_docs = bin_documents_by_entity_number(
        entity_type=entity_type, annotated_documents=annotated_documents, min_count=n_folds)

    # split documents with no entities
    folds = split_to_cross_validation_folds(X=binned_docs[0], n_folds=n_folds, random_state=random_state)

    # split documents with entities
    docs_with_entities = []
    for b in binned_docs:
        if b == 0:
            continue
        docs_with_entities.extend(binned_docs[b])
    folds_ent = _create_cv_folds_min_entity_overlap_ann(
        X=docs_with_entities, n_folds=n_folds, entity_type=entity_type)
    for i in range(n_folds):
        folds[i].extend(folds_ent[i])

    return folds


def _create_cv_folds_strat_min_entity_overlap(entity_type, n_folds, annotated_documents, random_state):
    # bin documents by number of entities
    binned_docs = bin_documents_by_entity_number(
        entity_type=entity_type, annotated_documents=annotated_documents, min_count=n_folds)

    # split documents with no entities
    folds = split_to_cross_validation_folds(X=binned_docs[0], n_folds=n_folds, random_state=random_state)

    # split documents with entities
    for b, docs in binned_docs.items():
        if b == 0:
            continue
        folds_b = _create_cv_folds_min_entity_overlap_ann(
            X=docs, n_folds=n_folds, entity_type=entity_type)
        for i in range(n_folds):
            folds[i].extend(folds_b[i])

    return folds


def _create_test_train_by_document_scope(annotated_documents, test_ratio, random_state):
    # bin documents by identifier
    idx_docs_map = bin_documents_by_identifier(annotated_documents)
    # get counts
    sent_counts = [len(idx_docs_map[idx]) for idx in idx_docs_map]
    # resolve insufficient counts
    _resolve_insufficient_counts(sent_counts, 1.0 / test_ratio)

    # randomly select document identifiers based on stratification
    test, train = [], []
    ids_train, ids_test = split_to_train_test_random(
        list(idx_docs_map.keys()), test_ratio=test_ratio, random_state=random_state, strata=sent_counts)
    for i in ids_test:
        test.extend(idx_docs_map[i])
    for i in ids_train:
        train.extend(idx_docs_map[i])

    return test, train


def _create_cv_folds_by_document_scope(n_folds, annotated_documents, random_state):
    # bin documents by identifier
    idx_docs_map = bin_documents_by_identifier(annotated_documents)
    # get counts
    sent_counts = [len(idx_docs_map[idx]) for idx in idx_docs_map]
    # resolve insufficient counts
    _resolve_insufficient_counts(sent_counts, n_folds)

    # make folds based on stratification
    folds = []
    ids_folds = split_to_cross_validation_folds(
        X=list(idx_docs_map.keys()), n_folds=n_folds, random_state=random_state, strata=sent_counts)
    for ids_fold in ids_folds:
        fold = []
        for i in ids_fold:
            fold.extend(idx_docs_map[i])
        folds.append(fold)

    return folds
