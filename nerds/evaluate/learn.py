import random
from sklearn.base import clone

from nerds.evaluate.score import annotation_precision_recall_f1score
from nerds.ner import NamedEntityRecognitionModel, ModelEnsembleNER
from nerds.relext import RelationExtractionModel, REModelEnsemble
from nerds.ner import CRF
from nerds.util.logging import get_logger
import numpy as np

log = get_logger()


def learning_curve(estimator, X_train, X_valid, score="f1", train_sizes=None,
                   hparams=None, shuffle=False, random_state=None):
    """ Given training and validation data, this function produces learning curves
        (as lists of scores) for a given estimator.

    Args:
        estimator: a model to be inspected
        X_train (list(AnnotatedDocument)): training data
        X_valid (list(AnnotatedDocument)): validation data
        score: the type of scores to be produced, one of {'precision', 'recall', 'f1'}
        train_sizes (list(float)): relative sizes of training subsets
        hparams (dict): hyper-parameters to be passed to a model for training
        shuffle (bool): if True, training data is shuffled
        random_state (int): used when shuffle=True to ensure reproducible results

    Returns:
        train_sizes (list(float)): relative sizes of training subsets
        train_scores (list(float)): model scores on training subsets of respective sizes
        valid_scores (list(float)): model scores on validation data
    """

    # check model type
    if isinstance(estimator, NamedEntityRecognitionModel):
        annotation_type = "annotation"
        if isinstance(estimator, ModelEnsembleNER):
            annotation_labels = set()
            for model in estimator.models:
                annotation_labels.update(model.entity_labels)
            annotation_labels = list(annotation_labels)
        else:
            annotation_labels = estimator.entity_labels
    elif isinstance(estimator, RelationExtractionModel):
        annotation_type = "relation"
        if isinstance(estimator, REModelEnsemble):
            annotation_labels = set()
            for model in estimator.models:
                annotation_labels.update(model.relation_labels)
            annotation_labels = list(annotation_labels)
        else:
            annotation_labels = estimator.relation_labels
    else:
        raise TypeError("Given estimator is of type '{}' which is not supported".format(type(estimator)))

    # determine annotation label
    if annotation_labels:
        if len(annotation_labels) > 1:
            log.debug("Learning curves currently support either one label or all labels: building for all labels")
            annotation_label = None
        else:
            annotation_label = annotation_labels[0]
    else:
        annotation_label = None

    # make default train sizes as fractions
    if not train_sizes:
        train_sizes = [s * 0.1 for s in range(1, 11)]

    # shuffle training data if necessary
    if shuffle:
        if random_state:
            random.Random(random_state).shuffle(X_train)
        else:
            random.shuffle(X_train)

    # collect scores for each training subset
    train_scores = []
    valid_scores = []

    for train_size in train_sizes:
        docs_to_train = X_train[:int(train_size * len(X_train))]
        if not docs_to_train:
            log.debug("No documents to train: check your train sizes")

        base_estimator = clone(estimator)

        if hparams:
            base_estimator.fit(X=docs_to_train, y=None, **hparams)
        else:
            base_estimator.fit(X=docs_to_train, y=None)

        X_train_pred = base_estimator.transform(docs_to_train)
        X_valid_pred = base_estimator.transform(X_valid)

        score_train = annotation_precision_recall_f1score(
            X_train_pred, docs_to_train, ann_label=annotation_label, ann_type=annotation_type)

        score_valid = annotation_precision_recall_f1score(
            X_valid_pred, X_valid, ann_label=annotation_label, ann_type=annotation_type)

        if score == "precision":
            train_scores.append(score_train[0])
            valid_scores.append(score_valid[0])
        elif score == "recall":
            train_scores.append(score_train[1])
            valid_scores.append(score_valid[1])
        elif score == "f1":
            train_scores.append(score_train[2])
            valid_scores.append(score_valid[2])
        else:
            raise ValueError("Cannot determine the type of scoring '{}'".format(score))

    return train_sizes, train_scores, valid_scores


def learning_curve_cv(estimator, X_folds, score="f1", train_sizes=None,
                      hparams=None, shuffle=False, random_state=None):
    """ Given data partitions of equal sizes (folds), this function produces learning curves
        (as lists of scores) for a given estimator per fold.

    Args:
        estimator: a model to be inspected
        X_folds (list(list(AnnotatedDocument))): training and validation data as folds of equal sizes
        score: the type of scores to be produced, one of {'precision', 'recall', 'f1'}
        train_sizes (list(float)): relative sizes of training subsets
        hparams (dict): hyper-parameters to be passed to a model for training
        shuffle (bool): if True, training data is shuffled
        random_state (int): used when shuffle=True to ensure reproducible results

    Returns:
        train_sizes_per_fold (list(list(float))): relative sizes of training subsets per fold
        train_scores_per_fold (list(list(float))): model scores on training subsets of respective sizes per fold
        valid_scores_per_fold (list(list(float))): model scores on validation data per fold
    """

    train_sizes_per_fold, train_scores_per_fold, valid_scores_per_fold = [], [], []
    for fold in X_folds:
        X_train = []
        for other_fold in X_folds:
            if other_fold != fold:
                X_train.extend(other_fold)
        # get scores for each fold
        result = learning_curve(estimator=estimator, X_train=X_train, X_valid=fold,
                                score=score, train_sizes=train_sizes, hparams=hparams,
                                shuffle=shuffle, random_state=random_state)
        train_sizes_per_fold.append(result[0])
        train_scores_per_fold.append(result[1])
        valid_scores_per_fold.append(result[2])

    return train_sizes_per_fold, train_scores_per_fold, valid_scores_per_fold


def learning_curves_kfold(train_folds, hparams):
    train_folds_list = [v for k, v in train_folds.items()]
    train_scores_kfold = {}
    test_scores_kfold = {}
    train_sizes_kfold = {}

    for idx, fold in enumerate(train_folds_list):
        X_test = np.asarray(fold)
        X_train = []
        for other_fold in train_folds_list:
            if other_fold == fold:
                continue
            X_train.extend(other_fold)
        # X_train = np.asarray(X_train)
        train_sizes, train_scores, valid_scores = learning_curve(estimator=CRF(entity_labels=["AdverseReaction"]),
                                                                 shuffle=True, X_train=X_train, X_valid=X_test,
                                                                 hparams=hparams)
        train_sizes_kfold[idx] = train_sizes
        train_scores_kfold[idx] = train_scores
        test_scores_kfold[idx] = valid_scores
        print(len(X_test), len(X_train), '\n')

    return train_scores_kfold, test_scores_kfold, train_sizes_kfold
