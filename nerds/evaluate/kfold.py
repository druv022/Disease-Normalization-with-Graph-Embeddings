import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import KFold

from nerds.dataset.stats import get_distinct_labels
from nerds.evaluate.score import annotation_precision_recall_f1score

ALL_LABELS = "ALL_LABELS"


class CVReport(object):
    def __init__(self, annotation_type):
        self.annotation_type = annotation_type
        self.results_lookup = dict()
        self.folds_lookup = dict()

    def set_fold(self, fold_index, fold):
        self.folds_lookup[fold_index] = fold

    def get_fold(self, fold_index):
        return self.folds_lookup[fold_index]

    def set_result(self, fold_index, result):
        self.results_lookup[fold_index] = result

    def get_result(self, fold_index):
        return self.results_lookup[fold_index]

    def mean_by_label(self, annotation_label):
        results = [self.results_lookup[idx][annotation_label] for idx in self.results_lookup]
        res_matrix = np.asmatrix(results)
        return tuple(res_matrix.mean(axis=0).tolist()[0])

    def std_by_label(self, annotation_label):
        results = [self.results_lookup[idx][annotation_label] for idx in self.results_lookup]
        res_matrix = np.asmatrix(results)
        return tuple(res_matrix.mean(axis=0).tolist()[0])

    @property
    def mean(self):
        results = [self.results_lookup[idx][ALL_LABELS] for idx in self.results_lookup]
        res_matrix = np.asmatrix(results)
        return tuple(res_matrix.mean(axis=0).tolist()[0])

    @property
    def std(self):
        results = [self.results_lookup[idx][ALL_LABELS] for idx in self.results_lookup]
        res_matrix = np.asmatrix(results)
        return tuple(res_matrix.std(axis=0).tolist()[0])


class KFoldCV(object):
    """ Wrapper class that offers k-fold cross validation functionality directly
        on `AnnotatedDocument` objects.

        It accepts an `NERModel` object as input along with the cross
        validation parameters, therefore a `KFoldCV` instance needs to be
        created for every different model (instead of passing the model in the
        `cross_validate` function directly as parameter). The reason for that
        is that we may want to hold model-specific metadata for every model
        e.g. for visualization purposes.

        Attributes:
            model (NERModel or RelExtModel): A model to be tuned.
            k (int, optional): The number of folds in the k-fold cross
                validation. If 1, then `eval_split` will be used to determine
                the split. Defaults to 5.
            shuffle (bool, optional): Whether to shuffle the data before
                the cross validation. Defaults to True.
            annotation_labels (str, optional): The entity label for which the
                precision, recall, and f1-score metrics are calculated.
                Defaults to None, which means all the available entities.
            annotation_type (str, optional): The type of annotations used to evaluate predictions.
                                The default value is "annotation" which means only named entities are counted.
                                Other values are "relation" (to evaluate relations) and
                                "norm" (to evaluate normalizations).
            random_state (int, optional): A random seed, used for data shuffling and as KFold's argument
    """

    def __init__(self,
                 model,
                 k=5,
                 shuffle=True,
                 annotation_labels=None,
                 annotation_type="annotation",
                 random_state=2019):
        self.model = model
        self.k = k
        self.annotation_labels = annotation_labels
        self.shuffle = shuffle
        self.random_state = random_state
        self.annotation_type = annotation_type
        self.report = CVReport(annotation_type=annotation_type)

    def cross_validate(self, X, hparams):
        """ Method that performs k-fold cross validation on a set of annotated
            documents.

            Args:
                X (list(AnnotatedDocument)): A list of annotated documents.
                hparams (dict): The hyperparameters of the model, to be passed
                    in the `fit` method.
        """
        if not self.annotation_labels:
            self.annotation_labels = get_distinct_labels(X, annotation_type=self.annotation_type)

        if not self.annotation_labels:
            raise ValueError("No labels are given: cannot train and validate")

        X = np.asarray(X)
        kfold = KFold(n_splits=self.k, shuffle=self.shuffle, random_state=self.random_state)
        idx = 0
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            self.report.set_fold(idx, list(X_test))
            result = self._evaluate_once(X_train, X_test, hparams)
            self.report.set_result(idx, result)
            idx += 1

        return self

    def cross_validate_on_folds(self, folds, hparams):
        """ Method that performs k-fold cross validation on given folds of annotated
            documents.

            Args:
                folds (list(list(AnnotatedDocument))): A list of folds each containing a list of documents.
                hparams (dict): The hyperparameters of the model, to be passed
                    in the `fit` method.
        """
        if len(folds) <= 1:
            raise ValueError("{} folds are given: cannot cross-validate".format(len(folds)))
        if not isinstance(folds[0], list):
            raise ValueError("Folds must contain lists of documents")

        if not self.annotation_labels:
            X = []
            for fold in folds:
                X.extend(fold)
            self.annotation_labels = get_distinct_labels(X, annotation_type=self.annotation_type)

        if not self.annotation_labels:
            raise ValueError("No labels are given: cannot train and validate")

        self.k = len(folds)

        # make train/validation sets
        for idx, fold in enumerate(folds):
            self.report.set_fold(idx, list(fold))
            X_test = np.asarray(fold)
            X_train = []
            for other_fold in folds:
                if other_fold == fold:
                    continue
                X_train.extend(other_fold)
            X_train = np.asarray(X_train)
            # shuffle if necessary
            if self.shuffle:
                rs = RandomState(self.random_state)
                rs.shuffle(X_test)
                rs.shuffle(X_train)
            # calculate precision, recall, F1-score
            result = self._evaluate_once(X_train, X_test, hparams)
            self.report.set_result(idx, result)

        return self

    def _evaluate_once(self, X_train, X_test, hparams):
        """ Helper function to evaluate the model on a set of data. """
        # clone the model
        base_estimator = self.model.clone()

        # fit an estimator
        base_estimator.fit(X=X_train, y=None, **hparams)
        X_pred = base_estimator.transform(X_test)

        # store evaluation results
        result = dict()
        result[ALL_LABELS] = annotation_precision_recall_f1score(
            X_pred, X_test, ann_label=self.annotation_labels, ann_type=self.annotation_type)
        for label in self.annotation_labels:
            result[label] = annotation_precision_recall_f1score(
                X_pred, X_test, ann_label=label, ann_type=self.annotation_type)
        return result
