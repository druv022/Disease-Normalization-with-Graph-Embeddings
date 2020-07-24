import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from numpy.random import RandomState

from nerds.evaluate.kfold import KFoldCV
from nerds.evaluate.score import annotation_precision_recall_f1score
from nerds.optimize.params import ExactListParam, RangeParam
from nerds.util.logging import get_logger

log = get_logger()


class Optimizer(object):
    """ This class wraps around the popular hyperopt optimization library.
        It accepts a `NERModel`, a dictionary corresponding to a parameter grid
        to perform search on, and the named entity label to be optimized for,
        and optimizes for F-score using Tree-structured Parzen Estimators. It
        returns only the `NERModel` corresponding to the best performing
        configuration.

        The parameter configuration is accepted as a list of `RangeParam`s
        or `ExactListParam`s. So if a neural network has two parameters to
        optimize, say learning rate and number of neurons, we make a param_grid
        as:
        {
            "learning_rate": RangeParam(0.1, 0.5),
            "number_of_neurons": ExactListParam(range(10, 100))
        }
        This means that the param_grid has two parameters to be optimized:
        1) learning_rate: it is a float parameter, varying between 0.1 and 0.5
        on a continuous domain. Hence, `RangeParam`.
        2) number_of_neurons: it is an integer parameter, varying between 10
        and 100. Hence we explicitly make a list of numbers between 10 and 100,
        and feed it in as an `ExactListParam`.

        Attributes:
            model (NERModel | RelationExtractionModel): A model object.
            param_grid (list(RangeParam) or list(ExactListParam)):
                A list of parameters to be optimized.
            objective_score (str): The objective score to optimize for, one from ('precision', 'recall', 'f1')
            shuffle (bool, optional): Whether to shuffle the data before
                the cross validation. Defaults to True.
            annotation_labels (str): The named entity to optimize for.
            annotation_type (str): The type of annotations, one from ('annotation', 'relation', 'norm').

        Returns:
            best_config: Best performing model configuration.

        Example use:
        brat_input_train = BratInput("..")
        X = brat_input_train.transform()
        X_sentences = split_annotated_documents(X)

        hparams = {
            "c1": RangeParam(0.01, 0.5),
            "c2": RangeParam(0.01, 0.5)
        }

        model = CRF()
        opt = Optimizer(model, hparams)
        opt.optimize(X_sentences)
        print(opt.score_max)
        print(opt.best_config)
    """

    def __init__(self,
                 model,
                 param_grid,
                 objective_score="f1",
                 shuffle=True,
                 annotation_labels=None,
                 annotation_type="annotation"):
        self.model = model
        self.param_grid = param_grid
        self.objective_score = objective_score
        self.shuffle = shuffle
        self.annotation_labels = annotation_labels
        self.annotation_type = annotation_type
        self.trials = None
        self.best_config = None

        # Prepare the param_grid for hyperopt
        self._hparams = {}
        for param_name in param_grid:
            if isinstance(param_grid[param_name], ExactListParam):
                self._hparams[param_name] = hp.choice(
                    param_name, param_grid[param_name].list_of_values)
            elif isinstance(param_grid[param_name], RangeParam):
                self._hparams[param_name] = hp.uniform(
                    param_name, param_grid[param_name].low, param_grid[param_name].high)
            else:
                raise TypeError("Unknown param type '{}' is detected in param_grid.".format(param_grid[param_name]))

    def optimize(self, X, max_evals=10, n_folds=5, random_state=2019):
        """ Main method to run the optimization process on given documents. """
        def objective_fn(hparams):
            kfold = KFoldCV(model=self.model, k=n_folds, shuffle=self.shuffle, random_state=random_state,
                            annotation_type=self.annotation_type, annotation_labels=self.annotation_labels)
            scores = kfold.cross_validate(X, hparams).report.mean
            score = self._score(scores)
            log.debug("score = {}.".format(score))
            return {"loss": -score, 'status': STATUS_OK}
        return self._find_best_config(objective_fn, max_evals, random_state)

    def optimize_on_folds(self, folds, max_evals=10, random_state=2019):
        """ Main method to run the optimization process on given documents. """
        def objective_fn(hparams):
            kfold = KFoldCV(model=self.model, k=len(folds), shuffle=self.shuffle, random_state=random_state,
                            annotation_type=self.annotation_type, annotation_labels=self.annotation_labels)
            scores = kfold.cross_validate_on_folds(folds, hparams).report.mean
            score = self._score(scores)
            log.debug("score = {}.".format(score))
            return {"loss": -score, 'status': STATUS_OK}
        return self._find_best_config(objective_fn, max_evals, random_state)

    def optimize_on_train_valid(self, X_train, X_valid, max_evals=10, random_state=2019):
        """ Main method to run the optimization process on given documents. """
        def objective_fn(hparams):
            opt_model = self.model.clone()
            opt_model.fit(X=X_train, y=None, **hparams)
            X_pred = opt_model.transform(X_valid)
            scores = annotation_precision_recall_f1score(
                X_pred, X_valid, ann_label=self.annotation_labels, ann_type=self.annotation_type)
            score = self._score(scores)
            log.debug("score = {}.".format(score))
            return {"loss": -score, 'status': STATUS_OK}
        return self._find_best_config(objective_fn, max_evals, random_state)

    def _score(self, scores):
        if self.objective_score == "precision":
            score = scores[0]
        elif self.objective_score == "recall":
            score = scores[1]
        elif self.objective_score == "f1":
            score = scores[2]
        else:
            raise ValueError("Unknown objective score '{}': use one from ('precision', 'recall', 'f1')"
                             .format(self.objective_score))
        return score

    def _find_best_config(self, objective_fn, max_evals, random_state):
        log.debug("Started fine-tuning the model...")
        self.trials = Trials()
        self.best_config = fmin(fn=objective_fn,
                                space=self._hparams,
                                algo=tpe.suggest,
                                max_evals=max_evals,
                                trials=self.trials,
                                rstate=RandomState(random_state))
        log.debug("Finished fine-tuning the model.")
        log.debug("Best configuration: {}".format(self.best_config))
        log.debug("\twith score mean = {:.3f} and std = {:.3f}".format(self.score_mean, self.score_std))
        return self.best_config

    @property
    def score_mean(self):
        return - np.asarray(self.trials.losses()).mean()

    @property
    def score_std(self):
        return np.asarray(self.trials.losses()).std()
