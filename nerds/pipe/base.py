from pathlib import Path

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from nerds.util.file import mkdir


class AnnotationPipeline(BaseEstimator, ClassifierMixin):
    """ Provides a basic interface to train and test models for document annotations.
        This includes Named Entity Recognition (NER) and Relation Extraction (RE).
        """

    def __init__(self, steps):
        """	Initializes the model.
                """
        self.pipeline = Pipeline(steps)
        self.key = "_".join([key for key, model in steps])

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dictionary of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """
        self.pipeline.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        """ Annotates the list of `Document` objects that are provided as
                        input and returns a list of `AnnotatedDocument` objects.
                """
        return self.pipeline.transform(X)

    def save(self, file_path):
        """ Saves models to the local disk, provided a file path.
                """
        save_path = Path(file_path)
        mkdir(save_path)
        for _, model in self.pipeline.steps:
            model.save(save_path.joinpath(model.name))

    def load(self, file_path):
        """ Loads models saved locally.
        """
        load_path = Path(file_path)
        for _, model in self.pipeline.steps:
            model.load(load_path.joinpath(model.name))
        return self
