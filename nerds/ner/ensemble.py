import json
from pathlib import Path

import numpy as np

from nerds.evaluate.kfold import KFoldCV
from nerds.doc.document import AnnotatedDocument
from nerds.ner.base import NamedEntityRecognitionModel
from nerds.util.file import mkdir
from nerds.util.logging import get_logger, log_progress

from nerds.ner import bilstm
from nerds.ner import crf
from nerds.ner import spacy

log = get_logger()

KEY = 'ensemble_ner'
""" The following lines define a mapping from the configuration keys for each
    model to their corresponding classes. If you add a new model, please add it here as well.
"""
NER_MODELS = {bilstm.KEY: bilstm.BidirectionalLSTM, crf.KEY: crf.CRF, spacy.KEY: spacy.SpaCyStatisticalNER}


class ModelEnsembleNER(NamedEntityRecognitionModel):
    """ Abstraction for ensembling multiple NER models and producing annotations.

        This class accepts a list of NER models as input and annotates a set of
        documents based on a voting mechanism. The `vote` method in this class
        will raise a NotImplementedError, and should be overriden by offspring.

        Attributes:
            models (list(NERModel)): The NER models that participate in the
                ensemble method.
    """

    def __init__(self, models):
        """
        Args:
            models: Input model types must be for Named Entity Recognition
        """
        super().__init__()
        self.models = models
        self.key = KEY
        for model in models:
            if not isinstance(model, NamedEntityRecognitionModel):
                raise ValueError("Cannot determine the ensemble type: the input models are not suitable")

    def fit(self, X, y=None, hparams=None):
        """ Train each NER model in the ensemble. The input is a list of
            `AnnotatedDocument` instances.

            Args:
                hparams (dict(dict)): Every model has a key associated with it
                    e.g. CRF has the key "crf_ner", BidirectionalLSTM "bilstm_ner" etc.
                    This parameter is a dictionary where the keys are the model
                    keys, and the values are a dictionary with their
                    hyperparameters and their values. Example:
                    {
                        "crf_ner": {
                            "c1": 0.1,
                            "c2": 0.1
                        },
                        "bilstm_ner"
                        ...
                    }
        """
        for model in self.models:
            log.info("Training {}...".format(type(model).__name__))
            model.fit(X, y, **hparams[model.key])
            log.info("Done")
        return self

    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
            input and returns a list of `AnnotatedDocument` objects.

            Needs an implementation of the `vote` method.
        """
        log.info("Annotating {} documents with ensemble '{}' containing models {}...".format(
            len(X),
            type(self).__name__, [type(model).__name__ for model in self.models]))

        entities_per_model = []
        for model in self.models:
            entities_per_model.append(model.extract(X, y))

        annotated_documents = []
        for idx, document in enumerate(X):
            # make entities
            entities = [entities_per_model[m][idx] for m in range(len(self.models))]
            entities = self.vote(np.array(entities)) if entities else []
            # make documents
            annotated_documents.append(
                AnnotatedDocument(
                    content=document.content,
                    annotations=entities,
                    relations=document.relations if type(document) == AnnotatedDocument else [],
                    normalizations=document.normalizations if type(document) == AnnotatedDocument else [],
                    encoding=document.encoding,
                    identifier=document.identifier,
                    uuid=document.uuid))
            # info
            log_progress(log, idx, len(X))
        return annotated_documents

    def save(self, file_path):
        """ Saves an ensemble of models to the local disk, provided a
            file path.
        """
        save_path = Path(file_path)
        mkdir(save_path)
        for model in self.models:
            model.save(save_path.joinpath(model.name))

    def load(self, file_path):
        """ Loads an ensemble of models saved locally. """
        load_path = Path(file_path)
        for model in self.models:
            model.load(load_path.joinpath(model.name))
        return self

    def vote(self, entity_matrix):
        """ If __k__ NER models have annotated a single document with entities,
            this method returns a single vector of entities as a result of an
            ensemble process. The ensemble process itself and thus the voting
            algorithm should be overriden in a subclass of this class.
        """
        raise NotImplementedError


class NERModelEnsemblePooling(ModelEnsembleNER):
    def vote(self, annotation_matrix):
        """ If __k__ NER models have annotated a single document with entities,
            this method returns a single vector with all unique entities that
            every model detected.

            Args:
                annotation_matrix (2d list): Entities that have been annotated in a
                    single document by __k__ different NER models.

            Returns:
                list: Entities that have been selected after the ensemble.

            Example:
                If entity_matrix: [[x1, x2, x3], [x1, x3, x4], [x1, x2]]
                Then the result is: [x1, x2, x3, x4].
        """
        feature_set = set()
        for entity_vector in annotation_matrix:
            feature_set.update(set(entity_vector))
        return sorted(list(feature_set))


class NERModelEnsembleMajorityVote(ModelEnsembleNER):
    def vote(self, annotation_matrix):
        """ If __k__ NER models have annotated a single document with entities,
            this method returns a single vector of annotations as a result of a
            majority vote ensemble process.

            Args:
                annotation_matrix (2d list): Annotations that have been annotated in a
                    single document by __k__ different NER models.

            Returns:
                list: Annotations that have been selected after the ensemble.

            Example:
                If entity_matrix: [[x1, x2, x3], [x1, x3, x4], [x1, x2]]
                Then the result is: [x1, x2, x3] because these annotations have
                been annotated by 2/3 of the NER models.
        """
        feature_set = set()
        for entity_vector in annotation_matrix:
            feature_set.update(set(entity_vector))
        feature_list = sorted(list(feature_set))
        feature_matrix = []
        for entity_vector in annotation_matrix:
            feature_vector = []
            for feature in feature_list:
                if feature in entity_vector:
                    feature_vector.append(1)
                else:
                    feature_vector.append(0)
            feature_matrix.append(feature_vector)
        result = []
        for feature_idx, total in enumerate(np.sum(feature_matrix, axis=0)):
            if total >= len(self.models) / 2:
                feature_list[feature_idx].score = 1.0 * total / len(self.models)
                result.append(feature_list[feature_idx])
        return result


class NERModelEnsembleWeightedVote(ModelEnsembleNER):
    def __init__(self, models):
        self.confidence_scores = []
        super().__init__(models)

    def fit(self, X, y=None, cv=5, eval_split=0.8):
        """ Train each NER model in the ensemble, and keep their cross
            validation scores to later assign a confidence level during
            voting.
        """
        for model in self.models:
            kfold = KFoldCV(model, k=cv, eval_split=eval_split, annotation_type="annotation")
            hparams = {}
            # Right now it will fall back to the default hyperparameters.
            kfold.cross_validate(X, hparams)
            # F1-score as a weight
            self.confidence_scores.append(kfold.report.mean[2])
        return super().fit(X, y)

    def vote(self, annotation_matrix):
        """ If __k__ NER models have annotated a single document with entities,
            this method returns a single vector of entities as a result of a
            weighted vote ensemble process. The weights are determined by the
            performance of each individual classifier during cross validation,
            i.e. the votes of strong predictors matter more.

            Args:
                annotation_matrix (2d list): Entities that have been annotated in a
                    single document by __k__ different NER models.

            Returns:
                list: Entities that have been selected after the ensemble.

            Example:
                Let models m1, m2, m3 have a cross-validation f-score of
                f1 = 0.6, f2 = 0.9, f3 = 0.3 respectively.
                If entity_matrix: [[x1, x2, x3], [x1, x3, x4], [x1, x2]]
                Then the result is: [x1, x2, x3, x4] (unlike in the majority
                vote), because x4 was selected by a strong predictor.
        """
        feature_set = set()
        for entity_vector in annotation_matrix:
            feature_set.update(set(entity_vector))
        feature_list = sorted(list(feature_set))
        feature_matrix = []
        for entity_vector in annotation_matrix:
            feature_vector = []
            for feature in feature_list:
                if feature in entity_vector:
                    feature_vector.append(1)
                else:
                    feature_vector.append(0)
            feature_matrix.append(feature_vector)
        result = []
        for feature_idx, total in enumerate(np.apply_along_axis(self._calculate_weighted_sum, 0, feature_matrix)):
            # 1) Threshold is set at 0.5 because the factors (w_i/w) in
            # _calculate_weighted_sum always sum up to 1; so 0.5 always
            # represents half of the voting capacity.
            # 2) np.around because we often get floats like 0.499999997...
            if np.around(total, 3) >= 0.5:
                feature_list[feature_idx].score = np.around(total, 3)
                result.append(feature_list[feature_idx])
        return result

    def _calculate_weighted_sum(self, r):
        """ If we have 3 models: m1, m2, m3 with confidence scores w1, w2, w3 then:
            E = sum: (w_i/w) * r_i
            Where r_i is a vertical slice of the annotation matrix, i.e. what
            did model m_i vote for an annotated entity x_i.
        """
        w = np.sum(self.confidence_scores)
        w_i = np.true_divide(self.confidence_scores, w)
        E = np.dot(w_i, r)  # "E" stands for "Ensemble".
        return E

    def save(self, file_path):
        """ Saves an ensemble of models to the local disk, provided a
            file path. This implementation also saves the confidence scores
            as metadata.
        """
        save_path = Path(file_path)
        super().save(save_path)
        metadata_save_path = save_path.joinpath("Ensemble_metadata.json")
        if len(self.confidence_scores) > 0:
            with open(metadata_save_path, "w") as fp:
                fp.write(json.dumps({"confidence_scores": self.confidence_scores}))

    def load(self, file_path):
        """ Loads an ensemble of models saved locally. This implementation
            also loads the confidence scores, if available.
        """
        load_path = Path(file_path)
        super().load(load_path)
        metadata_load_path = load_path.joinpath("Ensemble_metadata.json")
        try:
            with open(metadata_load_path, "r") as fp:
                init_metadata = json.loads(fp.read().strip())
            self.confidence_scores = init_metadata["confidence_scores"]
        except FileNotFoundError:
            self.confidence_scores = []
        return self
