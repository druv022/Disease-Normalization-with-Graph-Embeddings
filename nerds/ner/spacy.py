from collections import defaultdict
from pathlib import Path
from random import shuffle

import spacy
from spacy.util import minibatch

from nerds.config.base import SpacyModelConfiguration
from nerds.doc.annotation import Annotation
from nerds.doc.document import AnnotatedDocument
from nerds.ner.base import NamedEntityRecognitionModel
from nerds.util.file import mkdir
from nerds.util.logging import get_logger, log_progress

log = get_logger()

KEY = "spacy_ner"


class SpaCyStatisticalNER(NamedEntityRecognitionModel):
    def __init__(self, entity_labels=None):
        super().__init__(entity_labels)
        self.key = KEY
        self.config = SpacyModelConfiguration()
        if self.entity_labels:
            self.config.set_parameter("entity_labels", self.entity_labels)

        self.nlp = spacy.blank("en")
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.create_pipe("ner")
            self.nlp.add_pipe(self.ner)
        else:
            self.ner = self.nlp.get_pipe("ner")

    def _transform_to_spacy_format(self, X):
        """ Transforms an annotated set of documents to the format that
            spaCy needs to operate. It's a 2-tuple of text - dictionary, where
            the dictionary has "entities" as key, and a list of tuples as
            value.

            Example:
                (
                    "The quick brown fox jumps over the lazy dog",
                    {
                        "entities": [
                            (16, 19, "ANIMAL"),
                            (40, 43, "ANIMAL")
                        ]
                    }
                )
        """
        training_data = []
        # add all documents regardless of annotations
        for annotated_document in X:
            # if len(annotated_document.annotations) == 0:
            # 	continue
            training_record = (annotated_document.plain_text_, {"entities": []})
            # spaCy ends the offset 1 character later than we do - we consider
            # the exact index of the final character, while spaCy considers the
            # index of the cursor after the end of the token.
            for annotation in annotated_document.annotations:
                # In this case we're only looking for one label.
                if self.entity_labels and annotation.label not in self.entity_labels:
                    continue
                # otherwise, add all annotations.
                training_record[1]["entities"].append((annotation.offset[0], annotation.offset[1] + 1,
                                                       annotation.label))
            training_data.append(training_record)
        return training_data

    def fit(self, X, y=None, num_epochs=10, dropout=0.1, batch_size=3):
        """ Trains the NER model. The input is a list of
            `AnnotatedDocument` instances.

            We should be careful with batch size:
            if it is too big, Spacy crushes with an error
            For TAC2017, batch size=4 is already too big as sentences can be long.
        """

        log.info("Checking parameters...")
        self.config.set_parameters({"num_epochs": num_epochs, "dropout": dropout, "batch_size": batch_size})
        self.config.validate()

        # In this case we're only looking for one label.
        if self.entity_labels:
            for label in self.entity_labels:
                self.ner.add_label(label)

        # Otherwise, add support for all the annotated labels in the set.
        else:
            label_set = set()
            for annotated_document in X:
                for annotation in annotated_document.annotations:
                    label_set.add(annotation.label)
            for unq_label in label_set:
                self.ner.add_label(unq_label)

        training_data = self._transform_to_spacy_format(X)

        log.info("Training SpaCy...")
        # Get names of other pipes to disable them during training.
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):  # Only train NER.
            optimizer = self.nlp.begin_training()
            for epoch in range(self.config.get_parameter("num_epochs")):
                shuffle(training_data)
                batches = list(minibatch(training_data, size=self.config.get_parameter("batch_size")))
                losses = {}
                for idx, batch in enumerate(batches):
                    self.nlp.update([text for text, annotations in batch], [annotations for text, annotations in batch],
                                    sgd=optimizer,
                                    drop=self.config.get_parameter("dropout"),
                                    losses=losses)
                    log_progress(log, idx, len(batches), "epoch={}, loss={:.2f}".format(epoch + 1, losses['ner']))
        return self

    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
            input and returns a list of `AnnotatedDocument` objects.
        """
        # SpaCy currently doesn't support confidence scores for predictions.
        # We adopt the workaround from here: https://github.com/explosion/spaCy/issues/881

        # Number of alternate analyses to consider.
        # More is slower, and not necessarily better -- you need to experiment on your problem.
        beam_width = 16
        # This clips solutions at each step.
        # We multiply the score of the top-ranked action by this value, and use the result as a threshold.
        # This prevents the parser from exploring options that look very unlikely, saving a bit of efficiency.
        # Accuracy may also improve, because we've trained on greedy objective.
        beam_density = 0.0001

        with self.nlp.disable_pipes('ner'):
            docs = list(self.nlp.pipe([doc.plain_text_ for doc in X]))
        beams, _ = self.nlp.entity.beam_parse(docs, beam_width=beam_width, beam_density=beam_density)

        scores = []
        for doc, beam in zip(docs, beams):
            entity_scores = defaultdict(float)
            for score, ents in self.nlp.entity.moves.get_beam_parses(beam):
                for start, end, label in ents:
                    entity_scores[(start, end, label)] += score
            scores += [entity_scores]

        def score(doc_idx, ent_start, ent_end, ent_label):
            ent = (ent_start, ent_end, ent_label)
            return scores[doc_idx][ent] if ent in scores[doc_idx] else 1.0

        log.info("Annotating named entities in {} documents with Spacy...".format(len(X)))
        annotated_documents = []
        for idx, document in enumerate(X):
            annotated_document = self.nlp(document.plain_text_)
            annotations = []
            ent_id = 0
            for named_entity in annotated_document.ents:
                ent_id += 1
                annotation = Annotation(
                    text=named_entity.text,
                    label=named_entity.label_,
                    offset=(named_entity.start_char, named_entity.end_char - 1),
                    identifier="T{}".format(ent_id),
                    score=score(idx, named_entity.start, named_entity.end, named_entity.label_))
                # if there are annotations available, try to adjust the ID
                if type(document) == AnnotatedDocument:
                    for ann in document.annotations:
                        if ann == annotation:
                            annotation.identifier = ann.identifier
                            break
                annotations.append(annotation)
            annotated_documents.append(
                AnnotatedDocument(
                    content=document.content,
                    annotations=annotations,
                    relations=[],
                    normalizations=[],
                    encoding=document.encoding,
                    identifier=document.identifier,
                    uuid=document.uuid))
            # info
            log_progress(log, idx, len(X))
        return annotated_documents

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        save_path = Path(file_path)
        mkdir(save_path)
        model_save_path = save_path.joinpath("SpaCy.model")
        config_save_path = save_path.joinpath("SpaCy.config")
        self.nlp.to_disk(model_save_path)
        self.config.save(config_save_path)

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = Path(file_path)
        model_load_path = load_path.joinpath("SpaCy.model")
        config_load_path = load_path.joinpath("SpaCy.config")
        self.nlp = spacy.load(model_load_path)
        self.config.load(config_load_path)
        return self
