from nerds.ner import NamedEntityRecognitionModel
from nerds.norm.base import NormalizationModel
from nerds.relext import RelationExtractionModel


class DocumentAnnotator(object):

    def __init__(self, model):
        self.model = model

    def annotate(self, document):
        return self.model.transform([document])[0]


class NERDocumentAnnotator(DocumentAnnotator):

    def __init__(self, model):
        super().__init__(model)
        if not isinstance(self.model, NamedEntityRecognitionModel):
            raise TypeError("Given model is not a NER model: check your input")


class REDocumentAnnotator(DocumentAnnotator):

    def __init__(self, model):
        super().__init__(model)
        if not isinstance(self.model, RelationExtractionModel):
            raise TypeError("Given model is not a RE model: check your input")


class NormalizationDocumentAnnotator(DocumentAnnotator):

    def __init__(self, model):
        super().__init__(model)
        if not isinstance(self.model, NormalizationModel):
            raise TypeError("Given model is not a Normalization model: check your input")
