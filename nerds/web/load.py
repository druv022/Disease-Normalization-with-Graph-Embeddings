from nerds.ner.ensemble import NER_MODELS
from nerds.norm.metamap import MetaMapNormalizer
from nerds.relext.ensemble import RE_MODELS
from nerds.web.predict import NERDocumentAnnotator, REDocumentAnnotator, NormalizationDocumentAnnotator


class ModelLoader(object):

    def load_model(self, model_path):
        if not model_path:
            raise ValueError("Path to a model is empty: cannot load")
        for model_class in list(NER_MODELS.values()):
            try:
                model = model_class().load(model_path)
                if model:
                    return NERDocumentAnnotator(model)
            except Exception:
                pass
        for model_class in list(RE_MODELS.values()):
            try:
                model = model_class().load(model_path)
                if model:
                    return REDocumentAnnotator(model)
            except Exception:
                pass
        raise ValueError("Cannot determine the type of a model in '{}'".format(model_path))

    def load_normalizer(self):
        return NormalizationDocumentAnnotator(MetaMapNormalizer())
