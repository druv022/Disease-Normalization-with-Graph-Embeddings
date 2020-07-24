import requests

from nerds.doc.annotation import Normalization
from nerds.norm.base import NormalizationModel
from nerds.util.logging import get_logger, log_progress

log = get_logger()

METAMAP_URL = 'http://ec2-52-32-67-68.us-west-2.compute.amazonaws.com/metamap?'
KEY = "metamap"


class MetaMapNormalizer(NormalizationModel):
    def __init__(self, metamap_url=METAMAP_URL, entity_labels=None):
        super().__init__(entity_labels)
        self.key = KEY

        if metamap_url is not None:
            self.metamap_url = metamap_url
        else:
            # Must get a dictionary as an input!
            raise Exception("No MetaMap URL provided!")

    def transform(self, X, y=None):
        """ Annotates the list of `AnnotatedDocument` objects with existing
            entity annotations that is provided as input and returns a list
            of `AnnotatedDocument` objects with entity and normalization
            annotations.

            This method is based on the MetaMap tool developed by the National
            Library of Medicine [1]. For the moment, this implementation
            relies on a web API to a remote installation of MetaMap.
            [1] https://metamap.nlm.nih.gov/
        """
        log.info("Normalising named entities in {} documents with MetaMap...".format(len(X)))
        for idx, document in enumerate(X):
            if self.entity_labels:
                to_normalize = [a for a in document.annotations if a.label in self.entity_labels]
            else:
                to_normalize = document.annotations

            for entity_mention in to_normalize:
                payload = {'text': entity_mention.text}
                metamap_response = requests.get(self.metamap_url, params=payload, timeout=30)
                metamap_response = metamap_response.json().get('response')
                normalizations = metamap_response[0].get('normalizations')

                for normalization in normalizations:
                    for meddra_pt_id in normalization.get('meddra'):
                        document.normalizations.append(
                            Normalization(
                                argument_id=entity_mention.identifier,
                                resource_id='meddra',
                                external_id=meddra_pt_id,
                                preferred_term=normalization.get('preferred_term')))
            log_progress(log, idx, len(X))

        return X

    def save(self, file_path):
        """ We currently access MetaMap via a web API,
            so saving/loading to a file is unnecessary
        """

    def load(self, file_path):
        """ We currently access MetaMap via a web API,
            so saving/loading to a file is unnecessary
        """
        return self
