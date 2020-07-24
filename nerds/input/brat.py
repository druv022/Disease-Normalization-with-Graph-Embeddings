from pathlib import Path

from nerds.doc.annotation import Annotation
from nerds.doc.annotation import Relation
from nerds.doc.annotation import Normalization
from nerds.input.base import DataInput
from nerds.doc.document import AnnotatedDocument
from nerds.util.logging import get_logger, log_progress

log = get_logger()


class BratInput(DataInput):
    """
    Reads input data from a collection of BRAT txt/ann files

    This class provides the input to the rest of the pipeline,
    by transforming a collection of BRAT txt/ann files (in the provided path)
    into collection of documents. The `annotated` parameter differentiates
    between the already annotated input (required for training/evaluation)
    and the non-annotated input (required for entity extraction).

    Attributes:
        path_to_files (str): The path containing the input files,
            annotated or not.
        annotated (bool): If `False`, then the returned collection will
            consist of Document objects. If `True`, it will consist of
            AnnotatedDocument objects.
        encoding (str, optional): Specifies the encoding of the plain
            text. Defaults to 'utf-8'.
    """

    def __init__(self, path_to_files, annotated=True, encoding="utf-8"):
        super().__init__(path_to_files, annotated, encoding)

    def transform(self, X=None, y=None):
        """ Transforms the available documents into the appropriate objects,
            differentiating on the `annotated` parameter.
        """

        log.info("Reading documents from '{}'".format(self.path))

        # If not annotated, fall back to base class and simply read files
        if not self.annotated:
            return super().transform(X, y)
        # Else, read txt/ann
        else:
            annotated_docs = []
            txt_files = list(self.path.glob('*.txt'))
            for idx, txt_file in enumerate(txt_files):
                # Standard brat folder structure:
                # For every txt there should be an ann.
                ann_file = Path(self.path.joinpath(txt_file.name.replace(".txt", ".ann")))
                annotations, relations, normalizations = self._read_brat_ann_file(ann_file)

                with open(txt_file, 'rb') as doc_file:
                    annotated_docs.append(
                        AnnotatedDocument(
                            content=doc_file.read(),
                            annotations=annotations,
                            relations=relations,
                            normalizations=normalizations,
                            encoding=self.encoding,
                            identifier=txt_file.name.replace(".txt", "")))

                # info
                # log_progress(log, idx, len(txt_files))

            return annotated_docs

    def _read_brat_ann_file(self, path_to_ann_file):
        """ Helper function to read brat annotations for entities, relations,
            and normalizations.
        """
        annotations = []
        relations = []
        normalizations = []

        if path_to_ann_file.is_file():
            with open(path_to_ann_file, 'rb') as ann_file:
                for ann_line in ann_file:
                    ann_line = ann_line.decode(self.encoding)
                    split_ann_line = ann_line.strip().split("\t")

                    # Text-based annotations, i.e. entities
                    if ann_line.startswith("T"):

                        # Must be exactly 3 things, if they are entity
                        # related, e.g.: "T2\tGrant 475 491\tGIA G-14-0006063".
                        if len(split_ann_line) == 3:
                            entity_id = split_ann_line[0]
                            entity_str = split_ann_line[2]

                            # Looks like "Grant 475 491"
                            entity_type_offsets = split_ann_line[1].split(" ")

                            # The offsets are always after the entity type
                            offsets = entity_type_offsets[1:]
                            entity_name = entity_type_offsets[0]

                            if len(offsets) > 2:
                                # We have a discontinued offset to deal with...
                                # If the offset is discontinued then instead
                                # of a single offset it should be a list of
                                # offsets...
                                # e.g. AdverseReaction ((432, 437), (450, 460))
                                fixed_offset = _fix_discontinued_offsets(offsets)
                                annotations.append(
                                    Annotation(entity_str, entity_name, fixed_offset, entity_id, discontinued=True))
                            else:
                                # No discontinuity, should work as before..
                                start_offset = int(entity_type_offsets[1])
                                end_offset = int(entity_type_offsets[2]) - 1

                                annotations.append(
                                    Annotation(entity_str, entity_name, (start_offset, end_offset), entity_id))

                    # Relations
                    elif ann_line.startswith("R"):

                        # Must be exactly 2 things, if they are relation
                        # related, e.g.: "R2\tReceived Arg1:T1 Arg2:T2".
                        if len(split_ann_line) == 2:
                            relation_id = split_ann_line[0]

                            # Looks like "Received Arg1:T1 Arg2:T2"
                            relation_type_args = split_ann_line[1].split(" ")
                            relation_name = relation_type_args[0]
                            relation_source = relation_type_args[1].split(":")
                            relation_target = relation_type_args[2].split(":")

                            relations.append(
                                Relation(relation_name, relation_source[1], relation_target[1], relation_id,
                                         relation_source[0], relation_target[0]))

                    # Normalizations
                    elif ann_line.startswith("N"):

                        # Must be exactly 3 things, if they are normalization
                        # related, e.g.:
                        # "N1\tReference T1 Wikipedia:534366\tGrant (money)".
                        if len(split_ann_line) == 3:
                            normalization_id = split_ann_line[0]
                            normalization_str = split_ann_line[2]

                            # Looks like "Reference T1 Wikipedia:534366"
                            normalization_args = split_ann_line[1].split(" ")
                            normalization_name = normalization_args[0]
                            normalization_arg = normalization_args[1]
                            normalization_ids = normalization_args[2].split(":")

                            normalizations.append(
                                Normalization(normalization_arg, normalization_ids[0], normalization_ids[1],
                                              normalization_id, normalization_name, normalization_str))

                    # Other annotation types will be ignored
                    else:
                        continue

        return sorted(annotations), relations, normalizations


def _fix_discontinued_offsets(offsets):
    """
        Helper function to deal with discontinued offsets
        It currently concatenates all words in between,
        i.e. returns the start of the first offset and the end of the last offset
        Input (list(str)): offsets is a list of strings, e.g. ["12", "16; 25", "32"]
        Output (tuple): Fixed offset as tuple (start, end)
    """
    # first merge by space, then split by ';'
    start = -1
    end = -1
    for offset in " ".join(offsets).split(';'):
        offset_idx = offset.split(" ")
        if start < 0:
            start = int(offset_idx[0])
        # the second index is the character position next to the last character of an entity but
        # our internal Annotation uses the last character of an entity
        end = int(offset_idx[1]) - 1
    return start, end
