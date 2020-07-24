import codecs
from pathlib import Path

from nerds.output.base import DataOutput
from nerds.util.logging import log_progress, get_logger

log = get_logger()


class BratOutput(DataOutput):
    """
    Writes data from produced annotation files to a specified folder.

    Attributes:
        path_to_folder (str): The path to the output folder where the files
            will be written.
    """

    def __init__(self, path_to_files):
        super().__init__(path_to_files)

    def transform(self, X, y=None, file_names=None, write_scores=False):
        """
        Transforms the available documents into brat files.
        X (list): The documents to be transformed into Brat.
        file_names (list): The optional list of file names for output brat files.
            Its size must be equal to the number of input documents.
        """

        log.info("Writing documents to '{}'".format(self.path))

        if file_names and len(file_names) != len(X):
            raise ValueError("The number of file names is not equal to the number of documents.")

        if file_names:
            for file_name, doc in zip(file_names, X):
                doc.identifier = file_name

        for idx, doc in enumerate(X):
            file_name = doc.identifier
            path_to_txt_file = Path(self.path.joinpath(file_name + ".txt"))
            path_to_ann_file = Path(self.path.joinpath(file_name + ".ann"))

            with codecs.open(path_to_txt_file, "w", doc.encoding) as f:
                f.write(doc.plain_text_)

            self._write_brat_ann_file(doc.annotations, doc.relations, doc.normalizations, path_to_ann_file,
                                      write_scores)
            log_progress(log, idx, len(X))

        return None

    def _write_brat_ann_file(self, annotations, relations, normalizations, path_to_ann_file, write_scores):
        """ Helper function to write brat annotations.
        """

        # always create an empty annotation file when saving a document
        # because a Brat server throws errors otherwise
        # if not annotations and not relations and not normalizations:
        #     return

        # this is for writing confidence scores if required
        note_type = "AnnotatorNotes"

        # create an annotation file and write annotations, relations, and normalizations
        with codecs.open(path_to_ann_file, "w", "utf-8") as f:
            for i, annotation in enumerate(annotations):

                # Must be exactly 3 things, if they are entity related.
                # e.g.: "TEmma2\tGrant 475 491\tGIA G-14-0006063".
                tag = annotation.identifier

                offsets = ''
                # Write disjoint offsets if present, e.g. from original Brat
                # input files
                if hasattr(annotation, 'real_offset'):
                    for i, offset in enumerate(annotation.real_offset):
                        offsets += '{} {}'.format(offset[0], offset[1] + 1)
                        if i < len(annotation.real_offset) - 1:
                            offsets += ';'

                else:
                    # This is how brat's offsets are!
                    start_offset = annotation.offset[0]
                    # Corrected: this is actually wrong because in Brat
                    # (see http://brat.nlplab.org/standoff.html) the second
                    # index is the character position next to the last
                    # character of an entity; the Brat server raises parsing
                    # errors otherwise.
                    # end_offset = annotation.offset[1] - 1
                    end_offset = annotation.offset[1] + 1

                    offsets += '{} {}'.format(start_offset, end_offset)

                to_write = "{}\t{} {}\t{}\n".format(tag, annotation.label, offsets, annotation.text)

                f.write(to_write)

                if write_scores:
                    # add confidence values as notes
                    to_write = "#{}\t{} {}\t{:.3f}\n".format(tag, note_type, tag, annotation.score)

                    f.write(to_write)

            for i, relation in enumerate(relations):

                # Must be exactly 2 things, if they are entity related.
                # e.g.: "R2\tReceived Arg1:T1 Arg2:T2".
                tag = relation.identifier

                to_write = "{}\t{} {}:{} {}:{}\n".format(tag, relation.label, relation.source_role, relation.source_id,
                                                         relation.target_role, relation.target_id)

                f.write(to_write)

                if write_scores:
                    # add confidence values as notes
                    to_write = "#{}\t{} {}\t{:.3f}\n".format(tag, note_type, tag, relation.score)

                    f.write(to_write)

            for i, normalization in enumerate(normalizations):

                # Must be exactly 3 things, if they are normalization
                # related, e.g.:
                # "N1\tReference T1 Wikipedia:534366\tGrant (money)".
                tag = normalization.identifier

                to_write = "{}\t{} {} {}:{}\t{}\n".format(tag, normalization.label, normalization.argument_id,
                                                          normalization.resource_id, normalization.external_id,
                                                          normalization.preferred_term)

                f.write(to_write)

                if write_scores:
                    # add confidence values as notes
                    to_write = "#{}\t{} {}\t{:.3f}\n".format(tag, note_type, tag, normalization.score)

                    f.write(to_write)

    def save(self, file_path):
        """ Do nothing """

    def load(self, file_path):
        """ Do nothing """
