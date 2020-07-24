from uuid import uuid4

from spacy import displacy
from nerds.util.string import normalize_whitespaces


class Document(object):
    """ Represents a basic input document in the extraction pipeline.

        This is an abstraction that is meant to be extended in order to support
        a variety of document types. Offspring of this class should implement
        the method `to_plain_text` that transforms the input document into its
        plain text representation. The present class returns the text itself,
        thus it can be used with any simple `.txt` file.

        Attributes:
            content (bytes): The byte representation of the document object as
                read from the input stream (e.g. with the `rb` flag).
            encoding (str, optional): Specifies the encoding of the plain
                text. Defaults to 'utf-8'.

        Raises:
            TypeError: If `content` is not a byte stream.
    """

    def __init__(self, content, encoding='utf-8', identifier=None, uuid=None):
        if isinstance(content, bytes):
            self.content = content
        else:
            raise TypeError("Invalid type for parameter 'content'.")
        self.encoding = encoding
        self.uuid = uuid or uuid4()
        self.identifier = identifier or str(self.uuid)

    def __eq__(self, other):
        return self.content == other.content

    def __hash__(self):
        return hash(str(self.content))

    @property
    def plain_text_(self):
        """ Method that transforms a document into its plain text representation.

            Returns:
                str: The content of this document as plain text.
        """
        return self.content.decode(self.encoding)


class AnnotatedDocument(Document):
    """ Represents a document object that has been annotated with entities.

        This class serves primarily two purposes: (i) it holds the training
        data for the NER model, and (ii) it represents an input (unobserved)
        document __after__ it has been annotated by the pipeline.

        Attributes:
            content (bytes): The byte representation of the document object as
                read from the input stream (e.g. with the `rb` flag).
            annotations (list(Annotation), optional): The annotations on the
                plain text representation of the document. If None, it defaults
                to an empty list.
            relations (list(Relation), optional): The relations between the
                text-based annotations for the document. If None, it defaults
                to an empty list.
            normalizations (list(Normalization), optional): The normalizations
                for the text-based annotations for the document. If None, it
                defaults to an empty list.
    """

    def __init__(self,
                 content,
                 annotations=None,
                 relations=None,
                 normalizations=None,
                 encoding='utf-8',
                 identifier=None,
                 uuid=None):
        super().__init__(content=content, encoding=encoding, identifier=identifier, uuid=uuid)
        self.annotations = annotations or []
        self.relations = relations or []
        self.normalizations = normalizations or []
        self.stats = {}
        self.annotation_set = frozenset(self.annotations)
        self.relation_set = frozenset(self.relations)
        self.normalization_set = frozenset(self.normalizations)

    def __eq__(self, other):
        return (self.content == other.content and self.annotation_set == other.annotation_set
                and self.relation_set == other.relation_set and self.normalization_set == other.normalization_set)

    def __hash__(self):
        return hash(str(self.content)) + hash(self.annotation_set) + \
               hash(self.relation_set) + hash(self.normalization_set)

    @property
    def annotated_text_(self):
        """ Method that returns the document's text with inline annotated entities.

            Yields:
                str: Every line of text in the input document, where the
                    entities have been replaced with inline annotations.
        """
        cur_annotation_idx = 0
        text_idx = 0
        annotated_line = ""
        while cur_annotation_idx < len(self.annotations):
            # Iteratively append chunks of text plus the annotation.
            cur_annotation = self.annotations[cur_annotation_idx]
            annotated_line += (self.plain_text_[text_idx:cur_annotation.offset[0]] + cur_annotation.to_inline_string())
            text_idx = cur_annotation.offset[1] + 1
            cur_annotation_idx += 1
        else:
            # If no annotations are left, append the rest of the text.
            annotated_line += self.plain_text_[text_idx:]
        return annotated_line

    @property
    def annotated_colored_text_(self):
        """ Method that returns the document's text with inline annotated entities.

            Yields:
                str: Every line of text in the input document, where the
                    entities have been replaced with inline annotations.
        """
        cur_annotation_idx = 0
        text_idx = 0
        annotated_line = ""
        while cur_annotation_idx < len(self.annotations):
            # Iteratively append chunks of text plus the annotation.
            cur_annotation = self.annotations[cur_annotation_idx]
            annotated_line += (
                    self.plain_text_[text_idx:cur_annotation.offset[0]] + cur_annotation.to_inline_string_color())
            text_idx = cur_annotation.offset[1] + 1
            cur_annotation_idx += 1
        else:
            # If no annotations are left, append the rest of the text.
            annotated_line += self.plain_text_[text_idx:]
        return annotated_line

    def render(self, style, jupyter=True, options={}):

        if style == 'ent':

            visual_doc = [{
                'text':
                    self.plain_text_,
                # don't forget to add 1 to the end index
                'ents': [{
                    'start': ann.offset[0],
                    'end': ann.offset[1] + 1,
                    'label': ann.label
                } for ann in sorted(self.annotations)],
                'title':
                    None
            }]

            return displacy.render(visual_doc, style=style, manual=True, jupyter=jupyter, options=options)

        elif style == 'dep':

            words, arcs = self._visual_words_arcs()
            return displacy.render({
                'words': words,
                'arcs': arcs
            },
                style=style,
                manual=True,
                jupyter=jupyter,
                options=options)

        else:
            raise ValueError("Style '{}' is unknown:'ent' and 'dep' are acceptable.")

    def _visual_words_arcs(self):
        # words for visualisation
        sorted_anns = sorted(self.annotations)
        if not sorted_anns:
            return [], []

        text = self.plain_text_
        wid_word = dict()
        eid_wid = dict()

        prev_end = 0
        idx = 0
        words = []

        for ann in sorted_anns:
            if prev_end < ann.offset[0]:
                prev_words = text[prev_end:ann.offset[0]].split()
                for word in prev_words:
                    wid_word[idx] = word
                    idx += 1
                    words += [{'text': word, 'tag': ''}]

            wid_word[idx] = normalize_whitespaces(text[ann.offset[0]:ann.offset[1] + 1])
            words += [{'text': wid_word[idx], 'tag': ann.label}]
            eid_wid[ann.identifier] = idx
            idx += 1
            prev_end = ann.offset[1] + 1

        # the remaining text
        if prev_end < len(text):
            prev_words = text[prev_end:len(text)].split()
            for word in prev_words:
                wid_word[idx] = word
                words += [{'text': word, 'tag': ''}]
                idx += 1

        # arcs for visualisation
        arcs = []
        for rel in self.relations:
            sid = eid_wid[rel.source_id]
            tid = eid_wid[rel.target_id]

            start = sid if sid < tid else tid
            end = tid if sid < tid else sid
            dir = 'right' if sid < tid else 'left'

            arcs += [{'start': start, 'end': end, 'label': rel.label, 'dir': dir}]

        return words, arcs

    def annotations_by_type(self, ann_type):
        """ Given an annotated document, returns its annotations of a specified
            type. Ignores identifiers.
        :param ann_type: specified type for annotations
        :return: annotations of a specified type
        """
        if ann_type == "annotation":
            return self.annotations
        elif ann_type == "relation":
            return self.relations
        elif ann_type == "norm":
            return self.normalizations
        else:
            raise ValueError("Annotation type '{}' is unknown. It must be one of ('annotation',"
                             "'relation', 'norm').".format(ann_type))

    def annotations_by_label(self, ann_label, ann_type="annotation"):
        """ Get annotations of specified label and type
        Args:
            ann_label: label
            ann_type: type, one from {'annotation', 'relation', 'norm'}

        Returns: annotations of specified label and type
        """
        if not ann_label:
            return self.annotations_by_type(ann_type=ann_type)
        anns_by_label = []
        if type(ann_label) == str:
            label_set = {ann_label}
        else:
            label_set = set(ann_label)
        for ann in self.annotations_by_type(ann_type=ann_type):
            if ann.label in label_set:
                anns_by_label.append(ann)
        return anns_by_label

    def annotations_by_value(self, ann_value, ann_label=None, ann_type="annotation"):
        """ Get annotations of specified label and type
        Args:
            ann_value: value
            ann_label: label
            ann_type: type, one from {'annotation', 'relation', 'norm'}

        Returns: annotations with specified text value, label and type
        """
        if not ann_value:
            return self.annotations_by_label(ann_label=ann_label, ann_type=ann_type)
        anns_by_value = []
        if type(ann_value) == str:
            value_set = {ann_value}
        else:
            value_set = set(ann_value)
        for ann in self.annotations_by_label(ann_label=ann_label, ann_type=ann_type):
            if ann.text in value_set:
                anns_by_value.append(ann)
        return anns_by_value

    def annotation_values_by_label(self, ann_label, ann_type="annotation"):
        """ Get annotation values of specified label and type
        Args:
            ann_label: label
            ann_type: type, one from {'annotation', 'relation', 'norm'}

        Returns: annotation values of specified label and type
        """
        anns_by_label = self.annotations_by_label(
            ann_label=ann_label, ann_type=ann_type)
        ann_values_by_label = {ann.text for ann in anns_by_label}
        return ann_values_by_label
