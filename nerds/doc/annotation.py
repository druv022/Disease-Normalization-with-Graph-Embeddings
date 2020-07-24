from uuid import uuid4


class Annotation(object):
    """ A named entity object that has been annotated on a document.

        Note:
            The attributes of this class assume a plain text representation
            of the document, after normalization. For example, `text` will
            be in lower case, if the `norm` parameter of the `to_plain_text`
            method call contains lowercasing. Similarly `offset` refers to
            the offset on the potentially normalized/clean text.

        Attributes:
            text (str): The continuous text snippet that forms the named
                entity.
            label (str): The named entity type, e.g. "PERSON", "ORGANIZATION".
            offset (2-tuple of int): Indices that represent the positions of
                the first and the last letter of the annotated entity in the
                plain text.
            identifier (str, optional): A unique identifier for the annotation.
            discontinued (bool, optional): A flag to indicate the presence
                of discontinued offset.
            uuid (str): The unique identifier of a relation

    """

    def __init__(self, text, label, offset, identifier=None, uuid=None, discontinued=False, score=1.0):
        self.text = text
        self.label = label
        self.offset = offset
        self.discontinued = discontinued
        self.score = score
        self.uuid = uuid or uuid4()
        self.identifier = identifier or "T{}".format(self.uuid)

    def to_inline_string(self):
        """ Returns the annotated entity as a string of the form: label[text].
        """
        return "{}[{}]".format(self.label, self.text)

    def to_inline_string_color(self):
        """ Returns the annotated entity as a string of the form: label[text].
        """
        return "{}[{}]".format("\x1b[34m " + self.label + "\x1b[0m", "\x1b[32m " + self.text + "\x1b[0m")

    def __str__(self):
        return "{}\t{} {} {}\t{}".format(self.identifier, self.label, self.offset[0], self.offset[1], self.text)

    """ The following methods allow us to compare annotations for sorting,
        hashing, etc.

        Note: It doesn't make sense to compare annotations that occur in
        different documents.
    """

    def __eq__(self, other):
        return ((self.text == other.text) and (self.label == other.label) and (self.offset == other.offset))

    def __lt__(self, other):
        if self.offset[0] != other.offset[0]:
            return self.offset[0] < other.offset[0]
        return self.offset[1] < other.offset[1]

    def __gt__(self, other):
        if self.offset[0] != other.offset[0]:
            return self.offset[0] > other.offset[0]
        return self.offset[1] > other.offset[1]

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __hash__(self):
        return hash(self.text + self.label + str(self.offset[0]) + str(self.offset[1]))


class Relation(object):
    """ A relation object that has been annotated on a document.

        Attributes:
            label (str): The relation type, e.g. "Origin", "Effect".
            source_id (str): The identifier of the annotated entity in the
                plain text that is the source argument for the relation.
            target_id (str): The identifier of the annotated entity in the
                plain text that is the target argument for the relation.
            identifier (str, optional): A unique identifier for the relation.
            source_role (str, optional): The role of the annotated entity in
                the plain text that is the source argument for the relation.
            target_role (str, optional): The role of the annotated entity in
                the plain text that is the target argument for the relation.
            uuid (str): The unique identifier of a relation
    """

    def __init__(self,
                 label,
                 source_id,
                 target_id,
                 identifier=None,
                 source_role="arg1",
                 target_role="arg2",
                 score=1.0,
                 uuid=None):
        self.label = label
        self.source_id = source_id
        self.target_id = target_id
        self.source_role = source_role
        self.target_role = target_role
        self.score = score
        self.uuid = uuid or uuid4()
        self.identifier = identifier or "R{}".format(self.uuid)

    def __str__(self):
        return "{}\t{} {}:{} {}:{}".format(self.identifier, self.label, self.source_role, self.source_id,
                                           self.target_role, self.target_id)

    def __eq__(self, other):
        return ((self.label == other.label) and (self.source_id == other.source_id) and (
                self.target_id == other.target_id))

    def __lt__(self, other):
        sid1 = int(self.source_id[1:])
        sid2 = int(other.source_id[1:])
        if sid1 < sid2:
            return True
        if sid1 == sid2:
            tid1 = int(self.target_id[1:])
            tid2 = int(other.target_id[1:])
            return tid1 < tid2
        return False

    def __gt__(self, other):
        sid1 = int(self.source_id[1:])
        sid2 = int(other.source_id[1:])
        if sid1 > sid2:
            return True
        if sid1 == sid2:
            tid1 = int(self.target_id[1:])
            tid2 = int(other.target_id[1:])
            return tid1 > tid2
        return False

    def __le__(self, other):
        sid1 = int(self.source_id[1:])
        sid2 = int(other.source_id[1:])
        if sid1 <= sid2:
            return True
        return False

    def __ge__(self, other):
        sid1 = int(self.source_id[1:])
        sid2 = int(other.source_id[1:])
        if sid1 >= sid2:
            return True
        return False

    def __hash__(self):
        return hash(self.label + self.source_id + self.target_id)


class Normalization(object):
    """ A normalization object that has been annotated on a document.

        Attributes:
            argument_id (str): The identifier of the annotated entity in the
                plain text to which the normalization is associated.
            resource_id (str): An identifier for the external resource from
                which the normalization is taken.
            external_id (str): The identifier of the entry within the
                external resource from which the normalization is taken.
            identifier (str, optional): A unique identifier for the
                normalization.
            label (str, optional): The normalization type, e.g. "Reference".
            preferred_term (str, optional): The preferred term from the
                external resource for the normalization.

    """

    def __init__(self,
                 argument_id,
                 resource_id,
                 external_id,
                 identifier=None,
                 label="Reference",
                 preferred_term="",
                 score=1.0,
                 uuid=None):
        self.argument_id = argument_id
        self.resource_id = resource_id
        self.external_id = external_id
        self.label = label
        self.preferred_term = preferred_term
        self.score = score
        self.uuid = uuid or uuid4()
        self.identifier = identifier or "N{}".format(self.uuid)

    def __str__(self):
        return "{}\t{} {} {}:{}\t{}".format(self.identifier, self.label, self.argument_id, self.resource_id,
                                            self.external_id, self.preferred_term or "--")

    def __eq__(self, other):
        return ((self.label == other.label) and (self.argument_id == other.argument_id) and (
                    self.resource_id == other.resource_id) and (self.external_id == other.external_id))

    def __lt__(self, other):
        id1 = int(self.argument_id[1:])
        id2 = int(other.argument_id[1:])
        return id1 < id2

    def __gt__(self, other):
        id1 = int(self.argument_id[1:])
        id2 = int(other.argument_id[1:])
        return id1 > id2

    def __le__(self, other):
        id1 = int(self.argument_id[1:])
        id2 = int(other.argument_id[1:])
        return id1 <= id2

    def __ge__(self, other):
        id1 = int(self.argument_id[1:])
        id2 = int(other.argument_id[1:])
        return id1 >= id2

    def __hash__(self):
        return hash(self.label + self.argument_id + self.resource_id + self.external_id)
