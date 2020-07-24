from nerds.doc.document import Document


class InputDocumentFile(object):

    def __init__(self, file):
        self.content = self._read_file(file)

    def _read_file(self, file):
        return str(file.read())

    @property
    def document(self):
        return Document(self.content.encode('utf-8'))
