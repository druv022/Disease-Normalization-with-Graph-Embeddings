

class MESH(object):

    def __init__(self,mesh_h, scope_note, entry_terms, unique_id, other=None):
        self._scope_note = scope_note
        self._entry_terms = entry_terms
        self._unique_id = unique_id
        self._mesh_h = mesh_h
        self._other =other

    def __str__(self):
        return self._mesh_h

    def __eq__(self, other):
        return self._mesh_h == other.mesh_h

    def object_to_string(self):
        return str(self.__dict__)
    
    @property
    def scope_note(self):
        return self._scope_note

    @property
    def entry_terms(self):
        return self._entry_terms
    
    @property
    def unique_id(self):
        return self._unique_id

    @property
    def mesh_h(self):
        return self._mesh_h

    @property
    def other(self):
        return self._other

def read_mesh_file(filepath):

    with open(filepath, 'r', encoding='utf8') as f:
        data = f.readlines()

    mesh_dict = {}
    add_recod = False

    scope_note = ''
    entry_terms = []
    unique_id = ''
    mesh_h = ''
    other = []

    for line in data:
        if line.replace('\n','') == '*NEWRECORD':
            if add_recod:
                mesh_dict[unique_id] = MESH(mesh_h, scope_note, entry_terms, unique_id, other=other)
                add_recod = False
            scope_note = ''
            entry_terms = []
            unique_id = ''
            mesh_h = ''
            other = []
            continue

        if line == '\n':
            continue

        eq_indx = line.find('=')

        label = line[:eq_indx]
        label = label.strip(' ')

        value = line[eq_indx+2:].strip('\n')

        if label == 'MH':
            mesh_h = value
            add_recod = True
            value_split = value.split(',')
            value_split.reverse()
            value_split[0] = value_split[0].strip()
            value_split = ' '.join(value_split)
            mesh_h = value_split
            entry_terms.append(value_split.split())
        elif label == 'ENTRY' or label == 'PRINT ENTRY':
            value = value.split('|')[0]
            value_split = value.split(',')
            value_split.reverse()
            value_split[0] = value_split[0].strip()
            value_split = ' '.join(value_split)
            value_split = value_split.split()
            entry_terms.append(value_split)
        elif label == 'FX':
            other.append(value)
        elif label == 'MS':
            # TODO: use proper tokenizer
            scope_note_sentences = value.split('.')
            scope_note = [[r'<s>']+ sent.split()+[r'<\s>'] for sent in scope_note_sentences if len(sent.split()) > 0]
        elif label == 'UI':
            unique_id = value
    
    # add last entry
    if add_recod:
        mesh_dict[unique_id] = MESH(mesh_h, scope_note, entry_terms, unique_id, other=other)

    return mesh_dict