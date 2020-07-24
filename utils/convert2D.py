from process.read_c2m import *
from process.read_ctd import *

class Convert2D():
    def __init__(self, ctd_file, c2m_file):
        super(Convert2D, self).__init__()
        self.omim_dict = read_ctd(ctd_file)
        self.c2m_dict = read_c2m(c2m_file)

    def transform(self, item):

        item = item.split('|')[0]
        item = item.split('+')[0]

        if 'D' not in item:
            if item in self.omim_dict:
                item = self.omim_dict[item][0]
            if item in self.c2m_dict:
                item = self.c2m_dict[item][0]

        if 'Q' in item:
            idx = item.find('Q')
            item = item[0:idx]
        
        if 'D' in item:
            return item
        else:
            return None