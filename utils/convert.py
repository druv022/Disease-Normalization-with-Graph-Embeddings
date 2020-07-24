import argparse
import sys, os
import re
import copy

class Abstract():

    def __init__(self,text_data):
        self.data = text_data.split("\t")
        self.meshID = self.data[0]
        self.text = self.data[1] + self.data[2]
        self.filtered_text = []
        self.annotations = self._annotate()
        

    def _annotate(self):
        raw_text = copy.deepcopy(self.text)
        annotations = ''
        
        pattern_1 = r"<category=.*?</category>"
        pattern_2 = r"<category=\".*?\">"
        pattern_3 = r"</category>"

        start_position = 0

        tags = re.findall(pattern_1, raw_text)
        for i,tag in enumerate(tags):

            entity_type = re.findall(pattern_2, tag)[0]
            entity = entity_type.replace(r"<category=","").replace(">","").replace('"','')

            inner_text = tag.replace(entity_type,'').replace(pattern_3,'')
            start = raw_text.find(tag)

            self.filtered_text += raw_text[start_position:start]
            tag_start = len(self.filtered_text)
            self.filtered_text += inner_text
            tag_end = tag_start + len(inner_text)

            end = start + len(tag)
            raw_text = raw_text.replace(raw_text[0:end],'')

            assert ''.join(self.filtered_text[tag_start:tag_end]) == inner_text, "Text index mismatch"
            annotations = annotations +'T'+str(i) + '\t' + str('Disease') + ' ' + str(tag_start) + ' ' + str(tag_end) + '\t' + str(inner_text) + '\n'
            
        self.filtered_text = ''.join(self.filtered_text)
        return annotations


def main(args):

    with open(args.file_name,'r') as f:
        data = f.readlines()

        if not os.path.exists(args.file_path):
            os.mkdir(args.file_path)

        for item in data:
            new_abstract = Abstract(item)

            with open(args.file_path+'/Document_'+ new_abstract.meshID + '.txt', 'w') as f:
                f.writelines(new_abstract.filtered_text)
            
            with open(args.file_path+'/Document_'+ new_abstract.meshID + '.ann', 'w') as f:
                f.writelines(new_abstract.annotations)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', type=str, help='Name(Path) of the NCBI file to be converted to BRAT format')
    parser.add_argument('--file_path', type=str, help='Path to converted to BRAT format')

    return parser.parse_args(argv)

if __name__ == "__main__":
    sys.argv += ['--file_name','/media/druv022/Data1/Masters/Thesis/Data/NCBI_corpus_development.txt']
    sys.argv += ['--file_path','/media/druv022/Data1/Masters/Thesis/Data/Converted_development']

    main(parse_arguments(sys.argv[1:]))