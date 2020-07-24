import argparse
import sys, os
import re
import copy

class Abstract:
    def __init__(self,identifier):
        self.docID = identifier
        self.filtered_text = []
        self.annotations = []


def main(args):
    abstracts_dict = {}
    
    with open(args.file_name,'r') as f:
        data = f.readlines()

        for i,line in enumerate(data):
            if line is '\n':
                continue
            type_idx = re.findall('[^0-9]*\|t|a+?\|',line)
            if len(type_idx) != 0:
                line_split = line.split('|')
                if line_split[0] not in abstracts_dict:
                    new_abstract = Abstract(line_split[0])
                    new_abstract.filtered_text = line_split[2].replace('\n',' ')
                    abstracts_dict[line_split[0]] = new_abstract
                else:
                    old_abstract = abstracts_dict[line_split[0]]
                    old_abstract.filtered_text = old_abstract.filtered_text +line_split[2]
            else:
                line_split = line.split('\t')
                old_abstract = abstracts_dict[line_split[0]]
                annotations = old_abstract.annotations

                assert ''.join(old_abstract.filtered_text[int(line_split[1]):int(line_split[2])].replace("\"",' ')) == line_split[3], "Text index mismatch"
                annotations += ['T'+str(len(annotations))+'\t'+str('Disease')+' '+line_split[1]+' '+line_split[2] + '\t'+ line_split[3]+'\n']
                annotations += ['N'+str(len(annotations))+'\tReference T'+str(len(annotations)-1)+' MeSH:'+line_split[5].strip('\n')+'\t'+line_split[5]]
    
    if not os.path.exists(args.file_path):
            os.mkdir(args.file_path)

    for key in abstracts_dict:
        new_abstract = abstracts_dict[key]

        with open(args.file_path+'/Document_'+ new_abstract.docID + '.txt', 'w') as f:
            f.writelines(new_abstract.filtered_text)
        
        with open(args.file_path+'/Document_'+ new_abstract.docID + '.ann', 'w') as f:
            f.writelines(new_abstract.annotations)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', type=str, help='Name(Path) of the NCBI file to be converted to BRAT format')
    parser.add_argument('--file_path', type=str, help='Path to converted to BRAT format')

    return parser.parse_args(argv)

if __name__ == "__main__":
    # sys.argv += ['--file_name',"/media/druv022/Data1/Masters/Thesis/Data/NCBIdevelopset_corpus.txt"]
    # sys.argv += ['--file_path','/media/druv022/Data1/Masters/Thesis/Data/Converted_develop']

    main(parse_arguments(sys.argv[1:]))