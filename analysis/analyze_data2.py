import sys, argparse, os, re
from collections import Counter
import pickle
import numpy as np

# def filter_tags(data):
#     pattern_2 = r"<category=\".*?\">"
#     pattern_3 = r"</category>"

#     entity_type = re.findall(pattern_2, data)[0]
#     entity = entity_type.replace(r"<category=","").replace(">","").replace('"','')

#     inner_text = data.replace(entity_type,'').replace(pattern_3,'')

#     return [entity, inner_text]

def write_to_file_dict(args, file_name, data):
    with open(os.path.join(args.file_path,file_name),'w') as f:
        for key in data:
            value = data[key]
            f.write(str(key)+'\t'+str(value)+'\n')

def write_to_file_dict2(args, file_name, data):
    with open(os.path.join(args.file_path,file_name),'w') as f:
        ambiguity = []
        for key in data:
            value = data[key]
            v = set([str(v[1]) for v in value])
            ambiguity.append(len(v))
            f.write(str(key)+'\t'+ str(len(v))+'\t' + '\t'.join(v) +'\n')
        
        f.write('Ambiguity rate: '+ str(np.mean(np.asarray(ambiguity))))

def write_to_file_counter(args, file_name, data):
    with open(os.path.join(args.file_path, file_name),'w') as f:
        count = 0
        keys = []
        for key, value in data.most_common():
            f.write(str(key)+'\t'+str(value)+'\n')
            count += int(value)
            keys.append(key)
        
        f.write('Total count: '+ str(count)+'\n')
        f.write('Text: '+str(len(keys))+'\n')

def main(args):
    mesh_dict = {}
    mesh_counter = Counter()
    word_counter = Counter()
    entity_counter = Counter()
    inner_text_counter = Counter()
    tags_counter = Counter()
    inner_text_tag_dict = {}


    with open(args.file_name,'r') as f:
        data = f.readlines()

        if not os.path.exists(args.file_path):
            os.mkdir(args.file_path)

        for line in data:
            if line is '\n':
                continue
            type_idx = re.findall('[^0-9]*\|t|a+?\|',line)
            if len(type_idx) != 0:
                line_split = line.split('|')
                line_text_words = line_split[2].split()
                for word in line_text_words:
                    word_counter[word] += 1
            else:
                line_split = line.split('\t')
                if len(line_split) < 3:
                    print(len(line_split))
                    continue
                if line_split[5].replace('\n','') not in mesh_dict:
                    mesh_dict[line_split[5].replace('\n','')] = [[line_split[4],line_split[3]]]
                else:
                    value = mesh_dict[line_split[5].replace('\n','')]
                    if [line_split[4],line_split[3]] not in value:
                        value.append([line_split[4],line_split[3]])
                    mesh_dict[line_split[5].replace('\n','')] = value
                
                if line_split[3] in inner_text_tag_dict:
                    cnt = inner_text_tag_dict[line_split[3]]
                    cnt[line_split[4]] += 1
                    inner_text_tag_dict[line_split[3]] = cnt
                else:
                    cnt = Counter()
                    cnt[line_split[4]] += 1
                    inner_text_tag_dict[line_split[3]] = cnt

                mesh_ids = line_split[5].split('|')
                for id in mesh_ids:
                    mesh_counter[id.replace('\n','')] += 1

                entity_counter[line_split[4]] += 1
                inner_text_counter[line_split[3]] += 1
                tags_counter[line_split[4]+'\t'+line_split[3]] += 1

    write_to_file_dict(args,'mesh_id.txt', mesh_dict)

    write_to_file_dict2(args,'mesh_id2.txt', mesh_dict)

    with open(os.path.join(args.file_path,'mesh_dict.pkl'),'wb+') as f:
        pickle.dump(mesh_dict, f)

    with open(os.path.join(args.file_path,'inner_text_tag.txt'),'w') as f:
        for k in inner_text_tag_dict:
            cnt= inner_text_tag_dict[k]
            f.write(str(k))
            for key, value in cnt.most_common():
                f.write('\t'+str(key)+'\t'+str(value)+',')
            f.write('\n')

    write_to_file_counter(args, 'mesh_counter.txt', mesh_counter)
    write_to_file_counter(args, 'entity_counter.txt', entity_counter)
    write_to_file_counter(args, 'inner_text_counter.txt', inner_text_counter)
    write_to_file_counter(args, 'tags_counter.txt', tags_counter)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', type=str, help='Name(Path) of the NCBI file')
    parser.add_argument('--file_path', type=str, help='Path to save files')

    return parser.parse_args(argv)

if __name__ == "__main__":
    sys.argv += ['--file_name',r"C:\Thesis\Data\NCBItrainset_corpus\NCBItrainset_corpus.txt"]
    sys.argv += ['--file_path',r"C:\Thesis\Data\NCBItrainset_corpus\Analyze_train"]

    # Generate different types of stats of  NCBI dataset

    main(parse_arguments(sys.argv[1:]))