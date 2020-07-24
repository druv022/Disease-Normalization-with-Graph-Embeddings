import argparse
import os, sys
from analyze_data2 import write_to_file_dict
import pickle

def get_dict(data):
    return_dict = {}
    for line in data:
        line = line.replace('\n','')
        line = line.split('\t')
        if line[0] in return_dict:
            return_dict += [line[i] for i in range(1,len(line))]
        else:
            return_dict[line[0]] = [line[i] for i in range(1,len(line))]

    return return_dict

def read_dict(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    return data

def run_common(data1, data2, file_name):
    common_dict = {}
    different_dict = {}
    common_text_len = 0
    different_text_len = 0
    for key in data1:
        if key in data2:
            item_tr = [i[1] for i in data1[key]]
            item_test = [i[1] for i in data2[key]]
            common_text = []
            different_text = []
            for i in item_test:
                if i in item_tr and i not in common_text:
                    common_text.append(i)
                elif i not in item_tr:
                    different_text.append(i)

            common_text_len += len(common_text)
            different_text_len += len(different_text)

            if key in common_dict:
                value = common_dict[key]
                [common_text.remove(i) for i in common_text if i in value]
                value += common_text
                common_dict[key] = value
                different_dict[key] += different_text
            else:
                common_dict[key] = common_text
                different_dict[key] = different_text

    with open(file_name,'w') as f:
        for idx, key in enumerate(common_dict):
            value = common_dict[key]
            f.write('------------------Index: ' +str(idx) + ' -------------------\n')
            f.write('Common: '+str(key)+'\t'+str(value)+'\n')
            if key in different_dict:
                value = different_dict[key]
                f.write('Different: '+str(key)+'\t'+str(value)+'\n')

        f.write('Total common mentions: '+str(common_text_len)+' in total of '+str(common_text_len+different_text_len))

    return common_dict

def find_common_mentions(data, item_list):
    for key in data:
        for i in data[key]:
            if i[1] not in item_list:
                item_list.append(i[1])

def find_unique_in_all(data1, data2, data3, list_of_list=False):
    unique_item = []
    for i in data1:
        if not list_of_list:
            i = [i]
        for item in i:
            if list_of_list:
                item = item[1]
            if item in unique_item:
                continue
            for j in data2:
                if list_of_list:
                    j = [x[1] for x in j]
                if item in j:
                    for k in data3:
                        if list_of_list:
                            k = [x[1] for x in k]
                        if item in k and item not in unique_item:
                            unique_item.append(item)

    return unique_item

def main(args):

    data_1 = read_dict(args.file_name1)
    data_2 = read_dict(args.file_name2)
    data_3 = read_dict(args.file_name3)

    common_dict1 = run_common(data_1, data_2, args.new_file_name1)
    common_dict2 = run_common(data_1, data_3, args.new_file_name2)
    common_dict3 = run_common(data_2, data_3, args.new_file_name3)

    unique_mention_in_all = find_unique_in_all(data_1.values(), data_2.values(), data_3.values(), list_of_list=True)
    unique_id_in_all = find_unique_in_all(data_1.keys(), data_2.keys(), data_3.keys())

    common_keys = set(list(data_1.keys()) + list(data_2.keys()) + list(data_3.keys()))
    common_mention = []
    find_common_mentions(data_1, common_mention)
    find_common_mentions(data_2, common_mention)
    find_common_mentions(data_3, common_mention)

    with open(args.new_file_name4, 'w+') as f:
        f.write('Total unique keys: '+str(len(common_keys)) +'\n')
        f.write('Total unique mentions: '+str(len(common_mention))+'\n')
        f.write('Total Unique keys in all: '+str(len(unique_id_in_all))+'\n')
        f.write('Total Unique mentions in all: '+str(len(unique_mention_in_all))+'\n')

        f.write(str(common_keys)+'\n')
        f.write(str(common_mention)+'\n')



    
# def main(args):
#     common_dict = {}

#     with open(args.file_name1,'r') as f:
#         data_1 = f.readlines()
#         data_1 = get_dict(data_1)

#     with open(args.file_name2,'r') as f:
#         data_2 = f.readlines()
#         data_2 = get_dict(data_2)

#     for key in data_1:
#         if key in data_2:
#             if key in common_dict:
                
#                 common_dict[key] += [[data_1[key], data_2[key]]]
#             else:
#                 # item_tr = [ for i in data_1]
#                 common_dict[key] = [[data_1[key], data_2[key]]]

#     with open(args.new_file_name,'w') as f:
#         for key in common_dict:
#             value = common_dict[key]
#             f.write(str(key)+'\t'+str(value)+'\n')




def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name1', type=str, help='Path of the train file')
    parser.add_argument('--file_name2', type=str, help='Path of the develop file')
    parser.add_argument('--file_name3', type=str, help='Path of the test file')
    parser.add_argument('--new_file_name1', type=str, help='Path of the new file')
    parser.add_argument('--new_file_name2', type=str, help='Path of the new file')
    parser.add_argument('--new_file_name3', type=str, help='Path of the new file')
    parser.add_argument('--new_file_name4', type=str, help='Path of the new file')

    return parser.parse_args(argv)

if __name__ == "__main__":
    """ analyze common between training, validation and test set
    """
    # update the path
    sys.argv += ['--file_name1',r"-------------/Data/Analyze_train/mesh_dict.pkl"]
    sys.argv += ['--file_name2',r'-------------/Data/Analyze_develop/mesh_dict.pkl']
    sys.argv += ['--file_name3',r'-------------/Data/Analyze_train/mesh_dict.pkl']
    sys.argv += ['--new_file_name1',r'-------------/Data/Analyze_develop/common_mesh_id.txt']
    sys.argv += ['--new_file_name2',r'-------------/Data/Analyze_test/common_mesh_id.txt']
    sys.argv += ['--new_file_name3',r'-------------/Data/Analyze_test/common_mesh_id_with_dev.txt']
    sys.argv += ['--new_file_name4',r'-------------/Data/Analyze_train/common_all.txt']

    main(parse_arguments(sys.argv[1:]))