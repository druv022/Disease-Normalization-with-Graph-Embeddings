import argparse
import os, sys
import pickle
from utils.convert2D import Convert2D

def read_dict(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    return data

def clean_keys(data):
    r_keys = []
    for i in data:
        if '|' in i:
            i = i.split('|')
        elif '+' in i:
            i = i.split('+')
        else:
            i = [i]
        r_keys.extend([j.strip() for j in i])

    return r_keys

def get_unique_id(train_data1, test_data2, ctd_file, c2m_file):
    data1 = read_dict(train_data1)
    data2 = read_dict(test_data2)

    data1 = clean_keys(data1.keys())
    data2 = clean_keys(data2.keys())

    toD = Convert2D(ctd_file, c2m_file)
    unique_id = []
    for i in data2:
        if i in data1:
            continue
        else:
            i = toD.transform(i)
            if i:
                unique_id.append(i)

    return list(set(unique_id))

def main():
    file1 = r"/media/druv022/Data2/Final/Data/Analyze_train/mesh_dict.pkl"
    file2 = r'/media/druv022/Data2/Final/Data/Analyze_test/mesh_dict.pkl'

    c2m = '/media/druv022/Data2/Final/Data/C2M/C2M_mesh.txt'
    ctd = '/media/druv022/Data2/Final/Data/CTD/CTD_diseases.csv'

    unique_id = get_unique_id(file1, file2, ctd, c2m)
    print('Here')

# if __name__ == '__main__':
    
#     file1 = r"/media/druv022/Data2/Final/Data/Analyze_train/mesh_dict.pkl"
#     file2 = r'/media/druv022/Data2/Final/Data/Analyze_test/mesh_dict.pkl'

#     c2m = '/media/druv022/Data2/Final/Data/C2M/C2M_mesh.txt'
#     c2d = '/media/druv022/Data2/Final/Data/CTD/CTD_diseases.csv'

#     data1 = read_dict(file1)
#     data2 = read_dict(file2)

#     get_unique_id(data2.keys(), data1.keys(), c2m, c2d)
