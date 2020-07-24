from process.read_ctd import *
import os

if __name__ == '__main__':
    file1 = r'/media/druv022/Data1/Masters/Thesis/Data/CTD/CTD_diseases.csv'

    omim_dict = read_ctd(file1)

    other_list = []
    with open('/media/druv022/Data1/Masters/Thesis/Data/Analyze_development/OMIM.txt','r') as f:
        data = f.readlines()

    if os.path.exists('C.txt'):
        with open('C.txt','r') as f:
            other_list = f.readlines()

    for i,item in enumerate(data):
        item = item.split('\t')[0]

        if item in omim_dict:
            item = omim_dict[item][0]
        
        if 'C' in item:
            if item+'\n' not in other_list:
                other_list.append(item+'\n')

    with open('C.txt','w+') as f:
        f.writelines(other_list)
