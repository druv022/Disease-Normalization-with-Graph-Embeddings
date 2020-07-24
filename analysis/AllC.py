import os

if __name__ == '__main__':
    """ identify all C mesh
    """
    other_list = []
    # update the path
    with open('--------/Data/Analyze_Test/C_mesh.txt','r') as f:
        data = f.readlines()

    if os.path.exists('C.txt'):
        with open('C.txt','r') as f:
            other_list = f.readlines()

    for i,item in enumerate(data):
        item = item.split('\t')[0]
        
        if 'C' in item:
            if item+'\n' not in other_list:
                other_list.append(item+'\n')

    with open('C.txt','w+') as f:
        f.writelines(other_list)