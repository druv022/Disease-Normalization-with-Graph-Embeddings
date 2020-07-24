import pickle

def read_c2m(filepath):
    c2m_dict = {}
    with open(filepath, 'r') as f:
        data = f.readlines()

    for line in data:
        line = line.split('\t')
        c_id, d_id = line[0].replace('http://id.nlm.nih.gov/mesh/2018/',''), line[1].replace('http://id.nlm.nih.gov/mesh/2018/','').replace('\n','')
        
        if c_id in c2m_dict:
            values = c2m_dict[c_id]
            values += [d_id]
        else:
            c2m_dict[c_id] = [d_id]

    return c2m_dict

if __name__ == '__main__':
    c2m_file = r'/media/druv022/Data1/Masters/Thesis/Data/C2M/C2M_mesh.txt'
    c2m_dump = r'/media/druv022/Data1/Masters/Thesis/Data/C2M/c2m_dump.pkl'

    c2m_dict = read_c2m(c2m_file)

    with open(c2m_dump, 'wb') as f:
        pickle.dump(c2m_dict, f)
    