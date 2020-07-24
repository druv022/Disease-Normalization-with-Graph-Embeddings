import csv
import pickle
from tqdm import tqdm

def read_ctd(filepath):
    omim_dict = {}
    with open(filepath, 'r') as f:
        csv_reader = csv.reader(f)

        for i,row in enumerate(tqdm(csv_reader)):
            if i < 29:
                continue
            primary, alt = row[1], row[2]

            if 'MESH' in primary:
                primary = primary.replace('MESH:','')
            else:
                continue

            if 'OMIM' in alt:
                alt = alt.split('|')
                for name in alt:
                    if 'OMIM' in name:
                        if name in omim_dict:
                            values = omim_dict[name]
                            values += [primary]
                        else:
                            omim_dict[name] = [primary]
    
    return omim_dict

if __name__ == '__main__':
    ctd_file = r'/media/druv022/Data1/Masters/Thesis/Data/CTD/CTD_diseases.csv'
    omim_dump = r'/media/druv022/Data1/Masters/Thesis/Data/CTD/omim2mesh_dict.pkl'
    
    omim_dict = read_ctd(ctd_file)

    with open(omim_dump, 'wb') as f:
        pickle.dump(omim_dict, f)