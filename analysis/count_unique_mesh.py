import re, argparse, os, sys
from collections import Counter

def write_to_file_counter(args, file_name, data):
    with open(os.path.join(args.file_path, file_name),'w') as f:
        for key, value in data.most_common():
            f.write(str(key)+'\t'+str(value)+'\n')

def main(args):
    mesh_counter = Counter()
    D_mesh_counter = Counter()
    C_mesh_counter = Counter()
    OMIM_counter = Counter()

    with open(args.file_name,'r') as f:
        data = f.readlines()

        for item in data:
            item_split = item.split('\t')
            for i in item_split[0].split('+'):

                if i[0] =='D':
                    mesh_counter[i] += 1
                    D_mesh_counter[i] += 1
                elif i[0] == 'C':
                    C_mesh_counter[i] += 1
                elif i[0] == 'O':
                    OMIM_counter[i] += 1
            
    write_to_file_counter(args, 'unique_mesh.txt',mesh_counter)
    write_to_file_counter(args, 'D_mesh.txt',D_mesh_counter)
    write_to_file_counter(args, 'C_mesh.txt',C_mesh_counter)
    write_to_file_counter(args, 'OMIM.txt',OMIM_counter)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', type=str, help='Name(Path) of the NCBI file')
    parser.add_argument('--file_path', type=str, help='Path to save files')

    return parser.parse_args(argv)

if __name__ == "__main__":
    # update the path
    sys.argv += ['--file_name',r"------------------/Data/Analyze_development/mesh_counter.txt"]
    sys.argv += ['--file_path',r"------------------/Data/Analyze_development"]

    main(parse_arguments(sys.argv[1:]))