import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import csv, os
from utils.mesh import MESH, read_mesh_file
from config.paths import * 

def save_meshgraph(mesh_dict, filepath, graph_path):
    DG = nx.DiGraph()
    tree_label_dict = {}
    tree_node_dict = {}
    c_list = set()
    edge_node_list = set()

    with open(filepath,'r') as f:
        data = f.readlines()

    mesh_h_dict = {mesh_dict[key].mesh_h: key for key in mesh_dict}
   
    for line in data:
        label, tree = line.replace('\n','').split(';')

        if ',' in label:
            label_split = label.split(',')
            label_split.reverse()
            label_split[0] = label_split[0].strip()
            label = ' '.join(label_split)

        if label not in mesh_h_dict:
            print(label +" not in MeSH dict.")
            continue
        else:
            key = mesh_h_dict[label]

        if key in tree_label_dict:
            value = tree_label_dict[key]
            tree_label_dict[key] = value + [tree]
        else:
            tree_label_dict[key] = [tree]
        tree_node_dict[tree] = key

    # build graph
    for key in tree_label_dict:
        values = tree_label_dict[key]

        for value in values:
            value_split = value.split('.')
            if 'C' in value_split[0]:
                c_list.add(key+'\n')
            else: # uncomment for disease
                continue
            
            edge_node_list.add(key+'\n')

            value_split.reverse()

            for i in range(len(value_split)-1):
                source_node = value_split[i+1:]
                source_node.reverse()
                source_node = '.'.join(source_node)

                target_node = value_split[i:]
                target_node.reverse()
                target_node = '.'.join(target_node)

                DG.add_weighted_edges_from([(tree_node_dict[source_node],tree_node_dict[target_node],1)], weight='dummy' ,source=source_node, target=target_node)

    nx.write_gpickle(DG,graph_path)

    with open(os.path.join(os.path.dirname(filepath),'disease_list'), 'w') as f:
        f.writelines(c_list)

    with open(os.path.join(os.path.dirname(filepath),'all_D_list'), 'w') as f:
        f.writelines([str(i) + '\n' for i in tree_label_dict.keys()])

    with open(os.path.join(os.path.dirname(filepath),'edge_node_list'), 'w') as f:
        f.writelines(list(edge_node_list))

    return DG

def load_meshgraph(path):
    return nx.read_gpickle(path)


def analyze_graph(graph, filepath):

    with open(filepath, 'w') as f:
        write_entry = csv.writer(f, dialect='excel')
        write_entry.writerow(['Mesh Heading', 'Neighbors'])
        for key in graph.adj:
            write_entry.writerow([key, len(graph.adj[key])])

    sorted_degree = sorted(d for n,d in graph.degree())
    # sorted_in
    cnt= Counter(sorted_degree)

    plt.grid(True)
    plt.hist(sorted_degree,bins=range(min(sorted_degree), max(sorted_degree)+1))
    plt.show()
    print("Print max degree: ",max(sorted_degree))
    print("Print counter (degree:frequency)",cnt)
    print("Number of nodes: ", len(graph.node))

def export_graph(graph, folder_path):
    nx.write_gexf(graph, folder_path)


if __name__ == '__main__':
    """ Generate different MeSH stats
    """

    base_path = '/media/druv022/Data2/Final'
    paths = Paths(base_path)

    meshtree_path = paths.MeSH_tree
    # graph_path = paths.MeSH_graph 
    graph_path = paths.MeSH_graph_disease
    # csv_file = paths.MeSH_neighbors
    csv_file = paths.MeSH_neighbors_disease
    folder_path= paths.MeSH_folder
    mesh_file = paths.MeSH_file

    mesh_dict = read_mesh_file(mesh_file)
    
    save_meshgraph(mesh_dict, meshtree_path, graph_path)
    graph = load_meshgraph(graph_path)

    analyze_graph(graph, csv_file)

    # export_file_path = os.path.join(folder_path,'mesh_graph.gexf')
    export_file_path = os.path.join(folder_path,'mesh_graph_disease.gexf')
    export_graph(graph, export_file_path)

