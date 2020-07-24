import os

# base_folder = '/media/druv022/Data1/Masters/Thesis/'

paths = {
    'training': os.path.join('Data', 'Converted_train'),
    'develop': os.path.join('Data', 'Converted_develop'),
    'test': os.path.join('Data', 'Converted_test'),
    'Experiment_folder': os.path.join('Data', 'Experiment'),
    'pre_trained_embeddings': os.path.join('Data', 'Embeddings'),
    'ctd_file': os.path.join('Data','CTD', 'CTD_diseases.csv'),
    'c2m_file': os.path.join('Data','C2M','C2M_mesh.txt'),
    'mesh_file': os.path.join('Data', 'MeSH', 'ASCIImeshd2019.bin'),
    'mesh_tree': os.path.join('Data', 'MeSH', 'mtrees2019.bin'),
    'disease_file': os.path.join('Data', 'MeSH', 'disease_list'),
    'mesh_graph' : os.path.join('Data', 'MeSH', 'mesh_graph'),
    'mesh_graph_disease': os.path.join('Data', 'MeSH', 'mesh_graph_disease'),
    'MeSH_folder': os.path.join('Data', 'MeSH'),
    'MultiTask_folder': os.path.join('Data', 'Multitask'),
    'ELMO_options': os.path.join('Data', 'ELMO', 'elmo_2x4096_512_2048cnn_2xhighway_options.json'),
    'ELMO_weights': os.path.join('Data', 'ELMO', 'elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5'),
    'ELMO_folder': os.path.join('Data', 'ELMO'),
    'MeSH_node_emb': os.path.join('Data', 'MeSH', 'node2vec2', 'embedding'),
    'dump_folder': os.path.join('Data', 'Dump'),
    'NER_model' : os.path.join('Data','Best'),
    'EL_model' : os.path.join('Data', 'Best'),
    'mt_model' : os.path.join('Data','Best'),
    'linear_model' : os.path.join('Data','Best'),
    'node_neighbors':os.path.join('Data','MeSH','neighbor.csv'),
    'node_neighbors_disease':os.path.join('Data','MeSH','neighbor_disease.csv'),
    'node2vec_folder': os.path.join('Data', 'MeSH', 'Node2vec'),
}

node2vec_paths = {
    'dump_process': 'processed.pkl',
    'dump_context_dict': 'context_dict.pkl',
    'dump_context_list': 'context_list.pkl',
    'dump_walks': 'walks.pkl',
    'embedding_text': 'embedding.txt',
    'embedding_temp': 'embedding_temp.txt',
    'embedding': 'embedding.pkl',
}


class Paths():

    def __init__(self, base_folder='/media/druv022/Data1/Masters/Thesis/', node2vec_type='2'):
        self.base_folder = base_folder
        self.node2vec_type = node2vec_type
        self.reset()

    def reset(self):
        self.training = os.path.join(self.base_folder, paths['training'])
        self.develop = os.path.join(self.base_folder, paths['develop'])
        self.test = os.path.join(self.base_folder, paths['test'])
        self.experiment_folder = os.path.join(
            self.base_folder, paths['Experiment_folder'])
        self.pre_trained_embeddings = os.path.join(
            self.base_folder, paths['pre_trained_embeddings'])
        self.ctd_file = os.path.join(self.base_folder, paths['ctd_file'])
        self.c2m_file = os.path.join(self.base_folder, paths['c2m_file'])
        self.MeSH_file = os.path.join(self.base_folder, paths['mesh_file'])
        self.MeSH_tree = os.path.join(self.base_folder, paths['mesh_tree'])
        self.disease_file = os.path.join(self.base_folder, paths['disease_file'])
        self.MeSH_graph = os.path.join(self.base_folder, paths['mesh_graph'])
        self.MeSH_graph_disease = os.path.join(
            self.base_folder, paths['mesh_graph_disease'])
        self.MeSH_folder = os.path.join(self.base_folder, paths['MeSH_folder'])
        self.multitask_folder = os.path.join(
            self.base_folder, paths['MultiTask_folder'])
        self.elmo_options = os.path.join(
            self.base_folder, paths['ELMO_options'])
        self.elmo_weights = os.path.join(
            self.base_folder, paths['ELMO_weights'])
        self.elmo_folder = os.path.join(self.base_folder, paths['ELMO_folder'])
        self.MeSH_node_emb = os.path.join(
            self.base_folder, paths['MeSH_node_emb'])
        self.dump_folder = os.path.join(self.base_folder, paths['dump_folder'])
        self.ner_model = os.path.join(self.base_folder, paths['NER_model'])
        self.ner_model_name = 'Default.pkl'
        self.el_model = os.path.join(self.base_folder, paths['EL_model'])
        self.el_model_name = 'Default.pkl'
        self.mt_model = os.path.join(self.base_folder, paths['mt_model'])
        self.mt_model_name = 'Default.pkl'
        self.linear_model = os.path.join(self.base_folder, paths['linear_model'])
        self.linear_model_name = 'Default.pkl'
        self.MeSH_neighbors = os.path.join(self.base_folder, paths['node_neighbors'])
        self.MeSH_neighbors_disease = os.path.join(self.base_folder, paths['node_neighbors_disease'])

        self.node2vec_base = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type)
        if not os.path.exists(self.node2vec_base):
            os.makedirs(self.node2vec_base)
        self.dump_process = os.path.join(self.base_folder,self.experiment_folder, self.node2vec_type, node2vec_paths['dump_process'])
        self.dump_context_dict = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type, node2vec_paths['dump_context_dict'])
        self.dump_context_list = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type, node2vec_paths['dump_context_list'])
        self.dump_walks = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type, node2vec_paths['dump_walks'])
        self.embedding_text = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type, node2vec_paths['embedding_text'])
        self.embedding_temp = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type, node2vec_paths['embedding_temp'])
        self.embedding = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type, node2vec_paths['embedding'])

        #delete
        self.file1 = r"/media/druv022/Data2/Final/Data/Analyze_train/mesh_dict.pkl"
        self.file2 = r'/media/druv022/Data2/Final/Data/Analyze_test/mesh_dict.pkl'


    def update_node2vec(self):
        self.node2vec_base = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type)
        if not os.path.exists(self.node2vec_base):
            os.makedirs(self.node2vec_base)
        self.dump_process = os.path.join(self.base_folder,self.experiment_folder, self.node2vec_type, node2vec_paths['dump_process'])
        self.dump_context_dict = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type, node2vec_paths['dump_context_dict'])
        self.dump_context_list = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type, node2vec_paths['dump_context_list'])
        self.dump_walks = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type, node2vec_paths['dump_walks'])
        self.embedding_text = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type, node2vec_paths['embedding_text'])
        self.embedding_temp = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type, node2vec_paths['embedding_temp'])
        self.embedding = os.path.join(self.base_folder, self.experiment_folder, self.node2vec_type, node2vec_paths['embedding'])