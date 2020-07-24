from node2vec.node2vec1 import train_node2vec as node2vec1
from node2vec.node2vec2 import train_node2vec as node2vec2
from config.paths import Paths
from config.params import EL_params, Node2vec_Params
from EL.EL_partA import main
import os
from EL.EL_GCN import main as main_gcn

# Update the path
base_path = '/media/druv022/Data2/Final'

start_from = 5

if start_from < 2:
    paths = Paths(base_path, node2vec_type='1')
    params = EL_params()
    params.batch_size = 32
    params.use_elmo = True
    params.randomize = True
    params.num_epochs = 100

    n_params = Node2vec_Params(p = 1, q = 1)
    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_23',str(i))
        paths.update_node2vec()
        params.output = os.path.join(paths.experiment_folder, 'output.txt')
        node2vec1(paths, n_params)
        main(paths, params)


if start_from < 3:
    paths = Paths(base_path, node2vec_type='4')
    params = EL_params()
    params.batch_size = 32
    params.use_elmo = True
    params.randomize = True
    params.num_epochs = 100

    n_params = Node2vec_Params(p = 1, q = 1)
    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_24',str(i))
        paths.update_node2vec()
        params.output = os.path.join(paths.experiment_folder, 'output.txt')
        node2vec2(paths, n_params)
        main(paths, params)

if start_from < 4:
    paths = Paths(base_path, node2vec_type='2')
    params = EL_params()
    params.batch_size = 32
    params.use_elmo = True
    params.randomize = True
    params.num_epochs = 100

    n_params = Node2vec_Params(p = 2, q = 1)
    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_25',str(i))
        paths.update_node2vec()
        params.output = os.path.join(paths.experiment_folder, 'output.txt')
        node2vec1(paths, n_params)
        main(paths, params)

if start_from < 5:
    paths = Paths(base_path, node2vec_type='3')
    params = EL_params()
    params.batch_size = 32
    params.use_elmo = True
    params.randomize = True
    params.num_epochs = 100

    n_params = Node2vec_Params(p = 1, q = 2)
    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_26',str(i))
        paths.update_node2vec()
        params.output = os.path.join(paths.experiment_folder, 'output.txt')
        node2vec1(paths, n_params)
        main(paths, params)


# EL GCN
if start_from < 6:
    paths = Paths(base_path)
    params = EL_params()
    params.batch_size = 32
    params.use_elmo = True
    params.randomize = True
    params.num_epochs = 500

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_27',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        params.output = os.path.join(paths.experiment_folder, 'output.txt')
        main_gcn(paths, params)