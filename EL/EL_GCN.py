from models.gcn import GCN
from models.EL_models import EL_GCN
from scipy import sparse
import networkx as nx
import numpy as np
from EL.EL_utils import *
import os
import pickle
import torch
from time import time
from sklearn.metrics import classification_report, accuracy_score
from tensorboardX import SummaryWriter
from nerds.doc.bio import transform_annotated_documents_to_bio_format
from nerds.dataset.split import split_annotated_documents
from nerds.input.brat import BratInput
import random
from analysis.get_analysis import get_unique_id

def train_gcn(paths, params, X, annotated_docs_tr, predictions_tr, X_valid, annotated_docs_v, predictions_v, scope_text, id2idx_dict, idx2id_dict, mesh_graph, device=torch.device('cpu')):
    options_file = paths.elmo_options
    weight_file = paths.elmo_weights
    ELMO_folder = paths.elmo_folder
    elmo_dim = params.elmo_dim
    elmo = Elmo(options_file, weight_file, 2,dropout=0)
    elmo.to(device)

    scope_elmo_emb, nodes = get_scope_elmo(elmo, ELMO_folder, scope_text, elmo_dim, idx2id_dict, id2idx_dict, device=device)

    x_data, _ = construct_data(X,annotated_docs_tr, predictions_tr, scope_text, id2idx_dict, paths.ctd_file, paths.c2m_file, use_ELMO=params.use_elmo, elmo_model=elmo, elmo_dim=params.elmo_dim, device=device)
          
    x_v_data, _ = construct_data(X_valid,annotated_docs_v, predictions_v, scope_text, id2idx_dict, paths.ctd_file, paths.c2m_file, use_ELMO=params.use_elmo, elmo_model=elmo, elmo_dim=params.elmo_dim, device=device)

    node_list = list(idx2id_dict.values())

    # check if adjacancy matrix already calculated
    if not os.path.exists(os.path.join(paths.MeSH_folder, 'a_hat_matrix')):
        a_hat = get_adjacancy_matrix(mesh_graph, node_list)

        # save node_list and the calculated adjacancy matrix
        with open(os.path.join(paths.MeSH_folder, 'node_list'), 'wb') as f:
            pickle.dump(node_list, f)
        data = sparse.coo_matrix(a_hat)
        with open(os.path.join(paths.MeSH_folder, 'a_hat_matrix'), 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(os.path.join(paths.MeSH_folder, 'a_hat_matrix'), 'rb') as f:
            data = pickle.load(f)

    i = torch.tensor([data.row, data.col], dtype=torch.long, device=device)
    v = torch.tensor(data.data, dtype=torch.float32, device=device)
    a_hat = torch.sparse.FloatTensor(i,v, torch.Size([len(node_list), len(node_list)])).cuda()

    model = EL_GCN(elmo_dim, elmo_dim, scope_elmo_emb, hidden_dim=elmo_dim*2, dropout=0.5)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs= params.num_epochs
    batch_size= params.batch_size

    train_acc = 0
    val_acc = 0
    best_model = None
    ep = 0
    with open(os.path.join(params.output), 'w') as f:
        for epoch in range(epochs):
            # np.random.shuffle(x_data)

            training_loss= 0
            count = 0
            start_time = time()
            for t, x, _, y, mask in minibatch(x_data,scope_elmo_emb, id2idx=id2idx_dict, use_elmo=True, elmo_model=elmo, n_samples=0, batch_size=batch_size, device=device):
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                optimizer.zero_grad()

                mask = mask.unsqueeze(2).expand(-1,-1,x.shape[2])
                x = mask*x
                x = torch.mean(x, dim=1)
                x = model(x, a_hat)

                loss = nn.functional.cross_entropy(x, y)

                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                count += 1

            print(f'Epoch: {epoch}\tLoss: {training_loss/count}\tTime: {time()-start_time}')
            f.writelines(f'Epoch: {epoch}\tLoss: {training_loss/count}\tTime: {time()-start_time}')

            start_time = time()
            with torch.no_grad():
                pred_label, label = [], []
                for t, x, y, mask in minibatch_val(x_data, batch_size=batch_size):
                    x, mask = x.to(device), mask.to(device).float()

                    mask = mask.unsqueeze(2).expand(-1,-1,x.shape[2])
                    x = mask*x
                    x = torch.mean(x, dim=1)
                    x = model(x, a_hat)

                    x = nn.functional.softmax(x, dim=1)
                    _, max_idx = torch.max(x, dim=1)
                    label.extend(y)
                    _= [pred_label.append(idx2id_dict[i.item()]) for i in max_idx]
                
                # print(classification_report(label, pred_label))
                acc = accuracy_score(label, pred_label)
                print(f'Train Acc: {acc}\tTime: {time()-start_time}')
                f.writelines(f'Train Acc: {acc}\tTime: {time()-start_time}')
                
                if acc > train_acc:
                    train_acc = acc
                
                print('Best train acc: ', train_acc)
                f.writelines(f"Best training acc: {train_acc}")

                pred_label, label = [], []
                for t, x, y, mask in minibatch_val(x_v_data, batch_size=batch_size):
                    x, mask = x.to(device), mask.to(device).float()

                    mask = mask.unsqueeze(2).expand(-1,-1,x.shape[2])
                    x = mask*x
                    x = torch.mean(x, dim=1)
                    x = model(x, a_hat)

                    x = nn.functional.softmax(x, dim=1)
                    _, max_idx = torch.max(x, dim=1)
                    label.extend(y)
                    _= [pred_label.append(idx2id_dict[i.item()]) for i in max_idx]

                # print(classification_report(label, pred_label))
                acc = accuracy_score(label, pred_label)
                print(f'Valid Acc: {acc}\tTime: {time()-start_time}')
                f.writelines(f'Valid Acc: {acc}\tTime: {time()-start_time}')
                
                if acc > val_acc:
                    val_acc = acc
                    print('Updating model for acc: ', acc)
                    f.writelines(f'Updating model for acc: {acc}')
                    best_model = model
                    ep = epoch
                
                print(f'Best val acc: {val_acc}\t epoch: {ep}')
                f.writelines(f'Best val acc: {val_acc}\t epoch: {ep}')

    save(paths, params, scope_text, scope_elmo_emb, id2idx_dict, idx2id_dict, mesh_graph, best_model)

def save(paths, params, scope_text, scope_elmo_emb, id2idx, idx2id, mesh_graph, model):
    save_path = paths.experiment_folder

    with open(os.path.join(save_path, 'dump.pkl'),'wb+') as f:
        pickle.dump([scope_text, scope_elmo_emb, id2idx, idx2id, params, mesh_graph], f)

    torch.save(model, os.path.join(save_path,'EL_GCN_model.pkl'))

def load(paths):
    load_path = paths.experiment_folder

    with open(os.path.join(load_path, 'dump.pkl'),'rb+') as f:
        [scope_text, scope_elmo_emb, id2idx, idx2id, params, mesh_graph] = pickle.load(f)

    model = torch.load(os.path.join(load_path,'EL_GCN_model.pkl'))

    return [scope_text, scope_elmo_emb, id2idx, idx2id, params, mesh_graph, model]


def test(paths, params, X_test, annotated_docs_test, predictions_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    [scope_text, scope_elmo_emb, id2idx_dict, idx2id_dict, params, mesh_graph, model] = load(paths)
    options_file = paths.elmo_options
    weight_file = paths.elmo_weights
    elmo = Elmo(options_file, weight_file, 2,dropout=0)
    elmo.to(device)aaj din chadheya full song

    # delete------------
    unique_id = get_unique_id(paths.file1, paths.file2, paths.ctd_file, paths.c2m_file)

    node_list = list(idx2id_dict.values())
    # check if adjacancy matrix already calculated
    if not os.path.exists(os.path.join(paths.MeSH_folder, 'a_hat_matrix')):
        a_hat = get_adjacancy_matrix(mesh_graph, node_list)

        # save node_list and the calculated adjacancy matrix
        with open(os.path.join(paths.MeSH_folder, 'node_list'), 'wb') as f:
            pickle.dump(node_list, f)
        data = sparse.coo_matrix(a_hat)
        with open(os.path.join(paths.MeSH_folder, 'a_hat_matrix'), 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(os.path.join(paths.MeSH_folder, 'a_hat_matrix'), 'rb') as f:
            data = pickle.load(f)

    i = torch.tensor([data.row, data.col], dtype=torch.long, device=device)
    v = torch.tensor(data.data, dtype=torch.float32, device=device)
    a_hat = torch.sparse.FloatTensor(i,v, torch.Size([len(node_list), len(node_list)])).cuda()

    x_test_data, _ = construct_data(X_test, annotated_docs_test, predictions_test, scope_text, id2idx_dict, paths.ctd_file, paths.c2m_file, use_ELMO=params.use_elmo, elmo_model=elmo, elmo_dim=params.elmo_dim, device=device)

    model.eval()
    with torch.no_grad():
        pred_index, pred_label, labels, rank, reciprocal_rank = [], [], [], [], []
        for t, x, y, mask in minibatch_val(x_test_data, batch_size=params.batch_size):
            x, mask = x.to(device), mask.to(device).float()

            mask = mask.unsqueeze(2).expand(-1,-1,x.shape[2])
            x = mask*x
            x = torch.mean(x, dim=1)
            x = model(x, a_hat)

            x = nn.functional.softmax(x, dim=1)
            _, max_idx = torch.max(x, dim=1)
            labels.extend(y)
            _= [pred_label.append(idx2id_dict[i.item()]) for i in max_idx]

            _, index_sorted = torch.sort(x, descending=True)
            pred_index.extend(index_sorted)
            
        sorted_list = []
        zeroshot_rank, zeroshot_rrank = [], []
        pred_2, pred_5, pred_10, pred_15, pred_30 = [], [], [], [], []
        for idx, item in enumerate(pred_index):
            id_sorted = [idx2id_dict[i.item()] for i in item]
            sorted_list.append(id_sorted)

            if labels[idx] in id_sorted:
                rank.append(id_sorted.index(labels[idx])+1)
                reciprocal_rank.append(1/(id_sorted.index(labels[idx])+1))

                if labels[idx] in unique_id:
                    zeroshot_rank.append(id_sorted.index(labels[idx])+1)
                    zeroshot_rrank.append(1/(id_sorted.index(labels[idx])+1))
            else:
                print(f"ID {labels[idx]} not found")
            
            if labels[idx] in id_sorted[0:2]:
                pred_2.append(labels[idx])
            else:
                pred_2.append(id_sorted[0])
            if labels[idx] in id_sorted[0:5]:
                pred_5.append(labels[idx])
            else:
                pred_5.append(id_sorted[0])
            if labels[idx] in id_sorted[0:10]:
                pred_10.append(labels[idx])
            else:
                pred_10.append(id_sorted[0])
            if labels[idx] in id_sorted[0:15]:
                pred_15.append(labels[idx])
            else:
                pred_15.append(id_sorted[0])
            if labels[idx] in id_sorted[0:30]:
                pred_30.append(labels[idx])
            else:
                pred_30.append(id_sorted[0])

        print(classification_report(labels, pred_label))
        print(f'Mean Reciprocal Rank: {np.mean(reciprocal_rank)}')
        acc = accuracy_score(labels, pred_label)
        print(f'Test Acc@1: {acc}')
        acc = accuracy_score(labels, pred_2)
        print(f'Test Acc@2: {acc}')
        acc = accuracy_score(labels, pred_5)
        print(f'Test Acc@5: {acc}')
        acc = accuracy_score(labels, pred_10)
        print(f'Test Acc@10: {acc}')
        acc = accuracy_score(labels, pred_15)
        print(f'Test Acc@15: {acc}')
        acc = accuracy_score(labels, pred_30)
        print(f'Test Acc@30: {acc}')

        print(f'Zero shot MRR: {np.mean(zeroshot_rrank)}')


def main(paths, params):
    path_to_train_input = paths.training
    path_to_valid_input = paths.develop
    path_to_test= paths.test

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    X = BratInput(path_to_train_input)
    X = X.transform()
    X = split_annotated_documents(X)

    X_valid = BratInput(path_to_valid_input)
    X_valid = X_valid.transform()
    X_valid = split_annotated_documents(X_valid)

    if not params.randomize:
        torch.manual_seed(5)
        random.seed(5)
        np.random.seed(5)

    # Obtain MeSH information
    mesh_file = paths.MeSH_file
    disease_file= paths.disease_file
    mesh_graph_file = paths.MeSH_graph_disease

    # read disease file
    with open(disease_file,'r') as f:
        data = f.readlines()

    mesh_dict = read_mesh_file(mesh_file)

    mesh_graph = nx.read_gpickle(mesh_graph_file)
    mesh_graph = mesh_graph.to_undirected()

    # Construct usable data format
    x_text = annotated_docs_to_tokens(X)
    scope_text, id2idx_dict, idx2id_dict = mesh_dict_to_tokens(mesh_dict, data)

    
    _, predictions_tr = transform_annotated_documents_to_bio_format(X)
    annotated_docs_tr = X
    _, predictions_v = transform_annotated_documents_to_bio_format(X_valid)
    annotated_docs_v = X_valid

    # annotated_docs_tr, predictions_tr = get_NER_prediction(X)
    # annotated_docs_v, predictions_v = get_NER_prediction(X_valid)

    # training

    train_gcn(paths, params, X, annotated_docs_tr, predictions_tr, X_valid, annotated_docs_v, predictions_v,
                 scope_text, id2idx_dict, idx2id_dict, mesh_graph, device=device)


# if __name__ == "__main__":
#     # Obtain the training, validation and test dataset
#     path_to_train_input = r'/media/druv022/Data1/Masters/Thesis/Data/Converted_train_2'
#     path_to_valid_input = r'/media/druv022/Data1/Masters/Thesis/Data/Converted_develop'
#     path_to_test= r'/media/druv022/Data1/Masters/Thesis/Data/Converted_test'
#     path_to_embeddings = r'/media/druv022/Data1/Masters/Thesis/Data/Embeddings'
#     ctd_file = r'/media/druv022/Data1/Masters/Thesis/Data/CTD/CTD_diseases.csv'
#     c2m_file = r'/media/druv022/Data1/Masters/Thesis/Data/C2M/C2M_mesh.txt'

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     writer = SummaryWriter()

#     X = BratInput(path_to_train_input)
#     X = X.transform()
#     # X = split_annotated_documents(X)

#     X_valid = BratInput(path_to_valid_input)
#     X_valid = X_valid.transform()
#     # X_valid = split_annotated_documents(X_valid)

#     X_test = BratInput(path_to_test)
#     X_test = X_test.transform()

#     torch.manual_seed(5)
#     random.seed(5)
#     np.random.seed(5)

#     entity_names = ['B_Disease','I_Disease']
#     embeddings =  load_embedding_pkl(path_to_embeddings)

#     # Obtain MeSH information
#     mesh_file = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/ASCIImeshd2019.bin'
#     disease_file= r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/disease_list'
#     mesh_graph_file = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/mesh_graph_disease'
#     mesh_folder = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH'

#     # read disease file
#     with open(disease_file,'r') as f:
#         data = f.readlines()

#     mesh_dict = read_mesh_file(mesh_file)

#     mesh_graph = nx.read_gpickle(mesh_graph_file)
#     mesh_graph = mesh_graph.to_undirected()

#     # Construct usable data format
#     x_text = annotated_docs_to_tokens(X)
#     scope_text, id2idx_dict, idx2id_dict = mesh_dict_to_tokens(mesh_dict, data)

#     node_list = list(idx2id_dict.values())

#     _, predictions_tr = transform_annotated_documents_to_bio_format(X)
#     annotated_docs_tr = X
#     _, predictions_v = transform_annotated_documents_to_bio_format(X_valid)
#     annotated_docs_v = X_valid

#     # annotated_docs_tr, predictions_tr = get_NER_prediction(X)
#     # annotated_docs_v, predictions_v = get_NER_prediction(X_valid)

#     # training
#     epochs=500
#     batch_size=32
#     n_samples = 4

#     train_gcn()
