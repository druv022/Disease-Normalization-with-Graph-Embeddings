from EL.EL_utils import *
from allennlp.modules.elmo import Elmo, batch_to_ids
import pickle
import os
import torch.nn as nn
import torch
from nerds.doc.bio import transform_annotated_documents_to_bio_format
from nerds.dataset.split import split_annotated_documents
from models.EL_models import EL_model, EL_similarity
from sklearn.metrics import classification_report, accuracy_score
from nerds.input.brat import BratInput
from time import time
import random
from utils.vocab import Vocabulary
import copy
from tensorboardX import SummaryWriter
from collections import Counter
from analysis.get_analysis import get_unique_id

def get_char_dict(mesh_dict, char_vocab):
    char_dict = {}
    for key in mesh_dict:
        value = mesh_dict[key]
        names = []
        for i in value.entry_terms:
            char_names = []
            _ = [char_names.extend(char_vocab.doc2id(j)) for j in i]
            names.append(char_names)

        char_dict[key] = names

    return char_dict

def train(paths, params, X, mesh_dict, scope_text, id2idx_dict, idx2id_dict, predictions_tr, annotated_docs_tr, X_valid, predictions_v, annotated_docs_v, writer=None ,device=torch.device('cpu')):
    options_file = paths.elmo_options
    weight_file = paths.elmo_weights
    ELMO_folder = paths.elmo_folder
    elmo_dim = params.elmo_dim
    elmo = Elmo(options_file, weight_file, 2,dropout=0)
    elmo.to(device)

    # re-encode nodes
    scope_elmo_emb, _ = get_scope_elmo(elmo, ELMO_folder, scope_text, elmo_dim, idx2id_dict, id2idx_dict, device=device)
    scope_elmo_emb = re_encode(paths.embedding, scope_elmo_emb, idx2id_dict, device=device)

    # format trainable data
    x_data, texts = construct_data(X,annotated_docs_tr, predictions_tr, scope_text, id2idx_dict, paths.ctd_file, paths.c2m_file, use_ELMO=params.use_elmo, elmo_model=elmo, elmo_dim=params.elmo_dim, device=device)
        
    x_v_data, _ = construct_data(X_valid,annotated_docs_v, predictions_v, scope_text, id2idx_dict, paths.ctd_file, paths.c2m_file, use_ELMO=params.use_elmo, elmo_model=elmo, elmo_dim=params.elmo_dim, device=device)
        

    word_vocab, char_vocab = get_text_vocab([texts])
    char_dict= get_char_dict(mesh_dict, char_vocab)

    params_dict1 = {'batch_size':params.batch_size,'shuffle': True, 'num_workers': 12
    }

    EL_dataset = EL_sequence(x_data, scope_elmo_emb, id2idx_dict, copy.deepcopy(char_vocab))
    training_generator = DataLoader(EL_dataset, collate_fn=padding_EL, **params_dict1)
    
    model = EL_model(elmo_dim, elmo_dim, scope_elmo_emb)
    model.to(device)

    # sim_model = EL_similarity(200, len(char_vocab))
    # sim_model.to(device)

    optimizer1 = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=0.0)
    # optimizer2 = torch.optim.Adam(sim_model.parameters(), lr=params.lr, weight_decay=0.0)

    epochs = params.num_epochs
    batch_size = params.batch_size
    train_acc = 0
    val_acc = 0
    best_model1 = None
    best_model2 = None
    ep = 0
    with open(os.path.join(params.output), 'w') as f:
        for epoch in range(epochs):
            np.random.shuffle(x_data)
            x_data_ =  x_data

            training_loss1 = 0
            # training_loss2 = 0
            count = 0
            start_time= time()
            model.train()
            # sim_model.train()

            tr_labels, tr_mentions, pred_index,  = [], [], []
            for t, x, _, y, mask in minibatch(x_data_, scope_elmo_emb, char_vocab=char_vocab, id2idx=id2idx_dict, use_elmo=True, elmo_model=elmo, n_samples=0, batch_size=batch_size, device=device):
            # for t, x, _, y, mask in training_generator:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                optimizer1.zero_grad()

                mask_ = mask.unsqueeze(2).expand(-1,-1,x.shape[2])
                x = mask_*x
                x = torch.mean(x, dim=1)
                x = model(x)

                loss = nn.functional.cross_entropy(x, y)
                
                loss.backward()
                optimizer1.step()
                training_loss1 += loss.item()
                count += 1

                z = nn.functional.softmax(x, dim=1)
                _, index_sorted = torch.sort(z, descending=True)

                tr_labels.extend(y)
                pred_index.extend(index_sorted)
                tr_mentions.extend(t)

            writer.add_scalars('training',{'loss1':training_loss1/count},global_step=epoch)
            print(f'Epoch: {epoch}\tLoss1: {training_loss1/count}\tTime: {time()-start_time}')
            f.writelines(f'Epoch: {epoch}\tLoss1: {training_loss1/count}\tTime: {time()-start_time}\n')
            
            start_time = time()
            with torch.no_grad():
                model.eval()
                pred_label, label = [], []
                for t, x, y, mask in minibatch_val(x_data, batch_size=batch_size):
                    x, mask = x.to(device), mask.to(device).float()

                    mask = mask.unsqueeze(2).expand(-1,-1,x.shape[2])
                    x = mask*x
                    x = torch.mean(x, dim=1)
                    x = model(x)

                    x = nn.functional.softmax(x, dim=1)
                    _, max_idx = torch.max(x, dim=1)
                    label.extend(y)
                    _= [pred_label.append(idx2id_dict[i.item()]) for i in max_idx]
                
                # print(classification_report(label, pred_label))
                acc = accuracy_score(label, pred_label)
                print(f'Train Acc: {acc}\tTime: {time()-start_time}')
                # f.writelines(classification_report(label, pred_label))
                f.writelines(f'\nTrain Acc: {acc}\tTime: {time()-start_time}\n')
                
                if acc > train_acc:
                    train_acc = acc
                
                print('Best train acc: ', train_acc)
                f.writelines(f'Best train acc: {train_acc}\n')

                pred_label, label = [], []
                for t, x, y, mask in minibatch_val(x_v_data, batch_size=batch_size):
                    x, mask = x.to(device), mask.to(device).float()

                    mask = mask.unsqueeze(2).expand(-1,-1,x.shape[2])
                    x = mask*x
                    x = torch.mean(x, dim=1)
                    x = model(x)

                    x = nn.functional.softmax(x, dim=1)
                    _, max_idx = torch.max(x, dim=1)
                    label.extend(y)
                    _= [pred_label.append(idx2id_dict[i.item()]) for i in max_idx]

                # print(classification_report(label, pred_label))
                acc = accuracy_score(label, pred_label)
                print(f'Valid Acc: {acc}\tTime: {time()-start_time}')
                # f.writelines(print(classification_report(label, pred_label)))
                f.writelines(f'\nValid Acc: {acc}\tTime: {time()-start_time}, epoch: {epoch}\n')
                
                if acc > val_acc:
                    val_acc = acc
                    print('Updating best model for acc: ', acc)
                    f.writelines(f'Updating model at epoch: {epoch}\n')
                    best_model1 = model
                    # best_model2 = sim_model
                    ep = epoch
                
                print(f'Best val acc: {val_acc}, epoch: {ep}')
                f.writelines(f'Best val acc: {val_acc}, epoch: {ep}\n')

    save(paths, params, scope_text, scope_elmo_emb, id2idx_dict, idx2id_dict, char_vocab, char_dict, best_model1, best_model2)

def save(paths, params, scope_text, scope_elmo_emb, id2idx, idx2id, char_vocab, char_dict, model1, model2):
    save_path = paths.experiment_folder

    with open(os.path.join(save_path, 'dump.pkl'),'wb+') as f:
        pickle.dump([scope_text, scope_elmo_emb, id2idx, idx2id, char_vocab, char_dict, params], f)

    torch.save(model1, os.path.join(save_path,'EL_model.pkl'))
    # torch.save(model2, os.path.join(save_path,'EL_model_sim.pkl'))

def load(paths):
    load_path = paths.experiment_folder

    with open(os.path.join(load_path, 'dump.pkl'),'rb+') as f:
        [scope_text, scope_elmo_emb, id2idx, idx2id, char_vocab, char_dict, params] = pickle.load(f)

    model1 = torch.load(os.path.join(load_path,'EL_model.pkl'))
    model2 = None #torch.load(os.path.join(load_path,'EL_model_sim.pkl'))

    return [scope_text, scope_elmo_emb, id2idx, idx2id, char_vocab, char_dict, params, model1, model2]

def test(paths, params, X_test, annotated_docs_test, predictions_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    [scope_text, scope_elmo_emb, id2idx_dict, idx2id_dict, char_vocab, char_dict, old_params, model1, model2] = load(paths)
    options_file = paths.elmo_options
    weight_file = paths.elmo_weights
    elmo = Elmo(options_file, weight_file, 2,dropout=0)
    elmo.to(device)

    unique_id = get_unique_id(paths.file1, paths.file2, paths.ctd_file, paths.c2m_file)
    x_test_data, _ = construct_data(X_test,annotated_docs_test, predictions_test, scope_text, id2idx_dict, paths.ctd_file, paths.c2m_file, use_ELMO=params.use_elmo, elmo_model=elmo, elmo_dim=params.elmo_dim, device=device)

    
    model1.eval()
    # model2.eval()
    with torch.no_grad():
        pred_index, pred_label, labels, rank, reciprocal_rank, t_mentions =[], [], [], [], [], []
        for t, x, y, mask in minibatch_val(x_test_data, char_vocab=char_vocab, batch_size=params.batch_size):
            x, mask = x.to(device), mask.to(device).float()

            mask = mask.unsqueeze(2).expand(-1,-1,x.shape[2])
            x = mask*x
            x = torch.mean(x, dim=1)
            x = model1(x)

            x = nn.functional.softmax(x, dim=1)
            _, max_idx = torch.max(x, dim=1)
            labels.extend(y)
            _= [pred_label.append(idx2id_dict[i.item()]) for i in max_idx]

            _, index_sorted = torch.sort(x, descending=True)
            pred_index.extend(index_sorted)
            t_mentions.extend(t)
            
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
    writer = SummaryWriter()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = BratInput(path_to_train_input)
    X = X.transform()
    X = split_annotated_documents(X)

    X_valid = BratInput(path_to_valid_input)
    X_valid = X_valid.transform()
    X_valid = split_annotated_documents(X_valid)

    X_test = BratInput(path_to_test)
    X_test = X_test.transform()
    X_test = split_annotated_documents(X_test)


    if params.randomize:
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

    # train with gold set and without NER
    _, predictions_tr = transform_annotated_documents_to_bio_format(X)
    annotated_docs_tr = X
    _, predictions_v = transform_annotated_documents_to_bio_format(X_valid)
    annotated_docs_v = X_valid

    _, predictions_test = transform_annotated_documents_to_bio_format(X_test)
    annotated_docs_test = X_test

    # training with NER
    # annotated_docs_tr, predictions_tr = get_NER_prediction(X)
    # annotated_docs_v, predictions_v = get_NER_prediction(X_valid)

    train(paths, params,X, mesh_dict, scope_text, id2idx_dict, idx2id_dict, predictions_tr, annotated_docs_tr, X_valid, predictions_v, annotated_docs_v, writer=writer, device=device)


# if __name__ == "__main__":
    # # Obtain the training, validation and test dataset
    # path_to_train_input = r'/media/druv022/Data1/Masters/Thesis/Data/Converted_train_2'
    # path_to_valid_input = r'/media/druv022/Data1/Masters/Thesis/Data/Converted_develop'
    # path_to_test= r'/media/druv022/Data1/Masters/Thesis/Data/Converted_test'
    # path_to_embeddings = r'/media/druv022/Data1/Masters/Thesis/Data/Embeddings'
    # ctd_file = r'/media/druv022/Data1/Masters/Thesis/Data/CTD/CTD_diseases.csv'
    # c2m_file = r'/media/druv022/Data1/Masters/Thesis/Data/C2M/C2M_mesh.txt'

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # writer = SummaryWriter()

    # X = BratInput(path_to_train_input)
    # X = X.transform()
    # # X = split_annotated_documents(X)

    # X_valid = BratInput(path_to_valid_input)
    # X_valid = X_valid.transform()
    # # X_valid = split_annotated_documents(X_valid)

    # X_test = BratInput(path_to_test)
    # X_test = X_test.transform()

    # torch.manual_seed(5)
    # random.seed(5)
    # np.random.seed(5)

    # entity_names = ['B_Disease','I_Disease']
    # embeddings =  load_embedding_pkl(path_to_embeddings)

    # # Obtain MeSH information
    # mesh_file = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/ASCIImeshd2019.bin'
    # disease_file= r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/disease_list'
    # mesh_graph_file = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/mesh_graph_disease'

    # # read disease file
    # with open(disease_file,'r') as f:
    #     data = f.readlines()

    # mesh_dict = read_mesh_file(mesh_file)

    # mesh_graph = nx.read_gpickle(mesh_graph_file)
    # mesh_graph = mesh_graph.to_undirected()

    # # Construct usable data format
    # x_text = annotated_docs_to_tokens(X)
    # scope_text, id2idx_dict, idx2id_dict = mesh_dict_to_tokens(mesh_dict, data)

    # # train with gold set and without NER
    # _, predictions_tr = transform_annotated_documents_to_bio_format(X)
    # annotated_docs_tr = X
    # _, predictions_v = transform_annotated_documents_to_bio_format(X_valid)
    # annotated_docs_v = X_valid

    # # training with NER
    # # annotated_docs_tr, predictions_tr = get_NER_prediction(X)
    # # annotated_docs_v, predictions_v = get_NER_prediction(X_valid)

    # # training
    # epochs=100
    # batch_size=32
    # n_samples = 4
    # use_neg = False

    # train()
    # # train_word2vec()

    # print('Here')