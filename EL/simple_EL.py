from nerds.input.brat import BratInput
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from utils.embeddings import load_embedding_pkl, filter_embeddings
from tensorboardX import SummaryWriter
from utils.mesh import read_mesh_file
from utils.vocab import Vocabulary
from nerds.util.nlp import text_to_tokens, text_to_sentences
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from nerds.ner.pytorch_druv import PytorchNERModel
from utils.padding import pad_nested_sequences
from nerds.doc.bio import transform_annotated_document_to_bio_format
import networkx as nx
import math
from sklearn.utils import shuffle as Shuffle_lists
from sklearn.metrics import classification_report, accuracy_score
from time import time
import os
from allennlp.modules.elmo import Elmo, batch_to_ids
from utils.convert2D import Convert2D
from models.EL_models import EntityModel
from EL.EL_utils import *


class IndexTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, word_vocab, char_vocab, label_vocab,
                 use_char=True):
        self._use_char = use_char
        self._word_vocab = word_vocab
        self._char_vocab = char_vocab
        self._label_vocab = label_vocab

    def transform(self, X):
        """Transform documents to document ids.

        Uses the vocabulary learned by fit.

        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.
            y : iterabl, label strings.

        Returns:
            features: document id matrix.
            y: label id matrix.
        """
        word_ids = torch.tensor(self._word_vocab.doc2id(X), dtype=torch.long)

        if self._use_char:
            char_ids = [[self._char_vocab.doc2id(w) for w in X]]
            char_ids = pad_nested_sequences(char_ids)
            char_ids = torch.tensor(char_ids, dtype=torch.long)
            features = [word_ids, char_ids.squeeze(dim=0)]
        else:
            features = [word_ids]

        return features

    # used for minibatch method
    def transform2(self, X, y=None):
        word_ids = [torch.tensor(self._word_vocab.doc2id(doc), dtype=torch.double) for doc in X ]
        # word_idx = sorted(range(len(word_ids)),key=lambda idx: len(word_ids[idx]), reverse=True)
        # word_ids = [word_ids[i] for i in word_idx]        
        word_ids = nn.utils.rnn.pad_sequence(word_ids, batch_first=True)

        if self._use_char:
            char_ids = [[self._char_vocab.doc2id(w) for w in doc] for doc in X] 
            char_ids = pad_nested_sequences(char_ids)
            char_ids = torch.tensor(char_ids)
            features = [word_ids, char_ids]
        else:
            features = word_ids

        return features


# def get_normalizations(o_doc, pred_doc):

#     entity_list = []
#     # find the corresponding normalized concept in true annotated docs
#     for annotation in pred_doc.annotations:
#         # max predicted annotations
#         max_ann_length = len(o_doc.annotations)
#         ann_counter = 0
#         # iterate in original annotations
#         complete_flag = False
#         while ann_counter < max_ann_length:
#             # find the matching annotation text and offset in the predicted annotations
#             # TODO: properly fix inverted comma issue in the whole NERDS
#             if '"' in annotation.text:
#                 flag = o_doc.annotations[ann_counter].text.replace(' ','') == annotation.text.replace('"','').replace(' ','')
#             else:
#                 flag = False
#             if (annotation.text == o_doc.annotations[ann_counter].text or flag) and annotation.offset == o_doc.annotations[ann_counter].offset:
#                 ann_identifier = o_doc.annotations[ann_counter].identifier
#                 o_doc.annotations.remove(o_doc.annotations[ann_counter])
#                 for norm in o_doc.normalizations:
#                     if norm.argument_id == ann_identifier:
#                         entity_list.append(norm.preferred_term.strip())
#                         complete_flag = True
#                         break
#             if complete_flag:
#                 break
#             ann_counter += 1
        
#         if not complete_flag:
#             entity_list.append('<unk>')

#     return entity_list

def get_NER_prediction(data, model_path=None):
    ner_model = PytorchNERModel()
    ner_model.load('tmp','trained_model_20190509-021945.pkl')

    annotated_docs, predictions, _ = ner_model.predict_for_EL(data)
    del ner_model
    return annotated_docs, predictions


def inject_negative(tr_data, scope_note, id_dict, mesh_graph, use_elmo=False, elmo_model = None, elmo_dim=1024, device=torch.device('cpu')):
    new_data = []
    count = 0

    for item in tr_data:
        t, s_note, entity_id, mask, y = item
        s_note = torch.mean(s_note, dim=0)

        if entity_id in mesh_graph.nodes():
            neg_note = []
            neg_id = []
            for i in mesh_graph.neighbors(entity_id):
                # text, scope_note, Mesh_ID, Mask, positive_lable
                note = []
                if use_elmo:
                    note = scope_note[id_dict[i]]
                    note = batch_to_ids(note).to(device)
                    with torch.no_grad():
                        elmo_emb = elmo_model(note)
                    note = elmo_emb['elmo_representations'][0].view(-1, elmo_dim).detach().cpu()
                    note = torch.mean(note, dim=0)
                else:
                    _ = [note.extend(line[1:-1]) for line in scope_note[id_dict[i]] if len(line) > 1]
                neg_note.append(note)
                neg_id.append(i)
        else:
            count +=1
        sample = (t, (s_note, entity_id), (neg_note, neg_id), mask)
        new_data.append(sample)

    print('Number of MeSH ID not in graph: ', count)
    return new_data

def loss_fuction(x, z, y, mask, n_samples=1, margin=0.5, device='cpu'):
    if mask.shape[1] != x.shape[1]:
        mask = mask[:,0:x.shape[1]]
    mask = mask.unsqueeze(2).expand(-1,-1,x.shape[2]).float()
    x = torch.mean(mask*x,dim=1)

    cos_sim = nn.functional.cosine_similarity(x, z)

    y = y.float()
    loss_pos = y*(1.0/n_samples)*(1-cos_sim)**2

    another_mask = torch.where(cos_sim < margin, torch.tensor(1, device=device), torch.tensor(0, device=device)).float()
    loss_neg = another_mask * (1.0 - y)*cos_sim**2

    loss =  loss_pos + loss_neg

    return torch.sum(loss)
 
 
def minibatch_mesh(scope_note, transform=None, batch_size=1, use_elmo=False):
    s_note = []

    if not use_elmo:
        for item in scope_note:
            note = []
            _ = [note.extend(i[1:-1]) for i in item]
            s_note.append(note)
    else:
        s_note = scope_note

    length_x = len(s_note)

    for idx in range(0, math.ceil(length_x/batch_size)):
        if length_x - batch_size < 0:
            updated_size = length_x
        else:
            updated_size = batch_size
        length_x = length_x - batch_size
        batch_s = s_note[idx*batch_size : idx*batch_size + updated_size]
        if transform:
            yield transform(batch_s)
        else:
            batch_s = nn.utils.rnn.pad_sequence(batch_s, batch_first=True)
            
            yield batch_s

def find_id(mesh_emb, text_emb):
    pred_label_idx = []
    for item in text_emb:
        item = item.unsqueeze(0).repeat(mesh_emb.shape[0],1)

        cos_sim = nn.functional.cosine_similarity(item, mesh_emb)
        _, max_idx = torch.max(cos_sim, dim=0)
        pred_label_idx.append(max_idx)

    return pred_label_idx



# if __name__ == '__main__':

    # Obtain the training, validation and test dataset
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
    # scope_text, id_dict, idx2id_dict = mesh_dict_to_tokens(mesh_dict, data)

    # text = copy.deepcopy(x_text)
    # text.extend(scope_text)
    # word_vocab, char_vocab = get_text_vocab(text)

    # id_vocab = Vocabulary(lower=False)
    # _ = [id_vocab.add_token(id) for id in id_dict.keys()]
    # id_vocab.build()

    # word_embedding_size = 200
    # embeddings_ = filter_embeddings(embeddings, word_vocab.vocab, word_embedding_size)
    # embeddings_ = torch.tensor(embeddings_)

    # annotated_docs_tr, predictions_tr = get_NER_prediction(X)
    # annotated_docs_v, predictions_v = get_NER_prediction(X_valid)

    # train_data = construct_data(X,annotated_docs_tr, predictions_tr, scope_text, id_dict, ctd_file, c2m_file)
    # train_data_ = inject_negative(train_data, scope_text, id_dict, mesh_graph)

    # valid_data = construct_data(X_valid,annotated_docs_v, predictions_v, scope_text, id_dict, ctd_file, c2m_file)

    # # Define a BiLSTM model
    # model = EntityModel(len(word_vocab.vocab), len(char_vocab.vocab), embeddings_, device=device, dropout=0.0)
    # model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # n_samples = 8
    # epochs = 30
    # batch_size = 64
    # use_char = True

    # p = IndexTransformer(word_vocab, char_vocab, id_vocab)
    # train_acc = 0.0
    # val_acc = 0.0
    # for epoch in range(epochs):
    #     np.random.shuffle(train_data_)
    #     training_loss = 0
    #     count = 0
    #     start_time = time()
    #     for x, z, y, mask in minibatch(train_data_, scope_text, p.transform2, n_samples=n_samples, batch_size=batch_size):
    #         if use_char:
    #             x_0, x_1, z_0, z_1, y, mask = x[0].to(device), x[1].to(device), z[0].to(device), z[1].to(device), y.to(device), mask.to(device)
    #             x, z = (x_0, x_1), (z_0, z_1)
            
    #         optimizer.zero_grad()

    #         x = model(x)
    #         z = model(z)

    #         loss = loss_fuction(x, z, y, mask, n_samples=n_samples, margin=0.5, device=device)
    #         training_loss += loss.item()

    #         loss.backward()
    #         optimizer.step()

    #         count += 1

    #     print(f'Epoch: {epoch}\tloss: {training_loss/count}\tTime: {time() - start_time}')
    #     start_time = time()
    #     with torch.no_grad():
    #         m_emb = []
    #         for s in minibatch_mesh(scope_text, p.transform2, batch_size=batch_size):
    #             s_0, s_1 = s[0].to(device), s[1].to(device)

    #             s = model([s_0, s_1])
    #             s = torch.sum(s, dim=1)
    #             m_emb.extend(s)
    #         m_emb = torch.stack(m_emb, dim=0)

    #         text_emb, label = [], []
    #         for x, y, mask in minibatch_val(train_data, p.transform2, batch_size=batch_size):
    #             x_0, x_1, mask = x[0].to(device), x[1].to(device), mask.to(device).float()

    #             x = model([x_0, x_1])

    #             if mask.shape[1] != x.shape[1]:
    #                 mask = mask[:,0:x.shape[1]]
    #             mask = mask.unsqueeze(2).expand(-1,-1,x.shape[2])
    #             x = mask * x
    #             x = torch.sum(x, dim=1)
    #             text_emb.extend(x)
    #             label.extend(y)
            
    #         pred_idx = find_id(m_emb, text_emb)
    #         pred_label = [idx2id_dict[i.item()] for i in pred_idx]

    #         print(classification_report(label, pred_label))
    #         acc = accuracy_score(label, pred_label)
    #         print(f'Train Acc: {acc}\tTime: {time()-start_time}')
            
    #         if acc > train_acc:
    #             train_acc = acc
            
    #         print('Best train acc: ', train_acc)

    #         text_emb, label = [], []
    #         for x, y, mask in minibatch_val(valid_data, transform=p.transform2, batch_size=batch_size):
    #             x_0, x_1, mask = x[0].to(device), x[1].to(device), mask.to(device).float()

    #             x = model([x_0, x_1])

    #             if mask.shape[1] != x.shape[1]:
    #                 mask = mask[:,0:x.shape[1]]
    #             mask = mask.unsqueeze(2).expand(-1,-1,x.shape[2])
    #             x = mask * x
    #             x = torch.sum(x, dim=1)
    #             text_emb.extend(x)
    #             label.extend(y)
            
    #         pred_idx = find_id(m_emb, text_emb)
    #         pred_label = [idx2id_dict[i.item()] for i in pred_idx]

    #         print(classification_report(label, pred_label))
    #         acc = accuracy_score(label, pred_label)
    #         print(f'Valid Acc: {acc}\tTime: {time()-start_time}')
            
    #         if acc > val_acc:
    #             print('Saving model for acc: ', val_acc)
    #             torch.save(model, 'simple_entity_model.pkl')
    #             val_acc = acc
            
    #         print('Best val acc: ', val_acc)


    # print('Here')

    # # train and validate

    
