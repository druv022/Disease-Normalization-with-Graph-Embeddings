import torch
from tensorboardX import SummaryWriter
from nerds.input.brat import BratInput
import torch.nn as nn
import random
from utils.embeddings import load_embedding_pkl, filter_embeddings
import numpy as np
from utils.mesh import read_mesh_file
import networkx as nx
from nerds.util.nlp import text_to_tokens, text_to_sentences
import copy
from utils.vocab import Vocabulary
from allennlp.modules.elmo import Elmo, batch_to_ids
import pickle
import os
from models.EL_models import BiLSTM, EL_GCN
from models.NER_model import BiLSTMCRF, NERCRF
from nerds.doc.bio import transform_annotated_document_to_bio_format, transform_bio_tags_to_annotated_documents
from sklearn.metrics import classification_report as cls_rpt, accuracy_score as acc_scr
from seqeval.metrics import f1_score, classification_report, accuracy_score
from nerds.evaluate.score import annotation_precision_recall_f1score
from torch.utils.data import Dataset, DataLoader
from nerds.dataset.split import split_annotated_documents
from utils.convert2D import Convert2D
import math
from sklearn.utils import shuffle as Shuffle_lists
from torchsparseattn import Fusedmax, Oscarmax, Sparsemax
from config.paths import *
from config.params import *
from scipy import sparse
from EL.EL_utils import get_scope_elmo, get_adjacancy_matrix, mesh_dict_to_tokens, get_text_vocab

def annotated_docs_to_tokens(docs, sentence_pad=False):
    text_list = []
    label_list = []
    tokens_list = []
    for i,doc in enumerate(docs):
        if sentence_pad:
            text = [[r'<s>']+ text_to_tokens(sent)+[r'<\s>'] for sent in text_to_sentences(doc.plain_text_)[0] if len(sent.split()) > 0]
        else:
            text = [text_to_tokens(sent) for sent in text_to_sentences(doc.plain_text_)[0] if len(sent.split()) > 0]

        text_list.append(text)

        count = 0
        pad_index = []
        for line in text:
            for idx,word in enumerate(line):
                if word==r'<s>' or word==r'<\s>':
                    pad_index.append(count+idx)
            count+= len(line)

        tokens, labels = transform_annotated_document_to_bio_format(doc)

        count = 0
        for i,line in enumerate(text_list[-1]):
            start_count = 0
            for j,word in enumerate(line):
                if word not in [r'<s>',r'<\s>'] and word != tokens[count]:
                    k=0
                    start_count = count
                    if tokens[count] in word:
                        text_list[-1][i][j] = tokens[count+k]
                        k += 1
                    while count+k < len(tokens) and tokens[count+k] in word:
                        text_list[-1][i].insert(j+k,tokens[count+k])
                        k += 1
                    # print(f'Error: split text= {word}, token{tokens[start_count:count+k]}')
                    count += 1
                elif word not in [r'<s>',r'<\s>']:
                    count += 1

        [labels.insert(i, 'O') for i in pad_index]
        [tokens.insert(i,r'<s>') for i in pad_index]

        label_list.append(labels)
        tokens_list.append(tokens)

    return text_list, label_list, tokens_list

# copied from Index transform
def inverse_transform(y, label_vocab, lengths=None):
    # y = np.argmax(y, -1)
    inverse_y = [label_vocab.id2doc(ids) for ids in y]
    if lengths is not None:
        inverse_y = [iy[:l] for iy, l in zip(inverse_y, lengths)]

    return inverse_y

class Sequence(Dataset):

    def __init__(self,docs, text_emb, ner_labels, preprocess):
        self.docs = docs
        self.text_emb = text_emb
        self.ner_labels = ner_labels
        self.preprocess = preprocess

    def __getitem__(self, idx):
        text_emb = self.text_emb[idx]
        ner_tag = torch.tensor(self.preprocess(self.ner_labels[idx]), dtype=torch.long)
        doc = self.docs[idx]

        return text_emb, ner_tag, doc

    def __len__(self):
        return len(self.text_emb)


def pad_tensor(vec, pad, dim, dtype, device=torch.device('cpu')):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size, dtype=dtype, device=device)], dim=dim)

class PadCollate:
    
    def __init__(self, batch_first=True):
        self.batch_first = batch_first

    def padding(self, batch):
        max_len = max(map(lambda x: len(x[0]), batch))
        sorted_idx = sorted(range(len(batch)),key=lambda idx: len(batch[idx][0]), reverse=True)

        x, z, d, lengths = [], [], [], []
        for idx in sorted_idx:
            item = batch[idx]

            lengths.append(len(item[0]))
            x.append(pad_tensor(item[0].float(),pad=max_len,dim=0, dtype=torch.float))
            z.append(pad_tensor(item[1],pad=max_len,dim=0, dtype=torch.long))
            d.append(item[2])

        x, z = torch.stack(x, dim=0), torch.stack(z, dim=0)

        if not self.batch_first:
            x = x.transpose(0,1)
            z = z.transpose(0,1)

        x = nn.utils.rnn.pack_padded_sequence(x,lengths, batch_first=self.batch_first, enforce_sorted=False)

        return x, z, d

    def __call__(self, batch):
        return self.padding(batch)

def get_normalizations(o_doc):
    entity_list = []
    # find the corresponding normalized concept in true annotated docs
    for annotation in o_doc.annotations:
        ann_identifier = annotation.identifier
        for norm in o_doc.normalizations:
            if norm.argument_id == ann_identifier:
                entity_list.append(norm.preferred_term.strip())
                break

    return entity_list


def get_masks(tags, num_entity):
    masks = np.zeros((num_entity,len(tags)))
    last_visit = -1
    flag = False
    for i in range(num_entity):
        for j in range(len(tags)):
            if j > last_visit and 'B_' in tags[j]:
                masks[i][j] = 1
                last_visit = j
                j += 1
                while j < len(tags) and 'I_' in tags[j]:
                    masks[i][j] = 1
                    last_visit = j
                    j += 1
                flag = True
            if flag:
                flag=False
                break
    return masks

def adjust_mask(mask, text, tokens):
    count = 0
    t = copy.deepcopy(tokens)
    for line in text:
        for word in line:
            if word in [r'<s>', r'<\s>']:
                t.insert(count,word)
                mask.insert(count, 0.0)
            if t[count] != word:
                if '.' not in word:
                    # if combination of next few words makes the token the remove the extra mask
                    # eg: token[i]= malaria, token[i+1]=-endemic; word=malaria-endemic
                    flag = False
                    for i in range(1,10):
                        if ''.join(t[count:count+i]) == word:
                            flag = True
                            break
                    j=i
                    while j > 1 and flag:
                        item=t.pop(count+j-1)
                        t[count+j-2] = t[count+j-2]+item
                        item = mask.pop(count+j-1)
                        mask[count+j-2] = mask[count+j-2]+item
                        j -= 1
                elif t[count+1]=='.':
                    t.pop(count+1)
                    mask.pop(count+1)
                        
            count += 1
    
    # avoiding one special case of double period '..'; need proper fix
    max_idx = max(len(t), sum([len(x) for x in text])) - 1
    if t[max_idx] == '..' and text[len(text)-1][-1] != '..':
        mask = mask[0:max_idx]
    assert(len(mask) == sum([len(x) for x in text])), 'Mask length and text length mismatch.'
    return mask

def EL_set(docs, toD_mesh, id2idx_dict):
    data_dict = {}
    all_labels = []
    for idx, doc in enumerate(docs):
        _, bio_labels = transform_annotated_document_to_bio_format(doc)
        entity_list = get_normalizations(doc)
        masks = get_masks(bio_labels, len(entity_list))

        label, mask_list = [], []
        for i in range(len(entity_list)):
            # create C-2-D and UMIM-D and UMIM-C-M filter
            if '+' in entity_list[i]:
                entity_list[i] = entity_list[i].split('+')[0]
            elif '|' in entity_list[i]:
                entity_list[i] = entity_list[i].split('|')[0]
            if entity_list[i] not in id2idx_dict:
                item = toD_mesh.transform(entity_list[i])
                if item is not None:
                    if item not in id2idx_dict:
                        print(f"D MeSH {item} not found in Disease list. Skipping this normalization...")
                        continue
                    entity_list[i] = item
                else:
                    print(f"D MeSH equivalent of {entity_list[i]} not found. Skipping this normalization...")
                    continue
        
            label.append(torch.tensor(id2idx_dict[entity_list[i]]))
            mask = masks[i].tolist()
            # mask = adjust_mask(mask, t, tokens)
            mask_list.append(torch.tensor(mask))

            all_labels.append(entity_list[i])

        data_dict[doc.identifier] = (label, mask_list)
    
    return data_dict

def get_el_set(docs, x, data_dict, device=torch.device('cpu')):
    text, labels, mask_list = [], [], []
    for i,doc in enumerate(docs):
        label, mask = data_dict[doc.identifier]
        # number of entity
        num_entity = len(label)
        if num_entity == 0:
            continue
        t = x[i].repeat(num_entity, 1, 1)

        text.extend(t)
        labels.extend(label)
        mask_list.extend(mask)

    if len(text) == 0:
        return

    max_len = max(map(lambda x: len(x), text))
    sorted_idx = sorted(range(len(text)),key=lambda idx: len(text[idx]), reverse=True)

    x, y, z = [], [], []
    for idx in sorted_idx:
        x.append(pad_tensor(text[idx].float(),pad=max_len,dim=0, dtype=torch.float, device=device))
        y.append(labels[idx])
        z.append(pad_tensor(mask_list[idx].float(),pad=max_len,dim=0, dtype=torch.float))

    return [torch.stack(x, dim=0), torch.stack(y, dim=0), torch.stack(z, dim=0)]

def minibatch(docs, text_emb, ner_labels, ner_labels_vocab, batch_size=1, batch_first=True):
    length_x = len(docs)
    padding = PadCollate(batch_first=batch_first)
    for idx in range(0, math.ceil(len(docs)/batch_size)):
        if length_x - batch_size < 0:
            updated_size = length_x
        else:
            updated_size = batch_size
        length_x = length_x - batch_size
        batch_t = text_emb[idx*batch_size : idx*batch_size + updated_size]
        batch_ner = [torch.tensor(ner_labels_vocab.doc2id(i)) for i in ner_labels[idx*batch_size : idx*batch_size + updated_size]]
        batch_docs = [i for i in docs[idx*batch_size : idx*batch_size + updated_size]]
        batch = [i for i in zip(batch_t, batch_ner, batch_docs)]

        yield padding(batch)


def get_elmo_representation(data, elmo, elmo_dim=1024, device=torch.device('cpu')):
    text = []
    for idx, t in enumerate(data):
        char_id = batch_to_ids(t).to(device)
        with torch.no_grad():
            elmo_emb = elmo(char_id)
        t_emb = elmo_emb['elmo_representations'][0].view(-1, elmo_dim).detach().cpu()
        t_emb = torch.stack([tensor for tensor in t_emb if len(np.nonzero(tensor.numpy())[0])!=0],dim=0)
        text.append(t_emb)

    return text

def save_params(paths, params, scope_text, id2idx_dict, idx2id_dict, mesh_graph, ner_labels_vocab, ner_model, el_model, shared_model, linear_model):
    save_path = paths.experiment_folder

    with open(os.path.join(save_path,'MT_params'),'wb+') as f:
        pickle.dump([scope_text, id2idx_dict, idx2id_dict, mesh_graph, ner_labels_vocab, params],f)
    
    if params.only_NER:
        torch.save(ner_model, os.path.join(save_path, paths.ner_model_name ))
    elif params.only_EL:
        torch.save(el_model, os.path.join(save_path, paths.el_model_name))
    else:
        torch.save(ner_model, os.path.join(save_path, paths.ner_model_name ))
        torch.save(el_model, os.path.join(save_path, paths.el_model_name))

    torch.save(shared_model, os.path.join(save_path, paths.mt_model_name))
    torch.save(linear_model, os.path.join(save_path, paths.linear_model_name))


def load_params(paths):
    load_path = paths.experiment_folder

    with open(os.path.join(load_path,'MT_params'),'rb+') as f:
        scope_text, id2idx_dict, idx2id_dict, mesh_graph, ner_labels_vocab, params = pickle.load(f)

    ner_model, el_model = None, None
    if params.only_NER:
        ner_model = torch.load(os.path.join(load_path, paths.ner_model_name ))
    elif params.only_EL:
        el_model = torch.load(os.path.join(load_path, paths.el_model_name))
    else:
        ner_model = torch.load(os.path.join(load_path, paths.ner_model_name ))
        el_model = torch.load(os.path.join(load_path, paths.el_model_name))

    shared_model = torch.load(os.path.join(load_path, paths.mt_model_name))
    linear_model = torch.load(os.path.join(load_path, paths.linear_model_name))

    return ([scope_text, id2idx_dict, idx2id_dict, mesh_graph, ner_labels_vocab, params],[shared_model, ner_model, el_model, linear_model])


def train(paths, params, X, text_tr, ner_tr_tags, train_el_set, X_valid, x_val_tokens, text_val, ner_val_tags, val_el_set, ner_labels_vocab, scope_text, scope_embedding, a_hat, mesh_graph, id2idx_dict, idx2id_dict, writer, device=torch.device('cpu')):
    # Since we are dealing with RNN, this is an important field to be mindful of. Keep it False as default(dependency on variational dropout code)
    # this batch first denotes whether subsequent processing requires batch first oredered data;
    batch_first = params.batch_first

    params_dict = {'batch_size':params.batch_size,
                    'shuffle': True, 'num_workers': 12}

    # training generator
    training_set = Sequence(X,text_tr, ner_tr_tags, ner_labels_vocab.doc2id)
    training_generator = DataLoader(training_set, collate_fn=PadCollate(batch_first=batch_first),**params_dict)

    # shared model
    shared_model = BiLSTM(word_embedding_size=params.elmo_dim, word_rnn_units=params.rnn_units, use_word_self_attention=params.use_word_self_attention, dropout=params.dropout, device=device, use_gru=params.use_gru, num_layers=params.num_layers, batch_first=params.batch_first)
    shared_model.to(device)

    # NER models
    ner_model = BiLSTMCRF(len(ner_labels_vocab), word_embedding_size=params.rnn_units*2, word_rnn_units=params.ner.word_rnn_units, use_word_self_attention=params.ner.use_word_self_attention, dropout=params.ner.dropout, device=device, batch_first=params.batch_first)
    ner_model.to(device)

    # EL model
    el_model = EL_GCN(params.elmo_dim, params.elmo_dim, scope_embedding, hidden_dim=params.elmo_dim*2, dropout=params.EL.dropout)
    el_model.to(device)

    # Linear layer for attention
    linear = nn.Linear(params.rnn_units*2, out_features=1)
    linear.to(device)

    # types of activation functions
    if params.activation == 'fusedmax':
        activation = Fusedmax()
    elif params.activation == 'sparsemax':
        activation = Sparsemax()
    elif params.activation == 'oscarmax':
        activation = Oscarmax()

    # CHECK THE PARAMS
    optimizer1 = torch.optim.Adam(list(shared_model.parameters())+list(ner_model.parameters())+ list(el_model.parameters())+list(linear.parameters()) , lr=params.lr) # list(shared_model.parameters())+list(post_model.parameters())+list(ner_model.parameters()) + list(el_model.parameters())

    best_model_shared = None
    best_model_ner = None
    best_model_EL = None
    best_model_linear = None

    epochs = params.num_epochs
    best_f1, p_b, r_b, el_acc, en, el = 0, 0, 0, -0.1, 0, 0
    with open(os.path.join(paths.experiment_folder, params.output), 'w') as f:
        for epoch in range(epochs):

            trainine_loss = 0
            count = 0
            
            if params.only_NER:
                ner_model.train()
            elif params.only_EL:
                el_model.train()
            else:
                ner_model.train()
                el_model.train()
                
            shared_model.train()
            linear.train()
            
            # docs_tr, text_emb, ner_tr_tags_ = Shuffle_lists(copy.deepcopy(X), copy.deepcopy(text_tr), copy.deepcopy(ner_tr_tags)) # for debugging
            # for idx,(text_emb, ner_tr_labels, docs) in enumerate(minibatch(docs_tr, text_emb, ner_tr_tags_,  ner_labels_vocab, batch_size=2, batch_first=batch_first)):
            for text_emb, ner_tr_labels, docs in training_generator:
                text_emb,  ner_tr_labels = text_emb.to(device),  ner_tr_labels.to(device)

                optimizer1.zero_grad()
                x1, x2 = 0, 0

                mask_ner = torch.where(ner_tr_labels != ner_labels_vocab.vocab['<pad>'], torch.tensor([1], dtype=torch.uint8,device=device), \
                            torch.tensor([0], dtype=torch.uint8, device=device))
                
                x, lengths = shared_model(text_emb)

                score = linear(x).squeeze(2)
                if params.use_activation:
                    if not batch_first:
                        score = score.transpose(0,1)
                        x = x.transpose(0,1)
                        if params.activation == 'softmax':
                            att_weights = nn.functional.softmax(score, dim=-1)
                        else:
                            score = score.cpu()
                            att_weights = activation(score, lengths=lengths).to(device)
                        x = x * att_weights.unsqueeze(2)
                        x = x.transpose(0,1)
                    else:
                        score = score.cpu()
                        att_weights = activation(score, lengths=lengths).to(device)
                        x = x * att_weights.unsqueeze(2)

                if params.NER_type1:
                    x_ = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=batch_first)
                else:
                    x_ = x

                if not params.only_EL:
                    x1, _ = ner_model(x_, ner_tr_labels, mask_ner)
                    loss = x1
                else:
                    loss = 0
                

                if not batch_first:
                    x = x.transpose(0,1)

                if not params.only_NER:
                    el_data = get_el_set(docs, x, train_el_set, device=device)
                    if el_data is not None:
                        t, labels, mask = el_data

                        t, labels, mask = t.to(device), labels.to(device), mask.to(device)
                        mask = mask.unsqueeze(2).expand(-1,-1,t.shape[2])
                        t = mask*t
                        t = torch.mean(t, dim=1)
                        t = el_model(t, a_hat)
                        x2 = nn.functional.cross_entropy(t, labels)
                        
                        loss += x2
                    elif loss == 0:
                        continue

                loss.backward()
                optimizer1.step()
                
                trainine_loss += loss.item()
                count += 1

            writer.add_scalars('training',{'loss':trainine_loss/count},global_step=epoch)

            print(f'Epoch: {epoch}\t Loss: {trainine_loss/count}')
            f.writelines(f'Epoch: {epoch}\t Loss: {trainine_loss/count}\n')

            # evaluate on validation set
            with torch.no_grad():
                if params.only_NER:
                    ner_model.eval()
                elif params.only_EL:
                    el_model.eval()
                else:
                    ner_model.eval()
                    el_model.eval()
                    
                shared_model.eval()
                linear.eval()

                pred_labels_ner, orig_labels_ner = [], []
                pred_labels_el, orig_labels_el = [], []
                for text_emb, ner_val_labels, docs in minibatch(copy.deepcopy(X_valid), copy.deepcopy(text_val),
                            copy.deepcopy(ner_val_tags), ner_labels_vocab, batch_size=1, batch_first=batch_first):

                    orig_labels_ner.extend(ner_val_labels.cpu().numpy().tolist())

                    text_emb, ner_val_labels = text_emb.to(device), ner_val_labels.to(device)

                    mask_ner = torch.where(ner_val_labels != ner_labels_vocab.vocab['<pad>'], torch.tensor([1], dtype=torch.uint8,device=device), \
                            torch.tensor([0], dtype=torch.uint8, device=device))

                    x, lengths = shared_model(text_emb)

                    score = linear(x).squeeze(2)
                    if params.use_activation:
                        if not batch_first:
                            score = score.transpose(0,1)
                            x = x.transpose(0,1)
                            if params.activation == 'softmax':
                                att_weights = nn.functional.softmax(score.to(device), dim=-1)
                            else:
                                score = score.cpu()
                                att_weights = activation(score, lengths=lengths).to(device)
                            x = x * att_weights.unsqueeze(2)
                            x = x.transpose(0,1)
                        else:
                            score = score.cpu()
                            att_weights = activation(score, lengths=lengths).to(device)
                            x = x * att_weights.unsqueeze(2)

                    if params.NER_type1:
                        x_ = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=batch_first)
                    else:
                        x_ = x

                    if not params.only_EL:
                        x1, _ = ner_model.decode(x_, mask_ner)
                        pred_labels_ner.extend(x1)

                    if not batch_first:
                        x = x.transpose(0,1)

                    if not params.only_NER:
                        el_data = get_el_set(docs, x, val_el_set, device=device)
                        if el_data is not None:
                            t, labels, mask = el_data
                            t, labels, mask = t.to(device), labels.to(device), mask.to(device)
                            mask = mask.unsqueeze(2).expand(-1,-1,t.shape[2])
                            t = mask*t
                            t = torch.mean(t, dim=1)
                            t = el_model(t, a_hat)
                            t = nn.functional.softmax(t, dim=1)
                            _, max_idx = torch.max(t, dim=1)

                            [orig_labels_el.append(idx2id_dict[i.item()]) for i in labels]
                            [pred_labels_el.append(idx2id_dict[i.item()]) for i in max_idx]

                if len(pred_labels_ner) > 0:
                    lengths = map(len, x_val_tokens)
                    tags = inverse_transform(np.asarray(pred_labels_ner), ner_labels_vocab, lengths)
                    print('F1: ',f1_score(ner_val_tags, tags),'\t Acc: ', accuracy_score(ner_val_tags, tags))
                    print(classification_report(ner_val_tags, tags))

                    x_pred = transform_bio_tags_to_annotated_documents(x_val_tokens, tags, X_valid)

                    p, r, f1 = annotation_precision_recall_f1score(x_pred, X_valid)
                    print("Disease:\tPrecision: ", p, "\tRecall: " ,r, "\tF-score: ", f1)
                    f.writelines(f'Disease:\tPrecision: {p}\tRecall: {r}\tF-score: {f1}\n')
                    writer.add_scalars('Disease',{'Precision':p}, global_step=epoch)
                    writer.add_scalars('Disease',{'Recall':r},global_step=epoch)
                    writer.add_scalars('Disease',{'F1':f1},global_step=epoch)

                    if f1 > best_f1:
                        best_f1 = f1
                        p_b = p
                        r_b = r
                        en = epoch
                        if params.only_NER:
                            print('Saving model')
                            best_model_shared = shared_model
                            best_model_ner = ner_model
                            best_model_linear = linear
                
                acc = -0.1
                if len(orig_labels_el) > 0:
                    print(cls_rpt(orig_labels_el, pred_labels_el))
                    acc = acc_scr(orig_labels_el, pred_labels_el)
                    print('EL: accuracy ',acc)
                    f.writelines(F'EL: accuracy {acc}\n')                    
                    writer.add_scalars('EL',{'accuracy':acc}, global_step=epoch)

                    if acc > el_acc:
                        el_acc = acc
                        el = epoch
                        print('Saving model')
                        best_model_shared = shared_model
                        best_model_ner = ner_model
                        best_model_EL = el_model
                        best_model_linear = linear

                print("Disease: Best\tPrecision: ", p_b, "\tRecall: " ,r_b, "\tF-score: ", best_f1,"\t epoch:",en)
                f.writelines(f"Disease: Best\tPrecision: {p_b}\tRecall: {r_b}\tF-score: {best_f1}\t epoch:{en}\n")
                print('EL: Best \t Acc:', el_acc,'\tepoch:', el)
                f.writelines(f'EL: Best \t Acc: {el_acc}\tepoch:{el}\n')

    save_params(paths, params, scope_text, id2idx_dict, idx2id_dict, mesh_graph, ner_labels_vocab, ner_model, el_model, shared_model, linear)


def test(paths, params, X_test, x_test_tokens, text_test, ner_test_tags, sentence_pad = False, device=torch.device('cpu')):
    # Since we are dealing with RNN, this is an important field to be mindful of. Keep it False as default(dependency on variational dropout code)
    batch_first = False

    # load params
    [scope_text_, id2idx_dict, idx2id_dict, mesh_graph, ner_labels_vocab, params_old],[shared_model, ner_model, el_model, linear] = load_params(paths)

    node_list = list(idx2id_dict.values())
    mesh_folder = paths.MeSH_folder
    if not os.path.exists(os.path.join(mesh_folder, 'a_hat_matrix')):
        a_matrix = get_adjacancy_matrix(mesh_graph, node_list)

        a_matrix = sparse.coo_matrix(a_matrix)
        with open(os.path.join(mesh_folder, 'a_hat_matrix'), 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(os.path.join(mesh_folder, 'a_hat_matrix'), 'rb') as f:
            a_matrix = pickle.load(f)

    # evaluate a_hat matrix
    i = torch.tensor([a_matrix.row, a_matrix.col], dtype=torch.long, device=device)
    v = torch.tensor(a_matrix.data, dtype=torch.float32, device=device)
    a_hat = torch.sparse.FloatTensor(i, v, torch.Size([len(node_list), len(node_list)])).to(device)
    
    toD_mesh = Convert2D(paths.ctd_file, paths.c2m_file)
    test_el_set = EL_set(X_test, toD_mesh, id2idx_dict)
    
    # type of activation
    if params_old.activation == 'fusedmax':
        activation = Fusedmax()
    elif params_old.activation == 'sparsemax':
        activation = Sparsemax()
    elif params_old.activation == 'oscarmax':
        activation = Oscarmax()

    # evaluate
    with torch.no_grad():
        if params_old.only_NER:
            ner_model.eval()
        elif params_old.only_EL:
            el_model.eval()
        else:
            ner_model.eval()
            el_model.eval()

        shared_model.eval()
        linear.eval()

        pred_labels_ner, orig_labels_ner = [], []
        pred_labels_el, orig_labels_el, pred_index = [], [], []
        for text_emb, ner_val_labels, docs in minibatch(copy.deepcopy(X_test), copy.deepcopy(text_test),
                        copy.deepcopy(ner_test_tags), ner_labels_vocab, batch_size=1, batch_first=batch_first):
            orig_labels_ner.extend(ner_val_labels.cpu().numpy().tolist())

            text_emb,  ner_val_labels = text_emb.to(device),  ner_val_labels.to(device)

            mask_ner = torch.where(ner_val_labels != ner_labels_vocab.vocab['<pad>'], torch.tensor([1], dtype=torch.uint8,device=device), \
                    torch.tensor([0], dtype=torch.uint8, device=device))

            x, lengths = shared_model(text_emb)

            score = linear(x).squeeze(2)
            if params_old.use_activation:
                if not batch_first:
                    score = score.transpose(0,1)
                    x = x.transpose(0,1)
                    if params_old.activation == 'softmax':
                        att_weights = nn.functional.softmax(score.to(device), dim=-1)
                    else:
                        score = score.cpu()
                        att_weights = activation(score, lengths=lengths).to(device)
                    x = x * att_weights.unsqueeze(2)
                    x = x.transpose(0,1)
                else:
                    score = score.cpu()
                    att_weights = activation(score, lengths=lengths).to(device)
                    x = x * att_weights.unsqueeze(2)

            if params_old.NER_type1:
                x_ = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=batch_first)
            else:
                x_ = x

            if not params_old.only_EL:
                x1, _ = ner_model.decode(x_, mask_ner)
                pred_labels_ner.extend(x1)

            if not batch_first:
                x = x.transpose(0,1)

            if not params_old.only_NER:
                el_data = get_el_set(docs, x, test_el_set, device=device)
                if el_data is not None:
                    t, labels, mask = el_data
                    t, labels, mask = t.to(device), labels.to(device), mask.to(device)
                    mask = mask.unsqueeze(2).expand(-1,-1,t.shape[2])
                    t = mask*t
                    t = torch.mean(t, dim=1)
                    t = el_model(t, a_hat)
                    t = nn.functional.softmax(t, dim=1)
                    _, max_idx = torch.max(t, dim=1)

                    [orig_labels_el.append(idx2id_dict[i.item()]) for i in labels]
                    [pred_labels_el.append(idx2id_dict[i.item()]) for i in max_idx]
                    _, index_sorted = torch.sort(t, descending=True)
                    pred_index.extend(index_sorted)

        if len(pred_labels_ner) > 0:
            lengths = map(len, x_test_tokens)
            tags = inverse_transform(np.asarray(pred_labels_ner), ner_labels_vocab, lengths)
            print('F1: ',f1_score(ner_test_tags, tags),'\t Acc: ', accuracy_score(ner_test_tags, tags))
            print(classification_report(ner_test_tags, tags))

            x_pred = transform_bio_tags_to_annotated_documents(x_test_tokens, tags, X_test)

            p, r, f1 = annotation_precision_recall_f1score(x_pred, X_test)
            print("Disease:\tPrecision: ", p, "\tRecall: " ,r, "\tF-score: ", f1)
        
        if len(pred_index) > 0:
            labels = orig_labels_el
            sorted_list = []
            pred_2, pred_5, pred_10, pred_15, pred_30,rank, reciprocal_rank = [], [], [], [], [], [], []
            for idx, item in enumerate(pred_index):
                id_sorted = [idx2id_dict[i.item()] for i in item]
                sorted_list.append(id_sorted)

                if labels[idx] in id_sorted:
                    rank.append(id_sorted.index(labels[idx])+1)
                    reciprocal_rank.append(1/(id_sorted.index(labels[idx])+1))
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

            print(cls_rpt(labels, pred_labels_el))
            print(f'Mean Reciprocal Rank: {np.mean(reciprocal_rank)}')
            acc = accuracy_score(labels, pred_labels_el)
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


def main(paths, params):
    path_to_train_input = paths.training
    path_to_valid_input = paths.develop
    path_to_test= paths.test
    ctd_file = paths.ctd_file
    c2m_file = paths.c2m_file
    toD_mesh = Convert2D(ctd_file, c2m_file)

    sentence_pad = False # Don't pad sentence with begin and end sentence '<s>' and '<\s>

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

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
    mesh_folder = paths.MeSH_folder
    mt_folder = paths.multitask_folder


    # read disease file
    with open(disease_file,'r') as f:
        disease_data = f.readlines()

    mesh_dict = read_mesh_file(mesh_file)

    mesh_graph = nx.read_gpickle(mesh_graph_file)
    mesh_graph = mesh_graph.to_undirected()
    scope_text, id2idx_dict, idx2id_dict = mesh_dict_to_tokens(mesh_dict, disease_data)
    node_list = list(idx2id_dict.values())

    # A_HAT metrix for GCN
    if not os.path.exists(os.path.join(mesh_folder, 'a_hat_matrix')):
        a_matrix = get_adjacancy_matrix(mesh_graph, node_list)

        a_matrix = sparse.coo_matrix(a_matrix)
        with open(os.path.join(mesh_folder, 'a_hat_matrix'), 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(os.path.join(mesh_folder, 'a_hat_matrix'), 'rb') as f:
            a_matrix = pickle.load(f)

    i = torch.tensor([a_matrix.row, a_matrix.col], dtype=torch.long, device=device)
    v = torch.tensor(a_matrix.data, dtype=torch.float32, device=device)
    a_hat = torch.sparse.FloatTensor(i, v, torch.Size([len(node_list), len(node_list)])).to(device)

    # Construct usable data format
    x_tr_text, ner_tr_tags, x_tr_tokens = annotated_docs_to_tokens(X, sentence_pad=sentence_pad)
    x_val_text, ner_val_tags, x_val_tokens = annotated_docs_to_tokens(X_valid, sentence_pad=sentence_pad)
    x_test_text, ner_test_tags, x_test_tokens = annotated_docs_to_tokens(X_test, sentence_pad=sentence_pad)

    # elmo embeddings
    options_file = paths.elmo_options
    weight_file = paths.elmo_weights
    ELMO_folder = paths.elmo_folder
    elmo_dim = params.elmo_dim
    elmo = Elmo(options_file, weight_file, 2,dropout=0)
    elmo.to(device)

    with torch.no_grad():
        if not os.path.exists(os.path.join(mt_folder,'text_tr_elmo_split.pkl')):
            text_tr = get_elmo_representation(x_tr_text, elmo, elmo_dim=params.elmo_dim, device=device)
            with open(os.path.join(mt_folder,'text_tr_elmo_split.pkl'),'wb+') as f:
                pickle.dump(text_tr, f)
        else:
            with open(os.path.join(mt_folder,'text_tr_elmo_split.pkl'),'rb+') as f:
                text_tr = pickle.load(f)
        
        if not os.path.exists(os.path.join(mt_folder,'text_val_elmo_split.pkl')):
            text_val = get_elmo_representation(x_val_text, elmo, elmo_dim=params.elmo_dim, device=device)
            with open(os.path.join(mt_folder,'text_val_elmo_split.pkl'),'wb+') as f:
                pickle.dump(text_val, f)
        else:
            with open(os.path.join(mt_folder,'text_val_elmo_split.pkl'),'rb+') as f:
                text_val = pickle.load(f)

        if not os.path.exists(os.path.join(paths.multitask_folder,'text_test_elmo_split.pkl')):
            text_test = get_elmo_representation(x_test_text, elmo, elmo_dim=params.elmo_dim, device=device)
            with open(os.path.join(paths.multitask_folder,'text_test_elmo_split.pkl'),'wb+') as f:
                pickle.dump(text_test, f)
        else:
            with open(os.path.join(paths.multitask_folder,'text_test_elmo_split.pkl'),'rb+') as f:
                text_test = pickle.load(f)

    # NER label vocab
    ner_labels_vocab = Vocabulary(lower=False)
    ner_labels_vocab.add_documents(ner_tr_tags)
    ner_labels_vocab.build()

    # mesh scope embedding
    if not os.path.exists(os.path.join(paths.dump_folder, 'scope_emb.pkl')):
        scope_embedding, _ = get_scope_elmo(elmo, ELMO_folder, scope_text, elmo_dim, idx2id_dict, id2idx_dict, device=device)
        with open(os.path.join(paths.dump_folder, 'scope_emb.pkl'), 'wb') as f:
            pickle.dump(scope_embedding, f)
    else:
        with open(os.path.join(paths.dump_folder, 'scope_emb.pkl'), 'rb') as f:
            scope_embedding = pickle.load(f)
            
    train_el_set = EL_set(X, toD_mesh, id2idx_dict)
    val_el_set = EL_set(X_valid, toD_mesh, id2idx_dict)


    train(paths, params, X, text_tr, ner_tr_tags, train_el_set, X_valid, x_val_tokens, text_val,
            ner_val_tags, val_el_set, ner_labels_vocab, scope_text, scope_embedding, a_hat, mesh_graph, id2idx_dict, idx2id_dict, writer, device=device)
    

    # test(paths, params, X_test, x_test_tokens, text_test, ner_test_tags, test_el_set, device=device)


# if __name__ == '__main__':
    # Obtain the training, validation and test dataset

    # base_path = r'/media/druv022/Data1/Masters/Thesis/'
    # paths = Paths(base_folder = base_path)
    # paths.ner_model_name = 'ner_model.pt'
    # paths.el_model_name = 'el_model.pt'
    # paths.mt_model_name = 'shared_model.pt'
    # paths.linear_model_name = 'linear_model.pt'

    # params = MultiTask_Params()
    # params.batch_first = False
    # params.ner.use_word_self_attention = True
    # params.batch_size = 32
    # params.activation = 'sparsemax'
    # params.num_epochs = 500

    # main(paths, params)

