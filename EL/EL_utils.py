import numpy as np
import copy
from utils.convert2D import Convert2D
from nerds.doc.bio import transform_annotated_document_to_bio_format
from nerds.util.nlp import text_to_tokens, text_to_sentences
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import torch.nn as nn
import math
from utils.mesh import read_mesh_file
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from utils.vocab import Vocabulary
from sklearn.utils import shuffle as Shuffle_lists
from gensim.models import KeyedVectors

def get_adjacancy_matrix(mesh_graph, node_list):
    # get adjancy matrix
    adj_matrix = nx.linalg.graphmatrix.adjacency_matrix(mesh_graph, nodelist=node_list)
    adj_matrix = adj_matrix.toarray() + np.identity(len(node_list))

    laplacian_matrix = nx.linalg.laplacianmatrix.laplacian_matrix(mesh_graph, nodelist=node_list)
    laplacian_matrix = laplacian_matrix.toarray()

    d_matrix = laplacian_matrix + adj_matrix

    adj_matrix_compressed = sparse.csr_matrix(adj_matrix)

    del laplacian_matrix
    del adj_matrix

    d_matrix = np.linalg.inv(d_matrix)
    d_matrix = np.sqrt(d_matrix)
    prod = np.matmul(d_matrix, adj_matrix_compressed.toarray())

    return np.matmul(prod, d_matrix)


def get_scope_elmo(model, ELMO_folder, scope_text, elmo_dim, idx2id_dict, id2idx_dict, device=torch.device('cpu')):
    """ Get scope note ELMo embedding representation
    
    """
    with torch.no_grad():
        elmo_embeddings = [model(batch_to_ids(i).to(device)) for i in scope_text]
        elmo_scope_embeddings = [i['elmo_representations'][0].view(-1, elmo_dim) for i in elmo_embeddings]
        
    elmo_scope_embeddings = [torch.mean(item, dim=0) for item in elmo_scope_embeddings]
    elmo_scope_embeddings = torch.stack(elmo_scope_embeddings)
    
    return elmo_scope_embeddings, idx2id_dict

def re_encode(embedding_path, elmo_embeddings, idx2id_dict, device=torch.device('cpu')):
    embedding_size= 1024

    wv_node2vec = KeyedVectors(embedding_size).load(embedding_path)

    for item in idx2id_dict:
        m_id = idx2id_dict[item]
        if m_id in wv_node2vec:
            emb = wv_node2vec[m_id]
            elmo_embeddings[item] = torch.tensor(emb, device=device).unsqueeze(0)

    return elmo_embeddings


def get_text_vocab(texts):
    word_vocab = Vocabulary()
    char_vocab = Vocabulary(lower=False)

    for item in texts:
        word_vocab.add_documents(item)
        for words in item:
            char_vocab.add_documents(words)

    word_vocab.build()
    char_vocab.build()

    return word_vocab, char_vocab


def annotated_docs_to_tokens(docs):
    text_list = []
    for doc in docs:
        text = [[r'<s>']+ text_to_tokens(sent)+[r'<\s>'] for sent in text_to_sentences(doc.plain_text_)[0] if len(sent.split()) > 0]
        text_list.append(text)

    return text_list

def mesh_dict_to_tokens(mesh_dict, disease_list):
    scope_list = []
    id2idx_dict = {}
    idx2id_dict = {}
    for i,id in enumerate(disease_list):
        id = id.replace('\n','')
        mesh = mesh_dict[id]
        scope_list.append(mesh.scope_note)
        id2idx_dict[id] = i
        idx2id_dict[i] = id

    # add unknown
    scope_list.append([[r'<s>','<unk>',r'<\s>']])
    id2idx_dict['<unk>'] = i+1
    idx2id_dict[i+1] = '<unk>'

    return scope_list, id2idx_dict, idx2id_dict

def get_normalizations(o_doc, pred_doc):
    """ associate normalizing concept from original document to predicted document
    """
    entity_list = []
    # find the corresponding normalized concept in true annotated docs
    for annotation in o_doc.annotations:
        # max predicted annotations
        max_ann_length = len(pred_doc.annotations)
        ann_counter = 0
        # iterate in predicted annotations
        complete_flag = False
        while ann_counter < max_ann_length:
            # find the matching annotation text and offset in the predicted annotations
            # TODO: properly fix inverted comma issue in the whole NERDS
            if '"' in pred_doc.annotations[ann_counter].text:
                flag = annotation.text.replace(' ','') == pred_doc.annotations[ann_counter].text.replace('"','').replace(' ','')
            else:
                flag = False
            if (annotation.text == pred_doc.annotations[ann_counter].text or flag) and annotation.offset == pred_doc.annotations[ann_counter].offset:
                ann_identifier = annotation.identifier
                pred_doc.annotations.remove(pred_doc.annotations[ann_counter])
                for norm in o_doc.normalizations:
                    if norm.argument_id == ann_identifier:
                        entity_list.append(norm.preferred_term.strip())
                        complete_flag = True
                        break
            if complete_flag:
                break
            ann_counter += 1

    return entity_list

def get_masks(tags, num_entity):
    """Generate mask to select B and I tagged words
    """
    masks = np.zeros((num_entity,len(tags)))
    last_visit = -1
    flag = False
    for i in range(num_entity):
        for j in range(len(tags)):
            if j > last_visit and tags[j] == 'B':
                masks[i][j] = 1
                last_visit = j
                j += 1
                while j < len(tags) and tags[j] == 'I':
                    masks[i][j] = 1
                    last_visit = j
                    j += 1
                flag = True
            if flag:
                flag=False
                break

    return masks

def adjust_mask(mask, text, tokens):
    """ Align the mask correctly considering different preprocessing steps by ELMo and our tokenizer
    """
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

# here we nake sure correct tags are picked
def check_tags(o_tags, p_tags):
    max_sent_length = len(o_tags)
    new_correct_tags = []
    temp1=[]
    temp2=[]
    update = False

    assert(len(o_tags) == len(p_tags)), 'tags mismatch'
    for lbl_idx, label in enumerate(p_tags):
        # check if 'Begin' label matches for true and predicted tag
        if 'B_' in label and 'B_' in o_tags[lbl_idx]:
            temp2.append('O')
            # update if full entiy span is found (single or multi word)
            if update:
                update = False
                new_correct_tags.extend(temp1)
                temp1 = []
                temp2 = []

            # check if next tags of true and predicted tag matches; needed for complete 'BI' sequence
            if lbl_idx+1 < max_sent_length and p_tags[lbl_idx+1]!=o_tags[lbl_idx+1]:
                continue
            update = True
            temp1.append('B')
        # check if 'Inside' label matches for true and predicted tag
        elif 'I_' in label and 'I_' in o_tags[lbl_idx]:
            temp2.append('O')
            # check if next tags of true and predicted tag matches; needed for complete 'BI' sequence
            if lbl_idx+1 < max_sent_length and p_tags[lbl_idx+1]!=o_tags[lbl_idx+1]:
                update = False
                continue
            # add only if 'Begin' tag matches
            if update:
                temp1.append('I')

        else:
            # update if full entiy span is found (single or multi word)
            if update:
                update = False
                new_correct_tags.extend(temp1)
                temp1 = []
                temp2 = []

            new_correct_tags.extend(temp2)
            new_correct_tags.append('O')
            temp1 = []
            temp2 = []
        
        if lbl_idx == max_sent_length - 1:
            if update:
                update = False
                new_correct_tags.extend(temp1)
                temp1 = []
                temp2 = []
            # if last tags are 'I_' but incorrect
            if len(temp2) > 0:
                new_correct_tags.extend(temp2)


    assert(len(o_tags) == len(new_correct_tags)), 'tags mismatch'

    return new_correct_tags

def construct_data(data, annotated_docs, predictions, scope_note, id_dict, ctd_file, c2m_file, use_ELMO = True, elmo_model=None, elmo_dim=1024, device=torch.device('cpu')):
    """ re-format the data in easily trainable format using pytorch generators
    """
    text = [] # sentence
    text_emb = []
    scope = [] # scope note
    m_id = [] # mesh ID
    mask_list = [] # mask list
    label = [] # labels for positive and vegative examples

    toD = Convert2D(ctd_file, c2m_file)
    skipped_id = []
    for idx, pred_doc in enumerate(annotated_docs):
        tags = predictions[idx]

        o_doc = data[idx]

        tokens, bio_labels = transform_annotated_document_to_bio_format(o_doc)

        new_tags = check_tags(bio_labels, tags)
        entity_list = get_normalizations(o_doc, copy.deepcopy(pred_doc))

        masks = get_masks(new_tags, len(entity_list))

        for i in range(len(entity_list)):
            # create C-2-D and UMIM-D and UMIM-C-M filter
            if '+' in entity_list[i]:
                entity_list[i] = entity_list[i].split('+')[0]
            elif '|' in entity_list[i]:
                entity_list[i] = entity_list[i].split('|')[0]
            if entity_list[i] not in id_dict:
                item = toD.transform(entity_list[i])
                if item is not None:
                    if item not in id_dict:
                        print(f"D MeSH {item} not found in Disease list. Skipping this normalization...")
                        skipped_id.append(item)
                        continue
                    entity_list[i] = item
                else:
                    print(f"D MeSH equivalent of {entity_list[i]} not found. Skipping this normalization...")
                    skipped_id.append(entity_list[i])
                    continue
            note = []
            # text, scope_note, Mesh_ID, Mask, positive_lable
            if use_ELMO:
                t = [[r'<s>']+text_to_tokens(sent)+[r'<\s>'] for sent in text_to_sentences(pred_doc.plain_text_)[0] if len(sent.split()) > 0]

                char_id = batch_to_ids(t).to(device)
                with torch.no_grad():
                    elmo_emb = elmo_model(char_id)
                t_emb = elmo_emb['elmo_representations'][0].view(-1, elmo_dim).detach().cpu()
                t_emb = torch.stack([tensor for tensor in t_emb if len(np.nonzero(tensor.numpy())[0])!=0],dim=0)
                text_emb.append(t_emb)
                text.extend(t)

                note = scope_note[id_dict[entity_list[i]]]
                note = batch_to_ids(note).to(device)
                with torch.no_grad():
                    elmo_emb = elmo_model(note)
                note = elmo_emb['elmo_representations'][0].view(-1, elmo_dim).detach().cpu()
                scope.append(note)
                mask = masks[i].tolist()
                mask = adjust_mask(mask, t, tokens)
                mask_list.append(torch.tensor(mask))

            else:
                t = text_to_tokens(pred_doc.plain_text_)
                text.append(t)
                _ = [note.extend(line[1:-1]) for line in scope_note[id_dict[entity_list[i]]] if len(line) > 1]
                scope.append(note)
                mask = masks[i].tolist()
                mask = adjust_mask(mask, [t], tokens)
                mask_list.append(torch.tensor(mask))

                assert (len(t) == len(mask)), 'Length of mask is not equal to length of sentence.'

            m_id.append(entity_list[i])
            label.append(1)

    print('Total skipped: ', len(skipped_id), ' unique skips: ', len(set(skipped_id)))
    sample = []
    for i in range(len(text)):
        sample.append((text[i], text_emb[i], scope[i], m_id[i], mask_list[i], label[i]))

    return sample, text

def minibatch(tr_data, scope_notes, transform=None, char_vocab=None, id2idx=None, use_neg=False, use_elmo=False, elmo_model=None, elmo_dim=1024, n_samples = 1, batch_size=1, device=torch.device('cpu')):
    """ Generate minibatch
    """
    text, text_emb, scope, label, mask_list = [], [], [], [], []
    for sample in tr_data:
        if use_neg:
            t, t_emb, pos, neg, mask = sample
        else:
            t, t_emb, s, m_id, mask, l = sample

        if char_vocab:
            t = [torch.tensor(char_vocab.doc2id(i)) for i in t]
            t = nn.utils.rnn.pad_sequence(t, batch_first=True)
            char_mask = mask.unsqueeze(1).expand(-1,t.shape[1])
            t = torch.masked_select(t, char_mask.byte())

        text.append(t)
        text_emb.append(t_emb)
        mask_list.append(mask)

        # use negative sample for similarity based training
        if use_neg:
            scope.append(pos[0])
            label.append(1)
            neg_scope_notes = neg[0]
            np.random.shuffle(neg_scope_notes)

            neg_scope_notes = neg_scope_notes[0:math.ceil(int(len(neg_scope_notes)/2))]
            
            while len(neg_scope_notes) < 2*n_samples:
                rand_idx = np.random.randint(len(scope_notes))
                note = []
                if use_elmo:
                    note = scope_notes[rand_idx]
                else:
                    _ = [note.extend(line[1:-1]) for line in scope_notes[rand_idx] if len(line) > 1]
                neg_scope_notes.append(note)

            np.random.shuffle(neg_scope_notes)

            for i,item in enumerate(neg_scope_notes):
                if i > n_samples-1:
                    break
                text.append(t)
                text_emb.append(t_emb)
                scope.append(item)
                label.append(0)
                mask_list.append(mask)
        else:
            scope.append(s)
            label.append(id2idx[m_id])

    length_x = len(text)

    for idx in range(0, math.ceil(length_x/batch_size)):
        if length_x - batch_size < 0:
            updated_size = length_x
        else:
            updated_size = batch_size
        length_x = length_x - batch_size
        batch_t = text[idx*batch_size : idx*batch_size + updated_size]
        batch_t_emb = text_emb[idx*batch_size : idx*batch_size + updated_size]
        batch_s = scope[idx*batch_size : idx*batch_size + updated_size]
        batch_l = torch.tensor(label[idx*batch_size : idx*batch_size + updated_size])
        # pad the mask tensor
        batch_m = nn.utils.rnn.pad_sequence(mask_list[idx*batch_size : idx*batch_size + updated_size], batch_first=True)
        
        if transform is not None:
            yield transform(batch_t), transform(batch_s), batch_l, batch_m
        else:
            if char_vocab:
                batch_t = nn.utils.rnn.pad_sequence(batch_t, batch_first=True)
            batch_t_emb = nn.utils.rnn.pad_sequence(batch_t_emb, batch_first=True)
            batch_s = nn.utils.rnn.pad_sequence(batch_s, batch_first=True)

            yield batch_t, batch_t_emb, batch_s, batch_l, batch_m

def minibatch_val(val_data, char_vocab=None, transform=None, batch_size=1):
    """ Minibatch for validation
    """
    text, text_emb, m_id, mask_list= [],[],[], []
    for item in val_data:
        t = item[0]
        mask = item[4]
        if char_vocab is not None:
            t = [torch.tensor(char_vocab.doc2id(i)) for i in t]
            t = nn.utils.rnn.pad_sequence(t, batch_first=True)
            char_mask = mask.unsqueeze(1).expand(-1,t.shape[1])
            t = torch.masked_select(t, char_mask.byte())       
        
        text.append(t)
        text_emb.append(item[1])
        m_id.append(item[3])
        mask_list.append(item[4])

    length_x = len(text)

    for idx in range(0, math.ceil(length_x/batch_size)):
        if length_x - batch_size < 0:
            updated_size = length_x
        else:
            updated_size = batch_size
        length_x = length_x - batch_size
        batch_t = text[idx*batch_size : idx*batch_size + updated_size]
        batch_t_emb = text_emb[idx*batch_size : idx*batch_size + updated_size]
        batch_m = nn.utils.rnn.pad_sequence(mask_list[idx*batch_size : idx*batch_size + updated_size], batch_first=True)
        batch_id = m_id[idx*batch_size : idx*batch_size + updated_size]

        if transform is not None:
            yield transform(batch_t), batch_id, batch_m
        else:
            batch_t_emb = nn.utils.rnn.pad_sequence(batch_t_emb, batch_first=True)
            yield batch_t, batch_t_emb, batch_id, batch_m

def minibatch_mesh(scope_note, transform=None, batch_size=1, use_elmo=False):
    """minibatch for training MeSH encoding
    """
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


def padding_EL(batch):

    batch_t, batch_t_emb, batch_s, batch_l, batch_m = [], [], [], [], []
    for item in batch:
        batch_t.append(item[0])
        batch_t_emb.append(item[1])
        batch_s.append(item[2])
        batch_l.append(item[3])
        batch_m.append(item[4])

    batch_t_emb = nn.utils.rnn.pad_sequence(batch_t_emb, batch_first=True)
    batch_s = nn.utils.rnn.pad_sequence(batch_s, batch_first=True)
    batch_l = torch.tensor(batch_l)
    batch_m = nn.utils.rnn.pad_sequence(batch_m, batch_first=True)

    return batch_t, batch_t_emb, batch_s, batch_l, batch_m


class EL_sequence(Dataset):

    def __init__(self, tr_data, scope_notes, id2idx, char_vocab, use_elmo=True, n_samples = 0, use_neg=False):
        self.text, self.t_emb, self.scope, self.label, self.mask_list = [], [], [], [], []
        for sample in tr_data:
            if use_neg:
                t, t_emb, pos, neg, mask = sample
            else:
                t, t_emb, s, m_id, mask, l = sample
            t_ = []
            _ = [t_.append(torch.tensor(char_vocab.doc2id(i))) for i in t]
            t = nn.utils.rnn.pad_sequence(t_, batch_first=True)
            char_mask = mask.unsqueeze(1).expand(-1,t.shape[1])
            t = torch.masked_select(t, char_mask.byte())

            self.text.append(t)
            self.t_emb.append(t_emb)
            self.mask_list.append(mask)

            if use_neg:
                self.scope.append(pos[0])
                self.label.append(1)
                neg_scope_notes = neg[0]
                np.random.shuffle(neg_scope_notes)

                neg_scope_notes = neg_scope_notes[0:math.ceil(int(len(neg_scope_notes)/2))]
                
                while len(neg_scope_notes) < 2*n_samples:
                    rand_idx = np.random.randint(len(scope_notes))
                    note = []
                    if use_elmo:
                        note = scope_notes[rand_idx]
                    else:
                        _ = [note.extend(line[1:-1]) for line in scope_notes[rand_idx] if len(line) > 1]
                    neg_scope_notes.append(note)

                np.random.shuffle(neg_scope_notes)

                for i,item in enumerate(neg_scope_notes):
                    if i > n_samples-1:
                        break
                    self.text.append(t)
                    self.scope.append(item)
                    self.label.append(0)
                    self.mask_list.append(mask)
            else:
                self.scope.append(s)
                self.label.append(id2idx[m_id])

    def __getitem__(self, idx):
        item_t = self.text[idx]
        item_t_emb = self.t_emb[idx]
        item_s = self.scope[idx]
        item_l = torch.tensor(self.label[idx])
        # pad the mask tensor
        item_m = self.mask_list[idx]

        return item_t, item_t_emb, item_s, item_l, item_m

    def __len__(self):
        return len(self.text)

def minibatch_similarity(mentions, labels, pred_index, char_dict, batch_size=1, idx2id_dict=None, select=30):
    print('Here')
    x, y, target, key_list = [], [], [], []
    for idx, item in enumerate(mentions):
        key = idx2id_dict[labels[idx].item()]
        for name in char_dict[key]:
            x.append(item)
            y.append(torch.tensor(name))
            target.append(1)
            key_list.append(key)
        
        for pred_idx in pred_index[idx][0:select]:
            pred_id = idx2id_dict[pred_idx.item()]
            if pred_id != key and pred_id in char_dict:
                for name in char_dict[pred_id]:
                    x.append(item)
                    y.append(torch.tensor(name))
                    target.append(-1)
                    key_list.append(pred_id)
            # else:
            #     print(f'Pred key: {pred_id}, orig key: {key}')

    length_x = len(x)

    for idx in range(0, math.ceil(length_x/batch_size)):
        if length_x - batch_size < 0:
            updated_size = length_x
        else:
            updated_size = batch_size
        length_x = length_x - batch_size
        batch_x = x[idx*batch_size : idx*batch_size + updated_size]
        batch_y = y[idx*batch_size : idx*batch_size + updated_size]
        batch_t = torch.tensor(target[idx*batch_size : idx*batch_size + updated_size])
        batch_k = key_list[idx*batch_size : idx*batch_size + updated_size]

        batch_x = nn.utils.rnn.pad_sequence(batch_x, batch_first=True)
        batch_y = nn.utils.rnn.pad_sequence(batch_y, batch_first=True)


        yield batch_x, batch_y, batch_t, batch_k
    
def minibatch_sim_test(mentions, pred_labels, labels, char_dict, select=30):
    for idx, item in enumerate(mentions):
        key = labels[idx]
        # increase the minimum number of character to 10 by padding so that 2 layer CNN can atleast have 7 character. 
        if len(item) < 10:
            append_len = 10 - len(item)
            item = torch.cat((item, torch.tensor([0]*append_len)))
        x, y, pred_key_list= [], [], []
        for i in pred_labels[idx][0:select]:
            if i in char_dict: # avoid <unk> tag
                for j in char_dict[i]:
                    x.append(item)
                    y.append(torch.tensor(j))
                    pred_key_list.append(i)
        
        x = torch.stack(x, dim=0)
        y = nn.utils.rnn.pad_sequence(y, batch_first=True)

        yield x, y, pred_key_list, key


class Similarity_sequence(Dataset):
    def __init__(self, mentions, labels, pred_index, char_dict, idx2id_dict, select=10):
        self.mentions = mentions
        self.labels = labels
        self.pred_index = pred_index
        self.char_dict = char_dict
        self.idx2id = idx2id_dict

        self.x, self.y, self.target, self.key_list = [], [], [], []
        for idx, item in enumerate(self.mentions):
            key = idx2id_dict[labels[idx].item()]
            count_pos = 0
            for name in char_dict[key]:
                self.x.append(item)
                self.y.append(torch.tensor(name))
                self.target.append(1)
                self.key_list.append(key)
                count_pos += 1
            
            prob_y = []
            prob_keys = []
            for pred_idx in pred_index[idx][0:select]:
                pred_id = idx2id_dict[pred_idx.item()]
                if pred_id != key and pred_id in char_dict:
                    for name in char_dict[pred_id]:
                        prob_y.append(torch.tensor(name))
                        prob_keys.append(pred_id)

            Shuffle_lists(prob_y, prob_keys)
            for i in range(0,count_pos-1):
                if i == len(prob_y):
                    break
                self.x.append(item)
                self.y.append(prob_y[i])
                self.target.append(-1)
                self.key_list.append(prob_keys[i])


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.target[idx], self.key_list[idx]

def padding_similarity(batch):
    batch_x, batch_y, batch_t, batch_k = [], [], [], []
    for item in batch:
        batch_x.append(item[0])
        batch_y.append(item[1])
        batch_t.append(item[2])
        batch_k.append(item[3])

    batch_x = nn.utils.rnn.pad_sequence(batch_x, batch_first=True)
    batch_y = nn.utils.rnn.pad_sequence(batch_y, batch_first=True)
    batch_t = torch.tensor(batch_t)

    return batch_x, batch_y, batch_t, batch_k
    