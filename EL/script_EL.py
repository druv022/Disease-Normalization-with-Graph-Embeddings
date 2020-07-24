from nerds.ner.pytorch_druv import PytorchNERModel
from nerds.input.brat import BratInput
from nerds.doc.annotation import Annotation
from nerds.evaluate.score import annotation_precision_recall_f1score
from nerds.doc.bio import transform_annotated_document_to_bio_format
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from embeddings import load_embedding_pkl
from process.read_c2m import *
from process.read_ctd import *
import copy
from gensim.models import KeyedVectors
from node2vec.node2vec1 import *
from node2vec.node2vec2 import *
import math
from sklearn.metrics import classification_report, accuracy_score
# import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from sklearn.utils import shuffle as Shuffle_lists



# TODO: handle false positive NER case
def get_formatted_data(x_data, pred_data, tags, weights, device='cpu'):
    entity_list = []
    entity_emb_list = []
    for doc_idx, doc in enumerate(x_data):
        # if doc_idx == 308:
        #     print('Here')
        pred_doc = pred_data[doc_idx]
        pred_tag = tags[doc_idx]
        pred_weight = weights[doc_idx]
        max_sent_length = len(pred_tag)

        mask = torch.tensor([0 if i is 'O' else 1 for i in pred_tag], dtype=torch.float, device=device)
        mask = mask.unsqueeze(1).expand(-1,pred_weight.shape[1])
        # consider only the attention weight of the sentence and keep only the entity weights
        pred_weight = pred_weight[0:max_sent_length] * mask
        
        tokens, bio_labels = transform_annotated_document_to_bio_format(doc)
        update = False
        emb = []
        for lbl_idx, label in enumerate(pred_tag):
            # check if 'Begin' label matches for true and predicted tag
            if 'B_' in label and 'B_' in bio_labels[lbl_idx]:
                # update if full entiy span is found (single or multi word)
                if update:
                    update = False
                    emb = np.asarray(emb)
                    # take average pooling of the (single or) multi word entity
                    entity_emb_list.append(np.mean(emb, axis=0))
                    emb = []
                # check if next tags of true and predicted tag matches; needed for complete 'BI' sequence
                if lbl_idx+1 < max_sent_length and pred_tag[lbl_idx+1]!=bio_labels[lbl_idx+1]:
                    continue
                update = True
                emb.append(pred_weight[lbl_idx].cpu().detach().numpy())
            # check if 'Inside' label matches for true and predicted tag
            elif 'I_' in label and 'I_' in bio_labels[lbl_idx]:
                # check if next tags of true and predicted tag matches; needed for complete 'BI' sequence
                if lbl_idx+1 < max_sent_length and pred_tag[lbl_idx+1]!=bio_labels[lbl_idx+1]:
                    update = False
                    continue
                # add only if 'Begin' tag matches
                if update:
                    emb.append(pred_weight[lbl_idx].cpu().detach().numpy())

            if 'O' in label or lbl_idx == max_sent_length - 1:
                # update if full entiy span is found (single or multi word)
                if update:
                    update = False
                    emb = np.asarray(emb)
                    # take average pooling of the (single or) multi word entity
                    entity_emb_list.append(np.mean(emb, axis=0))
                    emb = []
            
        # find the corresponding normalized concept in true annotated docs
        for annotation in doc.annotations:
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
                    for norm in doc.normalizations:
                        if norm.argument_id == ann_identifier:
                            entity_list.append(norm.preferred_term)
                            complete_flag = True
                            break
                if complete_flag:
                    break
                ann_counter += 1

        assert(len(entity_list) == len(entity_emb_list)), 'Entity ID and embedding mismatch'

    return entity_list, entity_emb_list

def filter_entity(entity_list, embedding_list, file1, file2):
    omim_dict = read_ctd(file1)
    c2m_dict = read_c2m(file2)

    new_entity_list = []
    new_embedding_list = []
    for i,item in enumerate(entity_list):
        item = item.split('|')[0]
        item = item.split('+')[0]

        if 'D' not in item:
            if item in omim_dict:
                item = omim_dict[item][0]
            if item in c2m_dict:
                item = c2m_dict[item][0]
        
        if 'D' in item:
            new_entity_list.append(item)
            new_embedding_list.append(embedding_list[i])

    print('Number of items removed: ',len(entity_list)-len(new_entity_list))

    return new_entity_list, new_embedding_list


class EntityModel(nn.Module):

    def __init__(self, entity, in_features, out_features, hidden=250):
        super(EntityModel, self).__init__()

        self.entity = entity
        self.fc1 = nn.Linear(in_features, out_features, bias=True)
        self.fc2 = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        x = self.fc1(x)
        x = self.fc2(nn.functional.relu(x))
        entity = self.entity
        entity = entity.transpose(-2,-1).unsqueeze(0).repeat([batch_size,1,1])
        x = torch.bmm(x, entity)
        return x.squeeze()

def minibatch1(x, y=None, preprocess=None, batch_size=1):

    if y is not None:
        x, y = Shuffle_lists(x, y)
        y = torch.tensor(preprocess(y))
    else:
        x = np.random.shuffle(x)

    len_x = len(x)
    x = torch.tensor(x)
        
    for idx in range(0, math.ceil(len_x/batch_size)):
        if len_x - batch_size < 0:
            updated_size = len_x
        else:
            updated_size = batch_size
        len_x = len_x - batch_size
        batch_x = x[idx*batch_size: idx*batch_size+updated_size]
        
        if y is not None:
            batch_y = y[idx*batch_size:idx*batch_size+updated_size]
        else:
            batch_y = y
        yield (batch_x, batch_y)


class TrainingData1(Dataset):

    def __init__(self, x, y, preprocess=None):
        self.x = torch.tensor(x)
        self.y = torch.tensor(preprocess(y))

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def __len__(self):
        return len(self.x)

def get_EL_data(x, ctd_file, c2m_file, device=None):
    annotated_documents, predictions, att_weights_list = NER_model.predict_for_EL(x)
    entity_list, embedding_list = get_formatted_data(x, annotated_documents, predictions, att_weights_list, device=device)
    return filter_entity(entity_list, embedding_list, ctd_file, c2m_file)


def train1():
    epochs = 1000
    batch_size = 64
    model = EntityModel(entity_emb, embedding_size, embedding_size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=0.001)
    training_set = TrainingData1(embedding_train, entity_train, preprocess=vocab.doc2id)
    training_generator= DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=8)

    validation_set = TrainingData1(embedding_valid, entity_valid, preprocess=vocab.doc2id)
    validation_generator= DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=8)

    for epoch in range(epochs):
        training_loss = 0
        count = 0
        y_true = []
        pred = []
        start_time = time.time()
        model.train()
        # for x, y in minibatch1(embedding_train, entity_train, preprocess=vocab.doc2id, batch_size=batch_size):
        for x, y in training_generator:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            x_ = model(x)

            loss = nn.functional.cross_entropy(x_, y)

            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            count+=1

            with torch.no_grad():
                x_ = model(x)
                x_ = nn.functional.softmax(x_, dim=-1)
                _, max_idx = torch.max(x_, dim=-1)
                pred.extend(max_idx.cpu().numpy().tolist())
                y_true.extend(y.cpu().numpy().tolist())

        writer.add_scalars('loss', {'loss': training_loss/count}, epoch)
        writer.add_histogram('grad', optimizer.param_groups[0]['params'][0].grad, epoch )
        print(f'Epoch:{epoch}\tLoss: {training_loss/count}\tTime: {time.time()-start_time}')
        print('Train:',accuracy_score(y_true, pred))
        
        pred = []
        model.eval()
        # for x, y in minibatch1(embedding_valid, entity_valid, preprocess=vocab.doc2id, batch_size=batch_size):
        for x, y in validation_generator:
            x, y = x.to(device), y.to(device)
            x = model(x)

            x = nn.functional.softmax(x, dim=-1)
            _, max_idx = torch.max(x, dim=-1)
            pred.extend(max_idx.cpu().numpy().tolist())

        print('Valid:',accuracy_score(vocab.doc2id(entity_valid), pred))
        # print(classification_report(vocab.doc2id(entity_valid), pred))

class Linear(nn.Module):

    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(in_features, out_features)

    def forward(self, x, y, z):
        x = self.fc1(x)
        y = self.fc2(y)
        z = self.fc2(z)

        return x, y, z


class TrainingData2(Dataset):

    def __init__(self, x, y, sample_list, entity_emb, n_samples=5, preprocess=None):
        self.x = torch.repeat_interleave(torch.tensor(x), n_samples, dim=0)
        self.y = torch.repeat_interleave(torch.tensor(preprocess(y)), n_samples, dim=0)
        self.z = torch.tensor(sample_list)
        self.entity_emb = entity_emb

        self.total = len(self.x)
        self.count = 0

    def __getitem__(self, idx):
        self.count += 1
        if self.count == self.total:
            self.count = 0
            np.random.shuffle(self.z)
            
        return (self.x[idx], self.entity_emb[self.y[idx]], self.entity_emb[self.z[idx]])

    def __len__(self):
        return len(self.x)


def calculate_loss(x, y , z):
    # y, z = entity_emb[y], entity_emb[z]
    # dp = nn.functional.pairwise_distance(x, y)
    # dn = nn.functional.pairwise_distance(x, z)

    dp = nn.functional.cosine_similarity(x, y)
    dn = nn.functional.cosine_similarity(x, z)

    return torch.mean(1+dp-dn)

def minibatch2(x, y, z, entity_emb, preprocess=None, n_samples=5, batch_size=1):
    x, y = Shuffle_lists(x, y)
    x = torch.repeat_interleave(torch.tensor(x), n_samples, dim=0)
    y = torch.repeat_interleave(torch.tensor(preprocess(y)), n_samples, dim=0)
    np.random.shuffle(z)
    z = torch.tensor(z)
    len_x = len(x)

    for idx in range(0, math.ceil(len_x/batch_size)):
        if len_x - batch_size < 0:
            updated_size = len_x
        else:
            updated_size = batch_size
        len_x = len_x - batch_size

        batch_x = x[idx*batch_size: idx*batch_size+updated_size]
        batch_y = entity_emb[y[idx*batch_size: idx*batch_size+updated_size]]
        batch_z = entity_emb[z[idx*batch_size: idx*batch_size+updated_size]]

        yield(batch_x, batch_y, batch_z)

def minimum_distance(x, entity_emb, device='cpu'):
    pred = []
    x = torch.tensor(x, device=device)
    # remove <pad> and <unk>
    entity_emb = entity_emb[1:-2]
    for item in x:
        item = item.unsqueeze(0)
        item = item.repeat(entity_emb.shape[0], 1)

        dis = nn.functional.pairwise_distance(item, entity_emb)
        _,min_idx = torch.min(dis, 0)
        pred.append(min_idx+1)

    return pred 


def train2():
    epochs = 100
    batch_size = 32
    n_samples = 32
    sampling_table = negative_sampling_table(vocab.token_counter(), transform=vocab.token_to_id)

    model = Linear(in_features=200, out_features=200)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    training_set = TrainingData2(embedding_train, entity_train, sampling_table, entity_emb, n_samples=n_samples, preprocess=vocab.doc2id)
    training_generator= DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=8)

    validation_set = TrainingData2(embedding_valid, entity_valid, sampling_table, entity_emb, preprocess=vocab.doc2id)
    validation_generator= DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=8)

    for epoch in range(epochs):
        training_loss = 0
        count = 0
        y_true = []
        pred = []
        start_time = time.time()
        model.train()
        for x, y, z in minibatch2(embedding_train, entity_train, sampling_table, entity_emb, preprocess=vocab.doc2id, n_samples=n_samples, batch_size=batch_size):
        # for x, y, z in training_generator:
            x, y, z = x.to(device), y.to(device), z.to(device)
            optimizer.zero_grad()

            x, y, z = model(x, y, z)

            loss = calculate_loss(x, y, z)

            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            count+=1

        with torch.no_grad():
            pred = minimum_distance(embedding_train, entity_emb, device=device)

        print(f'Epoch:{epoch}\tLoss: {training_loss/count}\tTime: {time.time()-start_time}')
        print('Train:',accuracy_score(entity_train, vocab.id2doc(pred)))
        
        # pred = []
        # model.eval()
        # for x, y, z in minibatch2(embedding_valid, entity_valid, sampling_table, preprocess=vocab.doc2id, batch_size=batch_size):
        # # for x, y, z in validation_generator:
        #     x, y = x.to(device), y.to(device)
        #     x = model(x)


        # print('Valid:',accuracy_score(vocab.doc2id(entity_valid), pred))
        # print(classification_report(vocab.doc2id(entity_valid), pred))


    print('Here')






# TODO:
# def get_elmo_data(x, ctd_file, c2m_file, device=None):
#     annotated_documents, predictions, att_weights_list = NER_model.predict_for_EL(x)


if __name__ == '__main__':
    # entity_names = ['Disease']
    # entity_names = ['B_Disease','I_Disease']
    # entity_names = ['SpecificDisease', 'Modifier', 'DiseaseClass', 'Composite Mention']
    # entity_names = ['B-SpecificDisease','I-SpecificDisease', 'B-Modifier', 'I-Modifier','B-DiseaseClass', 'I-DiseaseClass', 'B-Composite Mention', 'I-Composite Mention']

    path_to_train_input = r'/media/druv022/Data1/Masters/Thesis/Data/Converted_train_2'
    path_to_valid_input = r'/media/druv022/Data1/Masters/Thesis/Data/Converted_develop'
    path_to_test= r'/media/druv022/Data1/Masters/Thesis/Data/Converted_test'
    path_to_embeddings = r'/media/druv022/Data1/Masters/Thesis/Data/Embeddings'
    ctd_file = r'/media/druv022/Data1/Masters/Thesis/Data/CTD/CTD_diseases.csv'
    c2m_file = r'/media/druv022/Data1/Masters/Thesis/Data/C2M/C2M_mesh.txt'

    path_to_data = '/media/druv022/Data1/Masters/Thesis/Data/Experiment'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    X = BratInput(path_to_train_input)
    X = X.transform()
    # X = split_annotated_documents(X)

    X_valid = BratInput(path_to_valid_input)
    X_valid = X_valid.transform()
    # X_valid = split_annotated_documents(X_valid)

    X_test = BratInput(path_to_test)
    X_test = X_test.transform()

    torch.manual_seed(5)
    random.seed(5)
    np.random.seed(5)

    entity_names = ['B_Disease','I_Disease']
    embeddings =  load_embedding_pkl(path_to_embeddings)

    NER_model = PytorchNERModel(word_embeddings=embeddings)
    NER_model.load('tmp','trained_model_20190502-134822.pkl')
    embedding_size = 200
    
    # training set
    entity_train, embedding_train = get_EL_data(X, ctd_file, c2m_file, device=device)
    entity_valid, embedding_valid = get_EL_data(X_valid, ctd_file, c2m_file, device=device)

    # node2vec1
    vocab_path = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/node2vec1/processed_en.pkl'
    embedding_path = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/node2vec1/embedding'

    # # node2vec2
    # vocab_path = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/node2vec2/processed_en.pkl'
    # embedding_path = r'/media/druv022/Data1/Masters/Thesis/Data/MeSH/node2vec2/embedding'

    vocab, wv = get_node2vec(vocab_path, embedding_path)
    entity_emb = torch.tensor(wv.vectors, device=device)
    # entity_list = torch.tensor(vocab.doc2id(entity_list), device=device)
    # embedding_tensor = torch.tensor(embedding_list, device=device)

    # plt.subplot(121)
    # plt.hist(wv.vectors, bins=10, label='Mesh embeddings Node2vec1')
    # plt.subplot(122)
    # plt.hist(np.stack(embedding_train), bins=10, label='Training embeddings')
    # plt.show()


    # train with cross entropy loss
    # train1()

    # train with max margin loss
    train2()


        

    # # # validation set
    # # annotated_documents, predictions, att_weights_list = NER_model.predict_for_EL(X_valid)
    # # p, r, f1 = annotation_precision_recall_f1score(annotated_documents, X_valid)
    # # print("NER Disease:\tPrecision: ", p, "\tRecall: " ,r, "\tF-score: ", f1)

    # # entity_list, embedding_list = get_formatted_data(X_valid, annotated_documents, predictions, att_weights_list, device=device)

    # # entity_list, embedding_list = filter_entity(entity_list, embedding_list, ctd_file, c2m_file)

    # print('Boom')