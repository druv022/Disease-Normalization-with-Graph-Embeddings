from nerds.ner.pytorch_druv import PytorchNERModel as NER_word2vec
from nerds.ner.pytorch_druv_elmo import PytorchNERModel as NER_ELMO
from nerds.input.brat import BratInput
from nerds.output.brat import BratOutput
from nerds.doc.annotation import Annotation
from nerds.util.nlp import text_to_tokens
from nerds.ner.crf import CRF
from nerds.ner.spacy import SpaCyStatisticalNER
from nerds.dataset.split import split_annotated_documents
from nerds.dataset.merge import merge_documents
# from nerds.ner.ensemble import (
#     NERModelEnsembleMajorityVote, NERModelEnsemblePooling, NERModelEnsembleWeightedVote)
import random
from nerds.evaluate.score import annotation_precision_recall_f1score
from utils.embeddings import load_embedding_pkl

import numpy as np
import re
import os
import copy
import time
import torch
from config.paths import *
from config.params import *


def test_BiLSTM(model, X_test, file_path=''):
    """ Test the model
    
    Arguments:
        model {obj} -- Model object
        X_test {list} -- list of annotated documents
    
    Keyword Arguments:
        file_path {str} -- path with filename to store resutls in BRAT format  (default: {''})
    """
    X_pred,[tokens, att_weights] = model.transform(X_test)
    name=None

    p, r, f = annotation_precision_recall_f1score(X_pred, X_test, ann_label=name)

    if not name:
        name = ''
    print("Disease: "+name+"\tPrecision: ", p, "\tRecall: " ,r, "\tF-score: ", f)
    
    brat = BratOutput(file_path)
    X_pred = merge_documents(X_pred)
    brat.transform(X_pred)

    return [tokens, att_weights]
    

def train_BiLSTM(X, params, paths, X_valid=None, embeddings=None, model=None):
    """ Train the model
    
    Arguments:
        X {list} -- List of annotated documents
        params {obj} -- parameters
        paths {obj} -- paths
    
    Keyword Arguments:
        X_valid {list} -- list of annotated documents (default: {None})
        embeddings {dict} -- dictionary of embeddings (key: vectors) (default: {None})
        model {obj} -- Model object (default: {None})
    
    Returns:
        obj -- Model object
    """
    if model is None:
        model = NER_word2vec(word_embeddings=embeddings)

    hparams_1 = {'use_char_emb': params.use_char,'use_char_cnn': params.use_char_cnn, 'use_char_attention':params.use_char_attention, 
            'use_word_self_attention': params.use_word_self_attention, 'use_EL': True, 'shuffle':True}
    hparams_2 = {'optimizer': 'adam','lr': params.lr, 'embedding_path': paths.pre_trained_embeddings, 'save_best':True, 'use_GRU':True, 'file_path':paths.experiment_folder,'output':params.output}

    model.fit(X, X_valid=X_valid, char_emb_size=params.char_emb_size, word_emb_size=params.word_emb_size,
                char_lstm_units=params.char_rnn_units, word_lstm_units=params.word_rnn_units, dropout=params.ner_dropout,
                batch_size=params.batch_size, num_epochs=params.num_epochs, use_char_emb=params.use_char, use_crf=True, hparams_1=hparams_1, hparams_2=hparams_2)

    print("Complete")

    return model

def train_BiLSTM_elmo(X, params, paths, X_valid=None, model=None):
    """Train ELMo embedding based NER model
    
    Arguments:
        X {list} -- List of annotated documents
        params {obj} -- parameters
        paths {obj} -- paths
    
    Keyword Arguments:
        X_valid {list} -- list of annotated documents (default: {None})
        model {obj} -- Model object (default: {None})
    
    Returns:
        obj -- Model object
    """
    if model is None:
        model = NER_ELMO()

    hparams_1 = {'use_char_emb': True, 'use_word_self_attention': params.use_word_self_attention, 'use_EL': True, 'shuffle':True}
    hparams_2 = {'optimizer': 'adam','lr':params.ner_lr, 'embedding_path': r'/media/druv022/Data1/Masters/Thesis/Data/Embeddings', 'save_best':True, 'use_GRU':True, 'output':params.output, 
                'options_file':paths.elmo_options, 
                'weight_file':paths.elmo_weights, 
                'ELMO_folder':paths.elmo_folder, 'file_path':paths.experiment_folder}

    model.fit(X, X_valid=X_valid, word_emb_size=params.word_emb_size, word_lstm_units=params.word_rnn_units, dropout=params.ner_dropout,
                batch_size=params.batch_size, num_epochs=params.num_epochs, use_crf=True, hparams_1=hparams_1, hparams_2=hparams_2)

    print("Complete")

    return model


def train_CRF(X, model=None):
    """Train CRF based model
    
    Arguments:
        X {list} -- List of annotated documents
    
    Keyword Arguments:
        model {obj} -- Model object (default: {None})
    
    Returns:
        obj -- Model object
    """
    if model is None:
        model = CRF()
    model.fit(X)

    model.save("tmp")
    return model

def train_spacy(X, model=None):
    """Train spacy based model
    
    Arguments:
        X {[type]} -- [description]
    
    Keyword Arguments:
        model {obj} -- Model object (default: {None})
    
    Returns:
        obj -- Model object
    """
    if model is None:
        model = SpaCyStatisticalNER()
    model.fit(X)

    # model.save("tmp")
    return model

def test(model, params, X_test, test_name = ''):
    """Test model
    
    Arguments:
        model {obj} -- Model object
        params {obj} -- Parameters
        X_test {obj} -- List of annotated documents
    
    Keyword Arguments:
        test_name {str} -- Type of model/test (default: {''})
    """
    X_pred = model.transform(X_test)

    for l in params.entity_names:
        p, r, f = annotation_precision_recall_f1score(X_pred,
                                                     X_test)
        print(test_name+" Label: ", 'Disease', "\tPrecision: ", p, "\tRecall: " ,r, "\tF-score: ", f)

def main(paths, params):
    """    
    Arguments:
        paths {obj} -- Paths 
        params {obj} -- Parameters
    """
    path_to_train_input = paths.training #'/media/druv022/Data1/Masters/Thesis/Data/Converted_train_2'
    path_to_valid_input =  paths.develop #'/media/druv022/Data1/Masters/Thesis/Data/Converted_develop'
    path_to_embeddings = paths.pre_trained_embeddings # '/media/druv022/Data1/Masters/Thesis/Data/Embeddings'

    path_to_data = paths.experiment_folder #'/media/druv022/Data1/Masters/Thesis/Data/Experiment'

    X = BratInput(path_to_train_input)
    X = X.transform()
    if params.use_sent_split:
        X = split_annotated_documents(X)

    X_valid = BratInput(path_to_valid_input)
    X_valid = X_valid.transform()
    if params.use_sent_split:
        X_valid = split_annotated_documents(X_valid)

    if not params.randomize:
        torch.manual_seed(5)
        random.seed(5)
        np.random.seed(5)

    # random.Random(5).shuffle(X)
    timestr = params.time

    # print('######################################################################################################')
    # print('                            Test BiLSTM-CRF ')
    # print('######################################################################################################')   

    # embedding_dict=None

    if params.use_elmo:
        biLSTM_model = NER_ELMO()

        biLSTM_model = train_BiLSTM_elmo(X, params, paths, X_valid=X_valid, model=biLSTM_model)
    else:
        embedding_dict =  load_embedding_pkl(path_to_embeddings)
        biLSTM_model = NER_word2vec(word_embeddings=embedding_dict)

        biLSTM_model = train_BiLSTM(X, params, paths, X_valid=X_valid, model=biLSTM_model,embeddings=embedding_dict)

# if __name__ == '__main__':
    # entity_names = ['B_Disease','I_Disease']
    
    # paths = Paths()
    # params = Params()

    # path_to_train_input = paths.training #'/media/druv022/Data1/Masters/Thesis/Data/Converted_train_2'
    # path_to_valid_input =  paths.validation #'/media/druv022/Data1/Masters/Thesis/Data/Converted_develop'
    # path_to_test= paths.test #'/media/druv022/Data1/Masters/Thesis/Data/Converted_test'
    # path_to_embeddings = paths.pre_trained_embeddings # '/media/druv022/Data1/Masters/Thesis/Data/Embeddings'

    # path_to_data = paths.experiment_folder #'/media/druv022/Data1/Masters/Thesis/Data/Experiment'

    # X = BratInput(path_to_train_input)
    # X = X.transform()
    # X = split_annotated_documents(X)

    # X_valid = BratInput(path_to_valid_input)
    # X_valid = X_valid.transform()
    # X_valid = split_annotated_documents(X_valid)

    # X_test = BratInput(path_to_test)
    # X_test = X_test.transform()

    # torch.manual_seed(5)
    # random.seed(5)
    # np.random.seed(5)

    # random.Random(5).shuffle(X)
    # timestr = time.strftime("%Y%m%d-%H%M%S")

    # print('######################################################################################################')
    # print('                            Test BiLSTM-CRF ')
    # print('######################################################################################################')   
    # entity_names = ['B_Disease','I_Disease']
    # embedding_dict =  load_embedding_pkl(path_to_embeddings)
    # embedding_dict=None

    # word2vec
    # biLSTM_model = NER_word2vec(word_embeddings=embedding_dict)
    # ELMO
    # elmo_dim = 1024
    # biLSTM_model = NER_ELMO(word_embeddings=embedding_dict)

    # biLSTM_model = train_BiLSTM_elmo(X, X_valid=X_valid, model=biLSTM_model,embeddings=None, char_emb_size=30, word_emb_size=elmo_dim,
    #                 char_lstm_units=30, word_lstm_units=100, dropout=0.5, batch_size=32, num_epochs=100, use_char_cnn=True,
    #                 use_attention_char=False, use_word_self_attention=True)
    # biLSTM_model.load('tmp','trained_model_20190427-115757.pkl')

    # Test on validation set
    # print("Testing on validation set:")
    # test_BiLSTM(biLSTM_model, copy.deepcopy(X_valid), file_path=path_to_data)
    # Test on Test set
    # print("Testing on test set:")
    # test_BiLSTM(biLSTM_model, copy.deepcopy(X_test), file_path=path_to_data)
    #-----------------------------------------------------------------------------------------------------
    
    # print('######################################################################################################')
    # print('                            Test CRF ')
    # print('######################################################################################################')  
    # entity_names = ['Disease']
    # crf_model = CRF()
    # crf_model = train_CRF(X,model=crf_model)
    # crf_model.load("tmp")
    
    # # Test on validation set
    # print("Testing on validation set:")
    # test(crf_model, copy.deepcopy(X_valid),test_name='CRF')
    # # Test on test set
    # print("Testing on test set:")
    # test(crf_model, copy.deepcopy(X_test),test_name='CRF')
    # #-----------------------------------------------------------------------------------------------------
    
    # print('######################################################################################################')
    # print('                            Test Spacy ')
    # print('######################################################################################################')  
    # entity_names = ['Disease']
    # spacy_model = SpaCyStatisticalNER()
    # spacy_model = train_spacy(X,model=spacy_model)
    # # spacy_model.load("tmp") # Require bug fix with file save

    # # Test on validation set
    # print("Testing on validation set:")
    # test(spacy_model, copy.deepcopy(X_valid), test_name='Spacy')
    # # Test on test set
    # print("Testing on test set:")
    # test(spacy_model, copy.deepcopy(X_test), test_name='Spacy')
    # #----------------------------------------------------------------------------------------------------
    
    # print('######################################################################################################')
    # print('                            Ensemble')
    # print('######################################################################################################')  
    # models = [biLSTM_model, crf_model, spacy_model]
    # ens1 = NERModelEnsembleMajorityVote(models)
    # ens2 = NERModelEnsemblePooling(models)
    # ens3 = NERModelEnsembleWeightedVote(models)

    # X_pred_1 = ens1.transform(copy.deepcopy(X_valid))
    # print("Majority Vote: \n")
    # for l in entity_names:
    #     p, r, f = calculate_precision_recall_f1score(X_pred_1,
    #                                                  X_valid,
    #                                                  entity_label=l)
    #     print("Label: ", 'Disease', "\tPrecision: ", p, "\tRecall: " ,r, "\tF-score: ", f)

    # X_pred_2 = ens2.transform(copy.deepcopy(X_valid))
    # print("Pooling: \n")
    # for l in entity_names:
    #     p, r, f = calculate_precision_recall_f1score(X_pred_2,
    #                                                  X_valid,
    #                                                  entity_label=l)
    #     print("Label: ", 'Disease', "\tPrecision: ", p, "\tRecall: " ,r, "\tF-score: ", f)
    # X_pred_3 = ens3.transform(copy.deepcopy(X_valid))
    # print("Weighted Vote: \n")
    # for l in entity_names:
    #     p, r, f = calculate_precision_recall_f1score(X_pred_3,
    #                                                  X_valid,
    #                                                  entity_label=l)
    #     print("Label: ", 'Disease', "\tPrecision: ", p, "\tRecall: " ,r, "\tF-score: ", f)
