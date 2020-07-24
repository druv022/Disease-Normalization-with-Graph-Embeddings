from config.params import *
from config.paths import *
from nerds.ner.pytorch_druv import PytorchNERModel as NER_word2vec
from nerds.ner.pytorch_druv_elmo import PytorchNERModel as NER_elmo
from NER.test_nerds3 import test_BiLSTM
from nerds.input.brat import BratInput
from nerds.dataset.split import split_annotated_documents
import os
from NER.NER_variational_dropout import test, load
from utils.plot_heatmaps import plot_heatmap 

base_path = '/media/druv022/Data2/Final'
paths = Paths(base_folder=base_path)


X_test = BratInput(paths.test)
X_test = X_test.transform()
X_test = split_annotated_documents(X_test)

#-------------- TEST Baseline + ELMO --------------------------------------
# path to stored file
paths.experiment_folder = '/media/druv022/Data2/Final/Data/Experiment/EX_17/2'
model_name = 'trained_model_20190701-085415.pkl'
# word2vec embeddings
# model = NER_word2vec().load(paths.experiment_folder, model_name)
# ELMo embeddings
model = NER_elmo().load(paths.experiment_folder, model_name)

# Update experiment folder to store BRAT output
paths.experiment_folder = '/media/druv022/Data2/Final/Data/Experiment/EX_17/2/Brat'
if not os.path.exists(paths.experiment_folder):
    os.makedirs(paths.experiment_folder)

tokens, att_weights = test_BiLSTM(model, X_test, file_path=paths.experiment_folder)

#---------------- TEST Variational dropout -----------------------------------
paths.experiment_folder = '/media/druv022/Data2/Final/Data/Experiment/EX_22/1'
model_name = 'Training_model_20190709-111814.pkl'
model, [params, ner_labels_vocab] = load(paths, model_name)

paths.experiment_folder = '/media/druv022/Data2/Final/Data/Experiment/EX_22/1/Brat'
if not os.path.exists(paths.experiment_folder):
    os.makedirs(paths.experiment_folder)

tokens, att_weights = test(paths, params, X_test, model, ner_labels_vocab, file_path=paths.experiment_folder)

paths.experiment_folder = '/media/druv022/Data2/Final/Data/Experiment/EX_17/2/Heatmaps'
if not os.path.exists(paths.experiment_folder):
    os.makedirs(paths.experiment_folder)
for idx,item in enumerate(tokens):
    if item[-1] == '.':
        item.pop()
    weights = att_weights[idx][0:len(item),0:len(item)]
    paths.experiment_folder = '/media/druv022/Data2/Final/Data/Experiment/EX_17/2/Heatmaps/Heatmaps_'+str(idx)+'.png'
    plot_heatmap(item, weights, file_path=paths.experiment_folder)