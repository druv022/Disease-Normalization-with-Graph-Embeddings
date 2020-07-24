from MTL.MTL_training_EL import test, annotated_docs_to_tokens, get_elmo_representation
import os
from config.paths import Paths
from config.params import *
from nerds.input.brat import BratInput
from nerds.dataset.split import split_annotated_documents
import pickle
import torch
from allennlp.modules.elmo import Elmo

base_path = '/media/druv022/Data2/Final'
paths = Paths(base_folder = base_path)
paths.reset()
paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_35')
paths.ner_model_name = 'ner_model.pt'
paths.el_model_name = 'el_model.pt'
paths.mt_model_name = 'shared_model.pt'
paths.linear_model_name = 'linear_model.pt'

params = MultiTask_Params()
params.batch_first = False

X_test = BratInput(paths.test)
X_test = X_test.transform()
X_test = split_annotated_documents(X_test)

x_test_text, ner_test_tags, x_test_tokens = annotated_docs_to_tokens(X_test)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(os.path.join(paths.multitask_folder,'text_test_elmo_split.pkl')):
    # elmo embeddings
    options_file = paths.elmo_options
    weight_file = paths.elmo_weights
    ELMO_folder = paths.elmo_folder
    elmo_dim = params.elmo_dim
    elmo = Elmo(options_file, weight_file, 2,dropout=0)
    elmo.to(device)
    with torch.no_grad():
        text_test = get_elmo_representation(x_test_text, elmo, elmo_dim=params.elmo_dim, device=device)
    with open(os.path.join(paths.multitask_folder,'text_test_elmo_split.pkl'),'wb+') as f:
        pickle.dump(text_test, f)
else:
    with open(os.path.join(paths.multitask_folder,'text_test_elmo_split.pkl'),'rb+') as f:
        text_test = pickle.load(f)

test(paths, params, X_test, x_test_tokens, text_test, ner_test_tags, device=device)