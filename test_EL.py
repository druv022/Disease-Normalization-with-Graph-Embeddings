from config.paths import Paths
from config.params import EL_params
from EL.EL_partA import test as test_node2vec
from nerds.input.brat import BratInput
from nerds.dataset.split import split_annotated_documents
from nerds.doc.bio import transform_annotated_documents_to_bio_format
import os
from EL.EL_GCN import test as test_GCN

# update base path
base_path = '/media/druv022/Data2/Final'
paths = Paths(base_path)
paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_27',str(4))

params = EL_params()
params.batch_size =1
params.use_elmo = True

X_test = BratInput(paths.test)
X_test = X_test.transform()
X_test = split_annotated_documents(X_test)

_, predictions_test = transform_annotated_documents_to_bio_format(X_test)
annotated_docs_test = X_test

# EL Node2Vec
# test_node2vec(paths, params, X_test, annotated_docs_test, predictions_test)
# EL GCN
test_GCN(paths, params, X_test, annotated_docs_test, predictions_test)
