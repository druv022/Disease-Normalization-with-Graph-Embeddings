from config.paths import Paths
from config.params import *
import os
from MTL.MTL_training_EL import main

base_path = '/media/druv022/Data2/Final'
paths = Paths(base_path)

start_from = 5

# if start_from < 2:
#     paths.reset()
#     paths.ner_model_name = 'ner_model.pt'
#     paths.el_model_name = 'el_model.pt'
#     paths.mt_model_name = 'shared_model.pt'
#     paths.linear_model_name = 'linear_model.pt'
#     paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_28')
#     if not os.path.exists(paths.experiment_folder):
#         os.makedirs(paths.experiment_folder)
#     if not os.path.exists(paths.multitask_folder):
#         os.makedirs(paths.multitask_folder)

#     params = MultiTask_Params()
#     params.batch_first = False
#     params.ner.use_word_self_attention = True
#     params.batch_size = 32
#     params.use_activation = False
#     params.num_epochs = 500

#     main(paths, params)

# if start_from < 3:
#     paths.reset()
#     paths.ner_model_name = 'ner_model.pt'
#     paths.el_model_name = 'el_model.pt'
#     paths.mt_model_name = 'shared_model.pt'
#     paths.linear_model_name = 'linear_model.pt'
#     paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_29')
#     if not os.path.exists(paths.experiment_folder):
#         os.makedirs(paths.experiment_folder)
#     if not os.path.exists(paths.multitask_folder):
#         os.makedirs(paths.multitask_folder)

#     params = MultiTask_Params()
#     params.batch_first = False
#     params.ner.use_word_self_attention = True
#     params.batch_size = 32
#     params.use_activation = False
#     params.num_epochs = 500
#     params.NER_type1 = False # change

#     main(paths, params)

# if start_from < 4:
#     paths.reset()
#     paths.ner_model_name = 'ner_model.pt'
#     paths.el_model_name = 'el_model.pt'
#     paths.mt_model_name = 'shared_model.pt'
#     paths.linear_model_name = 'linear_model.pt'
#     paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_30')
#     if not os.path.exists(paths.experiment_folder):
#         os.makedirs(paths.experiment_folder)
#     if not os.path.exists(paths.multitask_folder):
#         os.makedirs(paths.multitask_folder)

#     params = MultiTask_Params()
#     params.batch_first = False
#     params.ner.use_word_self_attention = True
#     params.batch_size = 32
#     params.use_activation = False
#     params.only_NER = True # change

#     params.num_epochs = 500

#     main(paths, params)

# if start_from < 5:
#     paths.reset()
#     paths.ner_model_name = 'ner_model.pt'
#     paths.el_model_name = 'el_model.pt'
#     paths.mt_model_name = 'shared_model.pt'
#     paths.linear_model_name = 'linear_model.pt'
#     paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_31')
#     if not os.path.exists(paths.experiment_folder):
#         os.makedirs(paths.experiment_folder)
#     if not os.path.exists(paths.multitask_folder):
#         os.makedirs(paths.multitask_folder)

#     params = MultiTask_Params()
#     params.batch_first = False
#     params.ner.use_word_self_attention = True
#     params.batch_size = 32
#     params.use_activation = False
#     params.only_EL = True # change
#     params.num_epochs = 500

#     main(paths, params)

if start_from < 6:
    paths.reset()
    paths.ner_model_name = 'ner_model.pt'
    paths.el_model_name = 'el_model.pt'
    paths.mt_model_name = 'shared_model.pt'
    paths.linear_model_name = 'linear_model.pt'
    paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_32','1')
    if not os.path.exists(paths.experiment_folder):
        os.makedirs(paths.experiment_folder)
    if not os.path.exists(paths.multitask_folder):
        os.makedirs(paths.multitask_folder)

    params = MultiTask_Params()
    params.reset()
    params.randomize = False
    params.batch_first = False
    params.ner.use_word_self_attention = True
    params.batch_size = 32
    params.use_activation = True
    params.activation = 'fusedmax'
    params.num_epochs = 500

    main(paths, params)

# if start_from < 7:
#     paths.reset()
#     paths.ner_model_name = 'ner_model.pt'
#     paths.el_model_name = 'el_model.pt'
#     paths.mt_model_name = 'shared_model.pt'
#     paths.linear_model_name = 'linear_model.pt'
#     paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_33')
#     if not os.path.exists(paths.experiment_folder):
#         os.makedirs(paths.experiment_folder)
#     if not os.path.exists(paths.multitask_folder):
#         os.makedirs(paths.multitask_folder)

#     params = MultiTask_Params()
#     params.batch_first = False
#     params.ner.use_word_self_attention = True
#     params.batch_size = 32
#     params.use_activation = True
#     params.activation = 'sparsemax'
#     params.num_epochs = 500

#     main(paths, params)

# if start_from < 8:
#     paths.reset()
#     paths.ner_model_name = 'ner_model.pt'
#     paths.el_model_name = 'el_model.pt'
#     paths.mt_model_name = 'shared_model.pt'
#     paths.linear_model_name = 'linear_model.pt'
#     paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_34')
#     if not os.path.exists(paths.experiment_folder):
#         os.makedirs(paths.experiment_folder)
#     if not os.path.exists(paths.multitask_folder):
#         os.makedirs(paths.multitask_folder)

#     params = MultiTask_Params()
#     params.batch_first = False
#     params.ner.use_word_self_attention = True
#     params.batch_size = 32
#     params.use_activation = True
#     params.activation = 'oscarmax'
#     params.num_epochs = 500

#     main(paths, params)

# if start_from < 8:
#     paths.reset()
#     paths.ner_model_name = 'ner_model.pt'
#     paths.el_model_name = 'el_model.pt'
#     paths.mt_model_name = 'shared_model.pt'
#     paths.linear_model_name = 'linear_model.pt'
#     paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_35')
#     if not os.path.exists(paths.experiment_folder):
#         os.makedirs(paths.experiment_folder)
#     if not os.path.exists(paths.multitask_folder):
#         os.makedirs(paths.multitask_folder)

#     params = MultiTask_Params()
#     params.reset()
#     params.randomize = False
#     params.batch_first = False
#     params.ner.use_word_self_attention = True
#     params.batch_size = 32
#     params.use_activation = True
#     params.activation = 'softmax'
#     params.num_epochs = 500

#     main(paths, params)