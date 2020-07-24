from config.params import *
from config.paths import *
from NER.test_nerds3 import main as ner_main
from utils.embeddings import save_embedding_pkl
import subprocess
import os
# from NER.NER_variational_dropout import main as ner_vd # Varitional Dropout code cannot be shared

base_path = '/media/druv022/Data2/Final'
paths = Paths(base_folder=base_path)

start_from = 1


# # Step 1: convert all the files in BRAT format by using the script Convert_2.py
# # training set
# subprocess.call(['python','/media/druv022/Data1/Masters/Thesis/code_latest_june18/process/Convert_2.py','--file_name','/media/druv022/Data2/Final/Data/NCBItrainset_corpus.txt','--file_path',paths.training], shell=False)
# # develop set
# subprocess.call(['python','/media/druv022/Data1/Masters/Thesis/code_latest_june18/process/Convert_2.py','--file_name','/media/druv022/Data2/Final/Data/NCBIdevelopset_corpus.txt','--file_path',paths.develop], shell=False)
# # test set
# subprocess.call(['python','/media/druv022/Data1/Masters/Thesis/code_latest_june18/process/Convert_2.py','--file_name','/media/druv022/Data2/Final/Data/NCBItestset_corpus.txt','--file_path', paths.test], shell=False)
# print('All files converted to BRAT format')

# # Step 2: re-format the pre-trained embeddings in gensim word2vec
# save_embedding_pkl(paths.pre_trained_embeddings)


# Experiment 1
if start_from < 2:
    print("Experiment 1")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.use_char_cnn = False
    params.randomize = True
    params.use_sent_split = True

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_1',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)


# Experiment 2
if start_from < 3:
    print("Experiment 2")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.use_char_cnn = False
    params.randomize = True
    params.use_sent_split = False

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_2',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)


if start_from < 4:
    print("Experiment 3")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.use_char_cnn = False
    params.randomize = True
    params.use_sent_split = True

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_3',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)

if start_from < 5:
    print("Experiment 4")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 150
    params.ner_dropout = 0.5
    params.use_char_cnn = False
    params.randomize = True
    params.use_sent_split = True

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_4',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)


if start_from < 6:
    print("Experiment 5")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.use_char_cnn = False
    params.randomize = True
    params.use_sent_split = True
    params.use_word_self_attention = False

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_5',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)



if start_from < 7:
    print("Experiment 6")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.3
    params.use_char_cnn = False
    params.randomize = True
    params.use_sent_split = True

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_6',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)


if start_from < 8:
    print("Experiment 7")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.1
    params.use_char_cnn = False
    params.randomize = True
    params.use_sent_split = True

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_7',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)


if start_from < 9:
    print("Experiment 8")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.use_char_cnn = False
    params.randomize = True
    params.use_sent_split = True
    params.use_char_attention = True

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_8',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)

# CuDNN error prone; requires debugging
if start_from < 10:
    print("Experiment 9")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.use_char_cnn = False
    params.randomize = True
    params.use_sent_split = False
    params.use_word_self_attention = False

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_9',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)


# # ----------------------------------------------------------------------- Ma and Hovy --------------------------------------------------------------------------
# Experiment 11
if start_from < 11:
    print("Experiment 10")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.use_char_cnn = True
    params.randomize = True
    params.use_sent_split = True
    params.ner_lr = 0.001

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_10',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder, 'output.txt')

        ner_main(paths, params)


if start_from < 12:
    print("Experiment 11")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.use_char_cnn = True
    params.randomize = True
    params.use_sent_split = False

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_11',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)


if start_from < 13:
    print("Experiment 12")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.use_char_cnn = True
    params.randomize = True
    params.use_sent_split = True

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_12',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)


if start_from < 14:
    print("Experiment 13")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 150
    params.ner_dropout = 0.5
    params.use_char_cnn = True
    params.randomize = True
    params.use_sent_split = True

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_13',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)

if start_from < 15:
    print("Experiment 14")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.use_char_cnn = True
    params.randomize = True
    params.use_sent_split = True
    params.use_word_self_attention = False

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_14',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)


if start_from < 16:
    print("Experiment 15")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.3
    params.use_char_cnn = True
    params.randomize = True
    params.use_sent_split = True

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_15',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)


if start_from < 17:
    print("Experiment 16")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.1
    params.use_char_cnn = True
    params.randomize = True
    params.use_sent_split = True

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_16',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)


if start_from < 18:
    print("Experiment 17")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.use_char_cnn = True
    params.randomize = True
    params.use_sent_split = True
    params.use_char_attention = True

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_17',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)

if start_from < 19:
    print("Experiment 18: Word2vec with word embedding size 200, char_rnn_units 30, word_rnn_units 150, dropout=0.5, char_LSTM, CRF, sent_split, regex tokennizer, No self attention")
    params = NER_Params()
    params.word_emb_size = 200
    params.char_rnn_units = 30
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.use_char_cnn = True
    params.randomize = True
    params.use_sent_split = False
    params.use_word_self_attention = False

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_18',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder,'output.txt')

        ner_main(paths, params)

# -----------------------------------------------      ELMO     --------------------------------------------------------

if start_from < 20:
    print("Experiment 19")
    params = NER_Params()
    params.word_emb_size = 1024
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.randomize = True
    params.use_sent_split = True
    params.ner_lr = 0.001
    params.use_elmo = True

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_19',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder, 'output.txt')

        ner_main(paths, params)

if start_from < 21:
    print("Experiment 20 ")
    params = NER_Params()
    params.word_emb_size = 1024
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.randomize = True
    params.use_sent_split = True
    params.ner_lr = 0.001
    params.use_elmo = True
    params.use_word_self_attention = False

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_20',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder, 'output.txt')

        ner_main(paths, params)

# -------------------------------------------------------------ELMO + variational Dropout ----------------------------------------------------------
if start_from < 22:
    print("Experiment 21: Variational dropout")
    params = NER_Params()
    params.word_emb_size = 1024
    params.word_rnn_units = 100
    params.ner_dropout = 0.5
    params.randomize = True
    params.use_sent_split = True
    params.ner_lr = 0.001
    params.use_elmo = True
    params.batch_first = False
    params.use_word_self_attention = False

    params.num_epochs = 100

    for i in range(5):
        paths.reset()
        paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_21',str(i))
        if not os.path.exists(paths.experiment_folder):
            os.makedirs(paths.experiment_folder)
        
        params.output = os.path.join(paths.experiment_folder, 'output.txt')

        ner_vd(paths, params)


# if start_from < 23:
#     print("Experiment 22: Variational dropout")
#     params = NER_Params()
#     params.word_emb_size = 1024
#     params.word_rnn_units = 100
#     params.ner_dropout = 0.5
#     params.randomize = True
#     params.use_sent_split = True
#     params.ner_lr = 0.001
#     params.use_elmo = True
#     params.batch_first = False
#     params.use_word_self_attention = True

#     params.num_epochs = 100

#     for i in range(5):
#         paths.reset()
#         paths.experiment_folder = os.path.join(paths.experiment_folder, 'EX_22',str(i))
#         if not os.path.exists(paths.experiment_folder):
#             os.makedirs(paths.experiment_folder)
        
#         params.output = os.path.join(paths.experiment_folder, 'output.txt')

#         ner_vd(paths, params)