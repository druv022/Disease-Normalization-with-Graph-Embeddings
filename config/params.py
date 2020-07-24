import time

params_dict = {
    'char_emb_size' : 30,
    'word_emb_size' : 200,
    'char_rnn_units' : 30,
    'word_rnn_units' : 100,
    'ner_dropout' : 0.5,
    'batch_size' : 32,
    'num_epochs' : 100,
    'use_char' : True,
    'use_char_cnn' : True,
    'use_char_attention' : False,
    'ner_use_word_self_attention' : True,
    'use_elmo' : False,
    'elmo_dim' : 1024,
    'entity_names' : ['B_Disease','I_Disease'],
    'use_gru' : False,
    'ner_layers' : 1,
    'randomize' : False,
    'ner_lr' : 0.001,
    'use_sent_split':True,
    'use_similarity' : False,
    'negative_samples' : 5,
    'EL_lr' : 0.001,
    'EL_dropout' : 0.5,
    'test' : False,
    'batch_first' : True,
    'mt_layers' : 1,
    'mt_use_word_self_attention' : False,
    'mt_activation' : 'fusedmax',
    'mt_lr' : 0.001,
    'mt_dropout' : 0.2
}


class Params():

    def __init__(self):
        self.reset()

    def reset(self):
        self.output = 'Default.txt'
        self.word_emb_size = params_dict['word_emb_size']
        self.batch_size = params_dict['batch_size']
        self.num_epochs = params_dict['num_epochs']
        self.model_name = 'Default'
        self.use_elmo = params_dict['use_elmo']
        self.elmo_dim = params_dict['elmo_dim']
        self.time = time.strftime("%Y%m%d-%H%M%S")
        self.randomize = params_dict['randomize']
        self.use_sent_split = params_dict['use_sent_split']
        self.test = params_dict['test']
        self.batch_first = params_dict['batch_first']
        self.use_gru = params_dict['use_gru']

class NER_Params(Params):
    def __init__(self):
        super(NER_Params,self).__init__()
        self.reset()
    
    def reset(self):
        super().reset()
        self.char_emb_size = params_dict['char_emb_size']
        self.char_rnn_units = params_dict['char_rnn_units']
        self.word_rnn_units = params_dict['word_rnn_units']
        self.dropout = params_dict['ner_dropout']
        self.lr = params_dict['ner_lr']
        self.use_char = params_dict['use_char']
        self.use_char_cnn = params_dict['use_char_cnn']
        self.use_char_attention = params_dict['use_char_attention']
        self.use_word_self_attention = params_dict['ner_use_word_self_attention']
        self.entity_names = params_dict['entity_names']
        self.num_layers = params_dict['ner_layers']

class EL_params(Params):
    def __init__(self):
        super(EL_params,self).__init__()
        self.reset()

    def reset(self):
        super().reset()
        self.use_similarity = params_dict['use_similarity']
        self.negative_samples = params_dict['negative_samples']
        self.lr = params_dict['EL_lr']
        self.dropout = params_dict['EL_dropout']



class MultiTask_Params(Params):
    def __init__(self):
        super(MultiTask_Params, self).__init__()

        self.reset()

    def reset(self):
        super().reset()
        self.ner = NER_Params()
        self.ner.reset()

        self.EL = EL_params()
        self.EL.reset()

        self.rnn_units = 512
        self.num_layers = params_dict['mt_layers']
        self.use_word_self_attention = params_dict['mt_use_word_self_attention']
        self.activation = params_dict['mt_activation']
        self.lr = params_dict['mt_lr']
        self.dropout = params_dict['mt_dropout']
        self.use_activation = False
        self.only_NER = False
        self.only_EL = False

        self.NER_type1 = True

        if self.only_NER and self.only_EL:
            print('By default, it runs both NER and EL module') 


class Node2vec_Params():
    def __init__(self, p, q):
        self.p = 1
        self.q = 1
        self.epochs = 20
        self.batch_size=512
        self.window = 2
        self.num_neg_sample = 5
        self.randomize = True
    


if __name__ == "__main__":
    print('Here')