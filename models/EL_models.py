import torch
import torch.nn as nn
from models.gcn import GCN
# from models.lstm import LSTM #Variational dropout LSTM

class EntityModel(nn.Module):

    def __init__(self, word_vocab_size, char_vocab_size, embeddings, word_embedding_size=200, char_embedding_size=30, word_rnn_units=100, 
                    char_rnn_units=30, use_char_emb=True, use_char_cnn=True, use_word_self_attention=False, dropout=0.5, device='cpu', use_gru=True):
        super(EntityModel, self).__init__()
        self.word_emb_size = word_embedding_size
        self.char_emb_size = char_embedding_size
        self.word_lstm_unit = word_rnn_units
        self.char_lstm_unit = char_rnn_units
        self.batch_first = True
        self.use_char_emb = use_char_emb
        self.use_char_cnn= use_char_cnn
        self.dropout = dropout
        self.device = device
        self.use_GRU = use_gru
        self.use_word_self_attention = use_word_self_attention
        
        self.word_embedding = nn.Embedding(word_vocab_size, self.word_emb_size, padding_idx=0)
        self.word_embedding.from_pretrained(embeddings)
        self.word_embedding.weight.requires_grad = False
        self.char_embedding = nn.Embedding(char_vocab_size, self.char_emb_size, padding_idx=0)
        self.num_layers = 2

        if self.use_GRU:
            self.char_rnn= nn.GRU(input_size = self.char_emb_size, hidden_size=self.char_lstm_unit, 
                                    batch_first=self.batch_first, dropout=self.dropout, bidirectional= True)

            if self.use_char_emb:
                self.word_rnn = nn.GRU(input_size=self.word_emb_size+2*self.char_lstm_unit, hidden_size=self.word_lstm_unit, num_layers=self.num_layers,
                                    batch_first=self.batch_first,  bidirectional=True)
            else:
                self.word_rnn = nn.GRU(input_size=self.word_emb_size, hidden_size=self.word_lstm_unit, num_layers=self.num_layers,
                                        batch_first=self.batch_first, dropout=self.dropout, bidirectional=True)
        else:
            self.char_rnn= nn.LSTM(input_size = self.char_emb_size, hidden_size=self.char_lstm_unit, 
                                        batch_first=self.batch_first, dropout=self.dropout, bidirectional= True)

            if self.use_char_emb:
                self.word_rnn = nn.LSTM(input_size=self.word_emb_size+2*self.char_lstm_unit, hidden_size=self.word_lstm_unit, num_layers=self.num_layers,
                                    batch_first=self.batch_first,  bidirectional=True)
            else:
                self.word_rnn = nn.LSTM(input_size=self.word_emb_size, hidden_size=self.word_lstm_unit, num_layers=self.num_layers,
                                        batch_first=self.batch_first, dropout=self.dropout, bidirectional=True)
        
        self.char_cov = CharCNN(self.char_emb_size, 2*self.char_emb_size)

        self.dropout_layer1 = nn.Dropout(p=self.dropout)
        self.dropout_layer2 = nn.Dropout(p=self.dropout)

        self.fc = nn.Linear(2*self.word_lstm_unit, self.word_emb_size)
        self.softmax = nn.Softmax(dim=-1)


    def _init_word_hidden(self, batch_size):
        # default bidirectional
        direction_dim = 2
        if self.use_GRU:
            word_hidden = torch.rand([direction_dim*self.num_layers, batch_size, self.word_lstm_unit]).to(self.device)
        else:
            word_hidden = torch.rand([direction_dim*self.num_layers, batch_size, self.word_lstm_unit]).to(self.device),\
                            torch.rand([direction_dim*self.num_layers, batch_size, self.word_lstm_unit]).to(self.device)
        return word_hidden

    
    def _init_char_hidden(self, batch_size):
        # default bidirectional
        direction_dim = 2
        if self.use_GRU:
            char_hidden = torch.rand([direction_dim, batch_size, self.char_lstm_unit]).to(self.device)
        else:
            char_hidden = torch.rand([direction_dim, batch_size, self.char_lstm_unit]).to(self.device),\
                            torch.rand([direction_dim, batch_size, self.char_lstm_unit]).to(self.device)
        
        return char_hidden

    def forward(self, X):
        # Process word
        word_emb = self.word_embedding(X[0].long())

        if self.use_char_emb:
            self.batch_size, w_seq_length, c_seq_length = X[1].shape
            char_emb = self.char_embedding(X[1])
            w_emb = []
            if self.use_char_cnn:
                for i in range(w_seq_length):
                    if c_seq_length != 1:
                        c_emb = torch.squeeze(char_emb[:,i,:,:],dim=1)
                    else:
                        c_emb = char_emb[:,i,:,:]
                    c_emb = self.char_cov(c_emb.transpose(-1,-2))
                    w_emb.append(c_emb)
            else:
                for i in range(w_seq_length):
                    self.char_hidden = self._init_char_hidden(self.batch_size)
                    if c_seq_length != 1:
                        c_emb = torch.squeeze(char_emb[:,i,:,:],dim=1)
                    else:
                        c_emb = char_emb[:,i,:,:]
                    c_emb, _ = self.char_rnn(c_emb, self.char_hidden)
                    c_emb = c_emb[:,-1,:]
                    w_emb.append(c_emb)
            
            char_emb = torch.stack(w_emb, dim=1)
            word_emb = torch.cat((word_emb, char_emb),-1)

        else:
            self.batch_size, w_seq_length = X[0].shape

        self.word_hidden = self._init_word_hidden(self.batch_size)
        z = self.dropout_layer1(word_emb)
        z, _ = self.word_rnn(word_emb, self.word_hidden)
        

        # self attention
        att_weight = torch.matmul(z, z.transpose(-1,-2))
        softmax_weight = self.softmax(att_weight)
        z_att = torch.matmul(softmax_weight, z)
        if self.use_word_self_attention:
            z = z_att
        # z = self.dropout_layer2(z)
        # z = self.fc(z)

        return z


class CharCNN(nn.Module):

    def __init__(self, num_features, out_feature, hidden_filters = 64):
        super(CharCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(num_features, hidden_filters, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_filters, out_feature, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x, _ = torch.max(x,-1)

        return x

class EL_model(nn.Module):
    def __init__(self, in_features, out_features, scope_embedding):
        super(EL_model, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

        self.scope_embedding = scope_embedding
        self.len_scope = scope_embedding.shape[0]

    def forward(self, X):
        batch_size, x_embed_size = X.shape
        X = self.fc(X)
        X = X.unsqueeze(1)
        y = self.scope_embedding.unsqueeze(0).repeat(batch_size,1,1)

        z = torch.bmm(X,y.transpose(1,2))
        return z.squeeze(1)

class EL_GCN(nn.Module):
    def __init__(self, in_features_dim, out_features_dim,  scope_embedding, hidden_dim=0, dropout=0.0):
        super(EL_GCN, self).__init__()
        self.gcn = GCN(in_features_dim, hidden_dim, dropout=dropout)
        self.linear = nn.Linear(in_features_dim, out_features_dim)
        self.scope_embedding = scope_embedding
        self.len_scope = scope_embedding.shape[0]

    def forward(self, x, a_hat):
        batch_size = x.shape[0]

        # x = self.linear(x)
        x = x.unsqueeze(1)

        y, y_ = self.gcn(self.scope_embedding, a_hat)
        y = y.unsqueeze(0).repeat(batch_size,1,1)

        z = torch.bmm(x, y.transpose(1,2))
        return z.squeeze(1)

class BiLSTM(nn.Module):
    def __init__(self, word_embedding_size=1024, word_rnn_units=100, 
                    use_word_self_attention=False, dropout=0.5, device='cpu', use_gru=False, num_layers=2, batch_first=True):
        super(BiLSTM, self).__init__()
        self.use_GRU = use_gru
        self.word_emb_size = word_embedding_size
        self.word_lstm_unit = word_rnn_units
        self.use_word_self_attention = use_word_self_attention
        self.batch_first = batch_first
        self.dropout = dropout
        self.num_layers = num_layers

        self.fc = nn.Linear(2*self.word_lstm_unit, self.word_emb_size)
        self.softmax = nn.Softmax(dim=-1)


        if self.use_GRU:
            self.word_rnn = nn.GRU(input_size=self.word_emb_size, hidden_size=self.word_lstm_unit, num_layers=self.num_layers,
                                        batch_first=self.batch_first, dropout=self.dropout, bidirectional=True)
        else:
            # replaced this with Variational Dropout based LSTM in original training
            self.word_rnn = nn.LSTM(input_size=self.word_emb_size, hidden_size=self.word_lstm_unit, num_layers=self.num_layers,
                                        batch_first=self.batch_first, dropout=self.dropout, bidirectional=True) 

    def forward(self, X):
        z, _ = self.word_rnn(X)
        z, lengths = nn.utils.rnn.pad_packed_sequence(z, batch_first=self.batch_first)
        
        # self attention
        att_weight = torch.matmul(z, z.transpose(-1,-2))
        softmax_weight = self.softmax(att_weight)
        z_att = torch.matmul(softmax_weight, z)
        if self.use_word_self_attention:
            z = z_att

        return z, lengths


class EL_similarity(nn.Module):
    def __init__(self, embedding_size, char_vocab_size):
        super(EL_similarity, self).__init__()
        self.emb = nn.Embedding(char_vocab_size, embedding_size)
        self.cnn = CharCNN(embedding_size, embedding_size, hidden_filters=128)
        self.fc = nn.Linear(embedding_size, embedding_size)

    def forward(self, x, y):
        x_emb = self.emb(x)
        y_emb = self.emb(y)

        x_emb = self.cnn(x_emb.transpose(-1,-2))
        y_emb = self.cnn(y_emb.transpose(-1,-2))

        x_emb = self.fc(x_emb)
        y_emb = self.fc(y_emb)

        return x_emb, y_emb