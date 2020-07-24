import torch
import torch.nn as nn

class SummeryCNN(nn.Module):

    def __init__(self, vocab_size ,embedding, embedding_size, output_size, padidx = 0.0):
        super(SummeryCNN, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padidx, _weight=embedding)
        self.embeddings.weight.requires_grad = False
        self.cnn = CNN(embedding_size, output_size)

    def forward(self, x):
        x = self.embeddings(x.long())
        return self.cnn(x.transpose(-1,-2))


class CNN(nn.Module):

    def __init__(self, num_features, out_feature):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, out_feature, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )

    def forward(self, x):
        x = self.conv1(x.float())
        x = self.conv2(x)
        x = self.conv3(x)

        x, _ = torch.max(x,-1)
        return x

class SummeryRNN(nn.Module):

    def __init__(self, vocab_size, embedding, embedding_size, output_size, padidx = 0.0):
        super(SummeryRNN, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padidx, _weight=embedding)
        self.embeddings.weight.requires_grad = False
        self.num_layers = 1
        self.hidden_size = int(output_size/2)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                        batch_first=True,  bidirectional=True)

    def forward(self, x):
        x = self.embeddings(x.long())

        x,_ = self.lstm(x)
        return x[:,-1,:]