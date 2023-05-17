import torch
from torch import nn, optim

class EncoderRNN(nn.Module):
    def __init__(self, input_size, input_embedding_size, hidden_size, optimizer, device, num_layers, dropout):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, input_embedding_size)
        self.gru = nn.GRU(input_embedding_size, hidden_size, num_layers, dropout = dropout)
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        if optimizer["name"] == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), **optimizer["default_params"])
        elif optimizer["name"] == 'adam':
            self.optimizer = optim.Adam(self.parameters(), **optimizer["default_params"])

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = self.dropout(embedded)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)