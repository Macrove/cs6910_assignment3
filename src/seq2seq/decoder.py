import torch
from torch import nn, optim
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, optimizer, device, num_layers, dropout = 0.1, max_length = 40):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers = 2, dropout = dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        if optimizer["name"] == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), **optimizer["default_params"])
        elif optimizer["name"] == 'adam':
            self.optimizer = optim.Adam(self.parameters(), **optimizer["default_params"])

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout(output)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, optimizer, num_layers, dropout=0.1, max_length=40):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.max_length = max_length
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers = num_layers, dropout=dropout)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.num_layers = num_layers
        if optimizer["name"] == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), **optimizer["default_params"])
        elif optimizer["name"] == 'adam':
            self.optimizer = optim.Adam(self.parameters(), **optimizer["default_params"])

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)