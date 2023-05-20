import torch
from torch import nn, optim
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, output_size, output_embedding_size, hidden_size, device, num_layers, dropout, cell_type):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.cell_type = cell_type

        self.embedding = nn.Embedding(output_size, output_embedding_size, device=self.device)
        if self.cell_type == "gru":
            self.rnn = nn.GRU(output_embedding_size, hidden_size, num_layers = num_layers, dropout = dropout, device=self.device)
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(output_embedding_size, hidden_size, num_layers = num_layers, dropout = dropout, device=self.device)
        self.fc = nn.Linear(hidden_size, output_size, device=self.device)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell = 0):
        if self.cell_type == "lstm":
            input = input.unsqueeze(0)
            #input shape now - (1, N)

            embedded = self.dropout(self.embedding(input))
            # embedding shape: (1, N, embedding_size)

            outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            # outputs shape: (1, N, hidden_size)

            predictions = self.fc(outputs)
            predictions = predictions.squeeze(0)

            return predictions, hidden, cell
        elif self.cell_type == "gru":
            input = input.unsqueeze(0)
            #input shape now - (1, N)
            embedded = self.dropout(self.embedding(input))
            # embedding shape: (1, N, embedding_size)
            outputs, hidden = self.rnn(embedded, hidden)
            # outputs shape: (1, N, hidden_size)

            predictions = self.fc(outputs)
            predictions = predictions.squeeze(0)

            return predictions, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, optimizer, num_layers, dropout, cell_type, max_length=40):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.max_length = max_length
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size, device=self.device)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length, device=self.device)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size, device=self.device)
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type
        if self.cell_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers = num_layers, dropout = dropout, device=self.device)
        else:
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers = num_layers, dropout = dropout, device=self.device)
        self.out = nn.Linear(self.hidden_size, self.output_size, device=self.device)
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
        if self.cell_type == "gru":
            output, hidden = self.rnn(output, hidden)
        else:
            output, hidden, _ = self.rnn(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)