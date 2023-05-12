import torch
from torch import nn, optim

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, optimizer, device, lr, num_layers):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.lr = lr

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers = num_layers)
        self.device = device
        self.num_layers = num_layers
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr, (0.99, 0.9))

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)