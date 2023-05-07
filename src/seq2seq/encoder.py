import torch
from torch import nn, optim

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, optimizer, device, lr):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.lr = lr

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = device
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)