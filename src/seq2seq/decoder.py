import torch
from torch import nn, optim
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, output_size, output_embedding_size, hidden_size, device, num_layers, dropout, cell_type, use_attention):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.use_attention = use_attention


        self.embedding = nn.Embedding(output_size, output_embedding_size)
        if self.cell_type == "gru":
            self.rnn = nn.GRU(output_embedding_size, hidden_size, num_layers = num_layers, dropout = dropout)
        elif self.cell_type == "lstm":
            if not self.use_attention:
                self.rnn = nn.LSTM(output_embedding_size, hidden_size, num_layers = num_layers, dropout = dropout)
            else:
                self.rnn = nn.LSTM(hidden_size * 2 + output_embedding_size, hidden_size, num_layers, dropout=dropout)

        else: #rnn
            self.rnn = nn.RNN(output_embedding_size, hidden_size, num_layers = num_layers, dropout = dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        if self.use_attention:
            self.energy = nn.Linear(hidden_size * 3, 1)
            self.softmax = nn.Softmax(dim=0)
            self.relu = nn.ReLU()

    def forward(self, input, hidden, cell = 0, encoder_states = 0):
        if self.cell_type == "lstm":
            input = input.unsqueeze(0)
            #input shape now - (1, N)

            embedded = self.dropout(self.embedding(input))
            # embedding shape: (1, N, embedding_size)

            if self.use_attention:
                sequence_length = encoder_states.shape[0]
                h_reshaped = hidden.repeat(sequence_length, 1, 1)

                energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))

                attention = self.softmax(energy)
                context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

                rnn_input = torch.cat((context_vector, embedded), dim=2)

                outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

                predictions = self.fc(outputs).squeeze(0)

                return predictions, hidden, cell

            else:
                # print(embedded.shape, hidden.shape, cell.shape)
                outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
                # outputs shape: (1, N, hidden_size)

                predictions = self.fc(outputs)
                predictions = predictions.squeeze(0)

                return predictions, hidden, cell
        elif self.cell_type == "gru" or self.cell_type == "rnn":
            input = input.unsqueeze(0)
            #input shape now - (1, N)
            embedded = self.dropout(self.embedding(input))
            # embedding shape: (1, N, embedding_size)
            outputs, hidden = self.rnn(embedded, hidden)
            # outputs shape: (1, N, hidden_size)

            predictions = self.fc(outputs)
            predictions = predictions.squeeze(0)

            return predictions, hidden
