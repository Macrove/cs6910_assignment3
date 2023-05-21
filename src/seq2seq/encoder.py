from torch import nn
import torch

class EncoderRNN(nn.Module):
    def __init__(self, input_size, input_embedding_size, hidden_size, device, num_layers, dropout, cell_type, use_attention):
        super(EncoderRNN, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, input_embedding_size, device=self.device)
        self.cell_type = cell_type
        if use_attention:
            self.use_attention = True
        else:
            self.use_attention = False
        if self.cell_type == "gru":
            self.rnn = nn.GRU(input_embedding_size, hidden_size, num_layers, dropout = dropout, device=self.device)
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(input_embedding_size, hidden_size, num_layers, dropout=dropout, device = self.device, bidirectional = self.use_attention)
        else: #rnn
            self.rnn = nn.RNN(input_embedding_size, hidden_size, num_layers, dropout = dropout, device=self.device)
        self.dropout = nn.Dropout(dropout)
        if self.use_attention:
            self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size).to(self.device)
            self.fc_cell = nn.Linear(hidden_size * 2, hidden_size).to(self.device)

    def forward(self, input_tensor):

        # input_tensor shape - seq_len, batch_size
        embedded = self.dropout(self.embedding(input_tensor))
        #embedded_shape - seq_len, batch_size, embed_size

        if self.cell_type == "gru" or self.cell_type == "rnn":
            output, hidden = self.rnn(embedded)
            return hidden
        
        elif self.cell_type == "lstm":
            if self.use_attention:
                encoder_states, (hidden, cell) = self.rnn(embedded)
                hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
                cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
                return encoder_states, hidden, cell
            else:
                output, (hidden, cell) = self.rnn(embedded)
                return hidden, cell

