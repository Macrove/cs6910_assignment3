from torch import nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, input_embedding_size, hidden_size, device, num_layers, dropout, cell_type):
        super(EncoderRNN, self).__init__()

        self.device = device
        self.embedding = nn.Embedding(input_size, input_embedding_size, device=self.device)
        self.cell_type = cell_type
        if self.cell_type == "gru":
            self.rnn = nn.GRU(input_embedding_size, hidden_size, num_layers, dropout = dropout, device=self.device)
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(input_embedding_size, hidden_size, num_layers, dropout=dropout, device = self.device)
        else: #rnn
            self.rnn = nn.RNN(input_embedding_size, hidden_size, num_layers, dropout = dropout, device=self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):

        # input_tensor shape - seq_len, batch_size
        embedded = self.dropout(self.embedding(input_tensor))
        #embedded_shape - seq_len, batch_size, embed_size

        if self.cell_type == "gru" or "rnn":
            output, hidden = self.rnn(embedded)
            return hidden
        
        elif self.cell_type == "lstm":
            output, (hidden, cell) = self.rnn(embedded)
            return hidden, cell