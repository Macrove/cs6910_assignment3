from tqdm import tqdm
from torch import nn, optim
import wandb
from torch import nn
import random
import torch
from seq2seq.encoder import EncoderRNN
from seq2seq.decoder import DecoderRNN, AttnDecoderRNN

class Transliterator(nn.Module):
    def __init__(self, loss, optimizer, use_wandb,
                 input_embedding_size, num_layer,
                 hidden_size, cell_type, bidirectional, dropout, teacher_forcing_ratio,
                 use_attention, device, batch_size, train_iterator, source_field,
                 target_field, valid_iterator, valid_dataset, epochs, max_length = 40):
        super(Transliterator, self).__init__()
        self.device = device
        self.source_field = source_field
        self.epochs = epochs
        self.target_field = target_field
        self.valid_dataset = valid_dataset
        self.valid_iterator = valid_iterator
        self.train_iterator = train_iterator
        self.encoder = EncoderRNN(len(source_field.vocab), input_embedding_size, hidden_size, device, num_layer, dropout, cell_type).to(device)
        self.use_attention = use_attention
        if self.use_attention:
            self.decoder = AttnDecoderRNN(hidden_size, device, optimizer, num_layer, dropout, cell_type, max_length)
        else:
            self.decoder = DecoderRNN(len(target_field.vocab), input_embedding_size, hidden_size, device, num_layer, dropout, cell_type).to(device)


        self.max_length = max_length
        if loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(ignore_index=target_field.vocab.stoi["<pad>"]) #ignore pad index
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.dropout = dropout
        self.cell_type = cell_type
        self.use_wandb = use_wandb
        self.input_embedding_size = input_embedding_size
        self.bidirectional = bidirectional
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.batch_size = batch_size
        if optimizer["name"] == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), **optimizer["default_params"])
        elif optimizer["name"] == 'adam':
            self.optimizer = optim.Adam(self.parameters(), **optimizer["default_params"])


    def forward(self, input_tensor, target_tensor):
        if self.cell_type == 'lstm':
            target_length = target_tensor.shape[0]
            target_n_chars = len(self.target_field.vocab)

            encoder_hidden, encoder_cell = self.encoder(input_tensor)

            decoder_input = target_tensor[0] # sos token
            decoder_outputs = torch.zeros(target_length, self.batch_size, target_n_chars, device=self.device)

            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell

            for t in range(1, target_length):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
                decoder_outputs[t] = decoder_output

                best_guess = decoder_output.argmax(1)

                decoder_input = target_tensor[t] if random.random() < self.teacher_forcing_ratio else best_guess

            return decoder_outputs

    def fit(self):
        num_epochs = self.epochs
        for epoch in range(num_epochs):
            print(f"[Epoch {epoch} / {num_epochs}]")
            self.train()
            for batch_idx, batch in tqdm(enumerate(self.train_iterator), total = len(self.train_iterator)):
                inp_data, inp_len = batch.src
                target, trg_len = batch.trg
                # Forward prop
                output = self(inp_data, target)

                output = output[1:].reshape(-1, output.shape[2])
                target = target[1:].reshape(-1)

                self.optimizer.zero_grad()
                loss = self.criterion(output, target)

                # Back prop
                loss.backward()

                # Clip to avoid exploding gradient issues, makes sure grads are
                # within a healthy range
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)

                # Gradient descent step
                self.optimizer.step()
            self.evaluate_loss(self.valid_iterator)
        

    def evaluate_loss(self, iterator):
        self.eval()
        with torch.no_grad():
            loss = 0
            for batch_idx, batch in enumerate(iterator):
                inp_data, inp_len = batch.src
                target, trg_len = batch.trg
                # Forward prop
                output = self(inp_data, target)

                output = output[1:].reshape(-1, output.shape[2])
                target = target[1:].reshape(-1)

                loss += self.criterion(output, target).item()
            print(f"val_loss - {loss:.4}")

    def validate(self):
        correct = 0
        for pair in self.valid_dataset:
            pred_word = self.predict(pair.src)
            target_word = "".join(pair.trg)
            if pred_word == target_word:
                correct+=1

        accuracy = correct/len(self.valid_dataset)
        return accuracy
        

    def predict(self, tokens):
        self.eval()
        tokens.insert(0, self.source_field.init_token)
        tokens.append(self.source_field.eos_token)


        text_to_indices = [self.source_field.vocab.stoi[token] for token in tokens]

        input_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(self.device)
        with torch.no_grad():
            hidden, cell = self.encoder(input_tensor)

        outputs = [self.target_field.vocab.stoi["<sos>"]]

        for _ in range(self.max_length):
            previous_char = torch.LongTensor([outputs[-1]]).to(self.device)
            with torch.no_grad():
                output, hidden, cell = self.decoder(previous_char, hidden, cell)
                best_guess = output.argmax(1).item()

            outputs.append(best_guess)
            if output.argmax(1).item() == self.target_field.vocab.stoi["<eos>"]:
                break

        pred_chars = [self.target_field.vocab.itos[idx] for idx in outputs]
        pred_chars = pred_chars[1:-1]
        pred_word = "".join(pred_chars)
        return pred_word

