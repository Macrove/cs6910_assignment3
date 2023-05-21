from tqdm import tqdm
from torch import nn, optim
import wandb
from torch import nn
import random
import torch
from seq2seq.encoder import EncoderRNN
from seq2seq.decoder import DecoderRNN

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
        self.use_attention = use_attention
        self.train_iterator = train_iterator
        self.encoder = EncoderRNN(len(source_field.vocab), input_embedding_size, hidden_size, device, num_layer, dropout, cell_type, self.use_attention).to(device)
        self.decoder = DecoderRNN(len(target_field.vocab), input_embedding_size, hidden_size, device, num_layer, dropout, cell_type, use_attention).to(device)


        self.max_length = max_length
        if loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(ignore_index=target_field.vocab.stoi["<pad>"]).to(device) #ignore pad index
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
        target_length = target_tensor.shape[0]
        target_n_chars = len(self.target_field.vocab)
        if self.use_attention == 0:
            if self.cell_type == 'lstm':

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

            elif self.cell_type == 'gru' or self.cell_type == "rnn":
                encoder_hidden = self.encoder(input_tensor)

                decoder_input = target_tensor[0] # sos token
                decoder_outputs = torch.zeros(target_length, self.batch_size, target_n_chars, device=self.device)

                decoder_hidden = encoder_hidden

                for t in range(1, target_length):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    decoder_outputs[t] = decoder_output

                    best_guess = decoder_output.argmax(1)

                    decoder_input = target_tensor[t] if random.random() < self.teacher_forcing_ratio else best_guess

                return decoder_outputs
        else: #using attention
            outputs = torch.zeros(target_length, self.batch_size, target_n_chars).to(self.device)
            encoder_states, hidden, cell = self.encoder(input_tensor)

            # First input will be <SOS> token
            x = target_tensor[0]

            for t in range(1, target_length):
                # At every time step use encoder_states and update hidden, cell
                output, hidden, cell = self.decoder(x, hidden, cell, encoder_states)

                # Store prediction for current time step
                outputs[t] = output

                # Get the best word the Decoder predicted (index in the vocabulary)
                best_guess = output.argmax(1)

                # With probability of teacher_force_ratio we take the actual next word
                # otherwise we take the word that the Decoder predicted it to be.
                # Teacher Forcing is used so that the model gets used to seeing
                # similar inputs at training and testing time, if teacher forcing is 1
                # then inputs at test time might be completely different than what the
                # network is used to. This was a long comment.
                x = target_tensor[t] if random.random() < self.teacher_forcing_ratio else best_guess

            return outputs


    def fit(self):
        num_epochs = self.epochs
        print("Training")
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
            self.evaluate_loss(self.train_iterator, "train")
            self.evaluate_loss(self.valid_iterator, "val")
        

    def evaluate_loss(self, iterator, dataset_type):
        self.eval()
        with torch.no_grad():
            loss = 0
            if dataset_type == "train":
                print("Evaluating training loss")
            elif dataset_type == "val":
                print("Evaluating validation loss")
            for batch_idx, batch in tqdm(enumerate(iterator), total=len(iterator)):
                inp_data, inp_len = batch.src
                target, trg_len = batch.trg
                # Forward prop
                output = self(inp_data, target)

                output = output[1:].reshape(-1, output.shape[2])
                target = target[1:].reshape(-1)

                loss += self.criterion(output, target).item()
            print(f"{dataset_type}_loss - {loss:.4f}")
            if self.use_wandb:
                wandb.log({f"{dataset_type}_loss": round(loss, 4)})

    def validate(self, dataset, return_preds = False):
        correct = 0
        pred_target_pairs = []
        print("Evaluating word level accuracy")
        for pair in tqdm(dataset, total=len(dataset)):
            pred_word = self.predict(pair.src)
            target_word = "".join(pair.trg)
            if pred_word == target_word:
                correct+=1
            if return_preds:
                if pred_word == target_word:
                    pred_target_pairs.append([pred_word, target_word, 1])
                else:
                    pred_target_pairs.append([pred_word, target_word, 0])
                    

        accuracy = correct/len(self.valid_dataset)
        return accuracy, pred_target_pairs
        

    def predict(self, tokens):
        self.eval()
        tokens.insert(0, self.source_field.init_token)
        tokens.append(self.source_field.eos_token)


        text_to_indices = [self.source_field.vocab.stoi[token] for token in tokens]

        input_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(self.device)
        if self.cell_type == "lstm":
            with torch.no_grad():
                if self.use_attention:
                    outputs_encoder, hidden, cell = self.encoder(input_tensor)
                else:
                    hidden, cell = self.encoder(input_tensor)

            outputs = [self.target_field.vocab.stoi["<sos>"]]

            for _ in range(self.max_length):
                previous_char = torch.LongTensor([outputs[-1]]).to(self.device)
                with torch.no_grad():
                    if self.use_attention:
                        output, hidden, cell = self.decoder(previous_char, hidden, cell, outputs_encoder)
                    else:
                        output, hidden, cell = self.decoder(previous_char, hidden, cell)
                    best_guess = output.argmax(1).item()

                outputs.append(best_guess)
                if output.argmax(1).item() == self.target_field.vocab.stoi["<eos>"]:
                    break
        elif self.cell_type == "gru" or self.cell_type == "rnn":
            with torch.no_grad():
                hidden = self.encoder(input_tensor)

            outputs = [self.target_field.vocab.stoi["<sos>"]]

            for _ in range(self.max_length):
                previous_char = torch.LongTensor([outputs[-1]]).to(self.device)
                with torch.no_grad():
                    output, hidden = self.decoder(previous_char, hidden)
                    best_guess = output.argmax(1).item()

                outputs.append(best_guess)
                if output.argmax(1).item() == self.target_field.vocab.stoi["<eos>"]:
                    break
            
        pred_chars = [self.target_field.vocab.itos[idx] for idx in outputs]
        pred_chars = pred_chars[1:-1]
        pred_word = "".join(pred_chars)
        return pred_word

