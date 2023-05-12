from tqdm import trange, tqdm
from torch import nn
import time
import random
import torch
from utils.prepare_tensors import tensors_from_pair
from seq2seq.encoder import EncoderRNN
from seq2seq.decoder import DecoderRNN, AttnDecoderRNN

class Transliterator():
    def __init__(self, input_lang, output_lang, pairs, hidden_size, lr, optimizer, criterion, device, teacher_forcing_ratio, num_layers, max_length, use_attention: bool = False):
        self.device = device
        self.encoder = EncoderRNN(input_lang.n_chars, hidden_size, optimizer, self.device, lr, num_layers)
        self.use_attention = use_attention
        if self.use_attention:
            self.decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars, device, optimizer, lr, num_layers)
        else:
            self.decoder = DecoderRNN(hidden_size, output_lang.n_chars, optimizer, self.device, lr, num_layers)

        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_length = max_length
        if criterion == 'nlll':
            self.criterion = nn.NLLLoss()
        self.pairs = pairs
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def train(self, input_tensor, target_tensor):
        encoder_hidden = self.encoder.initHidden()

        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[0]], dtype=torch.long, device=self.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                if self.use_attention:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                    
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                if self.use_attention:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == 1:
                    break
            loss.backward()

        self.encoder.optimizer.step()
        self.decoder.optimizer.step()

        return loss.item() / target_length


    def fit(self, n_iters, compute_loss_every=100):
        start = time.time()
        print_loss_total = 0  # Reset every print_every

        training_pairs = [tensors_from_pair(self.input_lang, self.output_lang, random.choice(self.pairs))
                        for i in range(n_iters)]

        t = trange(1, n_iters + 1, desc="Avg Loss", leave=False)
        loss_total = 0
        for iter in t:
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train(input_tensor, target_tensor)
            loss_total += loss
            print_loss_total += loss

            if iter % compute_loss_every == 0:
                print_loss_avg = print_loss_total / compute_loss_every
                t.set_description(f"Avg loss = {print_loss_avg:.5f}")
                print_loss_total = 0
        end = time.time()
        print(f"Training Time: {(end - start):.3f} seconds")
        print(f"Final average loss = {print_loss_avg:.4f}")
    
    def evaluate(self, pairs):
        with torch.no_grad():
            eval_pairs = [tensors_from_pair(self.input_lang, self.output_lang, pair)
                        for pair in pairs]
            tot_correct = 0
            for i, pair in tqdm(enumerate(eval_pairs)):
                input_tensor = pair[0]
                input_length = input_tensor.size()[0]

                encoder_hidden = self.encoder.initHidden()
                encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

                for ei in range(input_length):
                    encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                    encoder_outputs[ei] += encoder_output[0, 0]

                decoder_input = torch.tensor([[0]], dtype=torch.long, device=self.device)  # SOS

                decoder_hidden = encoder_hidden

                decoded_chars = []
                decoder_attentions = torch.zeros(self.max_length, self.max_length)

                pred_words = []

                for di in range(self.max_length):
                    if self.use_attention:
                        decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                        decoder_attentions[di] = decoder_attention.data
                    else:
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        
                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == 1:
                        decoded_chars.append('<')
                        break
                    else:
                        decoded_chars.append(self.output_lang.index_2_char[topi.item()])

                    decoder_input = topi.squeeze().detach()

                pred_word = "".join([c for c in decoded_chars])
                if(pred_word[-1] == '<'):
                    pred_word = pred_word[:-1]
                pred_words.append(pred_word)
                tot_correct += pairs[i][1] == pred_word
                # print(pred_word, pairs[i][1])
            val_acc = tot_correct/len(eval_pairs) * 100
            print("Word level accuracy", val_acc)
            return pred_word

