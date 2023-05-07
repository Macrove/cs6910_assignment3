from tqdm import trange
from torch import nn
import time
import random
import torch
from utils.prepare_tensors import tensors_from_pair
from seq2seq.encoder import EncoderRNN
from seq2seq.decoder import DecoderRNN

class Transliterator():
    def __init__(self, input_lang, output_lang, pairs, hidden_size, lr, optimizer, criterion, device, max_length = 30):
        self.device = device
        self.encoder = EncoderRNN(input_lang.n_chars, hidden_size, optimizer, self.device, lr)
        self.decoder = DecoderRNN(hidden_size, output_lang.n_chars, optimizer, self.device, lr)
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_length = max_length
        if criterion == 'nlll':
            self.criterion = nn.NLLLoss()
        self.pairs = pairs

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

        # decoder_input = torch.tensor([1], dtype=torch.long, device=self.device)

        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_input = target_tensor[di]  # Teacher forcing
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden)
            loss += self.criterion(decoder_output, target_tensor[di])

        loss.backward()

        self.encoder.optimizer.step()
        self.decoder.optimizer.step()

        return loss.item() / target_length


    def trainIters(self, n_iters, print_every=100):
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

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                t.set_description(f"Avg loss = {print_loss_avg:.5f}")
                print_loss_total = 0
        end = time.time()
        print(f"Training Time: {(end - start)} s")
        print(f"Final average loss = {print_loss_avg}")

