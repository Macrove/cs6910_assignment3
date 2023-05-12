import os
from utils.env import TRAIN_FILENAME, TEST_FILENAME, VALID_FILENAME, DIR_PATH, LANG1NAME, LANG2NAME
from prepare_langs import prepare_langs, get_pairs
from seq2seq.nn import Transliterator
from utils.env import DEVICE

train_file = os.path.join(DIR_PATH, TRAIN_FILENAME)
valid_file = os.path.join(DIR_PATH, VALID_FILENAME)
test_file = os.path.join(DIR_PATH, TEST_FILENAME)

input_lang, output_lang = prepare_langs(train_file, LANG1NAME, LANG2NAME)

train_pairs = get_pairs(train_file)
valid_pairs = get_pairs(valid_file)
test_pairs = get_pairs(test_file)
teacher_forcing_ratio = 0.5
num_layers = 2
hidden_size = 256
lr = 0.01
use_attention = True
max_length = 40
optimizer = 'sgd'
loss = 'nlll'
eng2hin = Transliterator(input_lang, output_lang, train_pairs, hidden_size, lr, optimizer, loss, DEVICE, teacher_forcing_ratio, num_layers, max_length, use_attention)
eng2hin.fit(50000)
eng2hin.evaluate(valid_pairs)