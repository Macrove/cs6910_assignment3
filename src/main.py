import os
from utils.env import TRAIN_FILENAME, TEST_FILENAME, VALID_FILENAME, DIR_PATH, LANG1NAME, LANG2NAME
from prepare_langs import prepare_langs
from seq2seq.nn import Transliterator
from utils.env import DEVICE

train_file = os.path.join(DIR_PATH, TRAIN_FILENAME)

input_lang, output_lang, pairs = prepare_langs(train_file, LANG1NAME, LANG2NAME)

hidden_size = 10
lr = 0.01
eng2hin = Transliterator(input_lang, output_lang, pairs, hidden_size, lr, 'sgd', 'nlll', DEVICE)
eng2hin.fit(10000, 100)