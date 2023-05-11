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
hidden_size = 256
lr = 0.01
eng2hin = Transliterator(input_lang, output_lang, train_pairs, hidden_size, lr, 'sgd', 'nlll', DEVICE, 0.5)
eng2hin.fit(10000, 100)
eng2hin.evaluate(valid_pairs)