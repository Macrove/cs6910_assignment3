import os
from utils.env import TRAIN_FILENAME, TEST_FILENAME, VALID_FILENAME, DIR_PATH, LANG1NAME, LANG2NAME
from prepare_langs import prepare_langs

train_file = os.path.join(DIR_PATH, TRAIN_FILENAME)

prepare_langs(train_file, LANG1NAME, LANG2NAME)