import torch.cuda as cuda

DIR_PATH = "./src/dataset/aksharantar_sampled/hin"
TEST_FILENAME = "hin_test.csv"
TRAIN_FILENAME = "hin_train.csv"
VALID_FILENAME = "hin_valid.csv"

LANG1NAME = "English"
LANG2NAME = "Hindi"

DEVICE = 'cuda' if cuda.is_available() else 'cpu'