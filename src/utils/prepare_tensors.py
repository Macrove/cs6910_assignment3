import torch
from utils.env import DEVICE

def indexes_from_word(lang, word):
    return [lang.char_2_index[char] for char in list(word)]


def tensor_from_word(lang, word):
    indexes = indexes_from_word(lang, word)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)

def tensors_from_pair(input_lang, output_lang, pair):
    input_tensor = tensor_from_word(input_lang, pair[0])
    target_tensor = tensor_from_word(output_lang, pair[1])
    return (input_tensor, target_tensor)