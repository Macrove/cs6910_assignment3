import torch
from utils.env import DEVICE

# def indexes_from_word(lang, word):
#     indexes = []
#     for char in list(word):
#         if char not in lang.char_2_index:
#             char = '~'
#         indexes.append(lang.char_2_index[char])
#     return indexes

# def tensor_from_word(lang, word):
#     indexes = indexes_from_word(lang, word)
#     indexes.insert(0, 0)
#     indexes.append(1)
#     return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)

# def tensors_from_pair(input_lang, output_lang, pair):
#     input_tensor = tensor_from_word(input_lang, pair[0])
#     target_tensor = tensor_from_word(output_lang, pair[1])
#     return (input_tensor, target_tensor)