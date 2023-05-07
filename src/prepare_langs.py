import csv
from lang import Lang
from tqdm import tqdm


def prepare_langs(train_file, name1, name2):
    lang1 = Lang(name1)
    lang2 = Lang(name2)

    with open(train_file, "r") as file:
        train_reader = csv.reader(file, delimiter=',')
        count = 0
        pairs = []
        for pair in train_reader:
            lang1.add_word(pair[0])
            lang2.add_word(pair[1])
            count += 1
            pairs.append(pair)
    
        print(f"Added {count} transliteration pairs to train dataset")
        print('\n')

        print(f"Total characters in {lang1.name} = {lang1.n_chars}")
        print(lang1.char_2_index)
        print('\n')
        print(f"Total characters in {lang2.name} = {lang2.n_chars}")
        print(lang2.char_2_index)
        return lang1, lang2, pairs

