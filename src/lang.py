class Lang:
    def __init__(self, name):
        self.name = name
        self.char_2_index = {}
        # self.word2count = {}
        self.index_2_char = {}
        self.n_chars = 0

    def add_word(self, word):
        chars = list(word)
        for char in chars:
            self.add_char(char)

    def add_char(self, char):
        if char not in self.char_2_index:
            self.char_2_index[char] = self.n_chars
            # self.char2count[char] = 1
            self.index_2_char[self.n_chars] = char
            self.n_chars += 1
        # else:
        #     self.word2count[word] += 1