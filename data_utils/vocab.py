import pandas as pd
import numpy as np
import os


class Vocab:
    def __init__(self, config):
        self.word_to_idx = {}
        self.idx_to_word = {}

        self.dataset_folder = config["dataset"]["dataset_folder"]
        self.train_path = config["dataset"]["train_path"]
        self.dev_path = config["dataset"]["dev_path"]
        self.test_path = config["dataset"]["test_path"]

        self.build_vocab()

    def all_word(self):
        train = pd.read_csv(os.path.join(self.dataset_folder, self.train_path), encoding= 'utf-8')
        dev = pd.read_csv(os.path.join(self.dataset_folder, self.dev_path), encoding= 'utf-8')
        test = pd.read_csv(os.path.join(self.dataset_folder, self.test_path), encoding= 'utf-8')

        word_count = {}

        for data_file in [train, dev, test]:
            for item in data_file["sentence"]:
                for word in item.split():
                    if word not in word_count:
                        word_count[word] = 1
                    else:
                        word_count[word] += 1

        special_tokens = ['<unk>', '<cls>', '<sep>']
        for w in special_tokens:
            if w not in word_count:
                word_count[w] = 1
            else:
                word_count[w] += 1
        
        sorted_word_count = dict(sorted(word_count.items(), key= lambda x: x[1], reverse= True))
        all_word = list(sorted_word_count.keys())

        return all_word, sorted_word_count

    def build_vocab(self):
        all_word, _ = self.all_word()
        self.word_to_idx = {word: idx + 1 for idx, word in enumerate(all_word)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def convert_tokens_to_ids(self, tokens):
        return [self.word_to_idx.get(token, self.word_to_idx['<unk>']) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.idx_to_word[idx] for idx in ids]
    
    def vocab_size(self):
        return len(self.word_to_idx) + 1
    
    def pad_token_idx(self):
        return 0