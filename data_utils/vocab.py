import pandas as pd
import numpy as np


class Vocab:
    def __init__(self, data):
        self.words = list(set(data))
        self.word_to_idx = {w: i + 1 for i, w in enumerate(self.words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        
    def convert_ids_to_tokens(self, ids):
        return [self.idx_to_word.get(idx) for idx in ids]
    
    def convert_tokens_to_ids(self, tokens):
        return [self.word_to_idx.get(token) for token in tokens]
    
    def pad_token_id(self):
        return 0 # ID for the padding token