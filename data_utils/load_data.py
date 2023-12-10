import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from data_utils.vocab import Vocab

class NERDataset(Dataset):
    def __init__(self, df, vocab, max_len, num_tags):
        super().__init__()
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        self.sentences = self.get_sentences(self.df)


    def get_sentences(self, data):
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        grouped = data.groupby("Sentence #").apply(agg_func)
        return [s for s in grouped]
    
    def build_data(self):
        X = []
        for s in self.sentences:
            