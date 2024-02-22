import torch
from torch import nn
from data_utils.vocab import Vocab

class WordEmbedding(nn.Module):
    def __init__(self, config):
        super(WordEmbedding, self).__init__()

        self.vocab = Vocab(config)
        self.max_length = config["text_embedding"]["max_length"]
        self.embedding_dim = config["text_embedding"]["embedding_dim"]
        self.embedding = nn.Embedding(self.vocab.vocab_size(), self.embedding_dim, self.vocab.pad_token_idx())
        self.dropout = config["text_embedding"]["dropout"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def padding(self, array, max_length, padding_value):
        current_length = len(array)
        if current_length < max_length:
            padding_length = max_length - current_length
            padding_array = array + [padding_value] * padding_length
            return torch.tensor(padding_array, dtype= torch.int)
        else:
            return torch.tensor(array[:max_length], dtype= torch.int)
        
    def forward(self, input_texts):
        sequence_vectors = []
        for input_text in input_text:
            tokens_to_ids = self.vocab.convert_tokens_to_ids(input_text.split())
            padding_squence = self.padding(tokens_to_ids, self.max_length, self.vocab.pad_token_idx())
            sequence_vectors.append(padding_squence)

        padding_sequences = torch.stack(sequence_vectors, dim = 0).to(self.device)
        out = self.embedding(padding_sequences)
        out = self.dropout(out)
        return out