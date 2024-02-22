import torch
from torch import nn
from data_utils.vocab import Vocab

class CountVectorizer(nn.Module):
    def __init__(self, config):
        super(CountVectorizer, self).__init__()
        self.vocab = Vocab(config)
        self.fc = nn.LazyLinear(config["text_embedding"]["embedding_dim"])

    def forward(self, input_texts):
        count_vectors = []
        for input_text in input_texts:
            count_vector = torch.zeros(1, self.vocab.vocab_size())
            for word in input_text.split():
                count_vector[0][self.vocab.word_to_idx.get(word, self.vocab.word_to_idx["<unk>"])] += 1
            count_vectors.append(count_vector)

        count_vectors = torch.stack(count_vectors, dim= 0).to(self.fc.weight.device)
        out = self.fc(count_vectors)
        return out