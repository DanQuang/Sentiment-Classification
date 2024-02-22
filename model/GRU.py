import torch
from torch import nn 
import torch.nn.functional as F
from text_module.init_embedding import build_text_embbeding

class GRU(nn.Module):
    def __init__(self, config):
        super(GRU, self).__init__()

        self.num_labels = config["num_labels"]
        self.hidden_units = config["model"]["hidden_units"]
        self.dropout = config["model"]["dropout"]
        self.embedding_dim = config["text_embedding"]["embedding_dim"]
        self.text_embedding = build_text_embbeding(config)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_units,
                          num_layers=config['model']['num_layers'],dropout=self.dropout)
        
        self.fc = nn.LazyLinear(self.num_labels)

    def forward(self, texts):
        embbed = self.text_embedding(texts)
        gru_output, _ = self.gru(embbed)
        out = torch.mean(gru_output, dim = 1)
        logits = self.fc(out)
        return logits