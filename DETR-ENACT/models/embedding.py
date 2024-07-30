import torch.nn.functional as F
from torch import nn

class Embedding(nn.Module):
    def __init__(self, d_embed):
        super().__init__()

        self.d_embed = d_embed
        self.embed_layer = nn.Linear(d_embed, d_embed, bias=False)
    
    def forward(self, query):
        return self.embed_layer(query)
