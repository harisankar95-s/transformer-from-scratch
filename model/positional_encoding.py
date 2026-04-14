import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_seq_length):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.positional_encoding = torch.zeros(max_seq_length,d_model)
        self.position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        self.pair_length = self.d_model//2
        self.column_pair = torch.arange(0,self.pair_length)
        self.two_i = (self.column_pair*2).float()
        self.div_term = torch.pow(10000, self.two_i / self.d_model)
        self.positional_encoding[:, 0::2] = torch.sin(self.position / self.div_term)
        self.positional_encoding[:, 1::2] = torch.cos(self.position / self.div_term)
        self.register_buffer('pe', self.positional_encoding.unsqueeze(0))

    def forward(self,x):
        return x + self.pe[:, :x.size(1), :]





