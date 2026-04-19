import torch
import torch.nn as nn
from model.embedding import TokenEmbedding
from model.positional_encoding import PositionalEncoding
from model.encoder import EncoderLayer
from model.decoder import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length):
        super().__init__()
        self.encoder_embedding = TokenEmbedding(d_model, src_vocab_size)
        self.decoder_embedding = TokenEmbedding(d_model, tgt_vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.positional_encoding(self.encoder_embedding(src))
        tgt = self.positional_encoding(self.decoder_embedding(tgt))

        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, src_mask)

        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(tgt, src, src_mask, tgt_mask)

        output = self.output_projection(tgt)
        return output