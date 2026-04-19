import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNorm

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        masked_attn_output = self.masked_attention(x, mask=tgt_mask)
        x = self.norm1(x + masked_attn_output)

        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, mask=src_mask)
        x = self.norm2(x + cross_attn_output)

        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)

        return x