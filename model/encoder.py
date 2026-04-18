import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Attention + Add & Norm
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # FFN + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x