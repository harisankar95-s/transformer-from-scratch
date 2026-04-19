import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_model // self.n_heads
        self.Wq = nn.Linear(self.d_model, self.d_model)
        self.Wk = nn.Linear(self.d_model, self.d_model)
        self.Wv = nn.Linear(self.d_model, self.d_model)
        self.Wo = nn.Linear(self.d_model, self.d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output

    def forward(self, Q_input, K_input=None, V_input=None, mask=None):
        if K_input is None:
            K_input = Q_input
        if V_input is None:
            V_input = Q_input

        batch_size = Q_input.size(0)
        seq_length = Q_input.size(1)
        seq_length_k = K_input.size(1)

        Q = self.Wq(Q_input)
        K = self.Wk(K_input)
        V = self.Wv(V_input)

        Q = Q.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_length_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length_k, self.n_heads, self.d_k).transpose(1, 2)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.Wo(attn_output)
        return output