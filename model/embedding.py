import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model =d_model
        self.vocab_size =vocab_size
        self.embedding = nn.Embedding(self.vocab_size,self.d_model)

    def forward(self,x):
        return self.embedding(x)* torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        


