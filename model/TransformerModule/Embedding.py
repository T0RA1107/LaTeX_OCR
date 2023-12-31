import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(
        self,
        d_model, # 分散表現の次元数
        V, # 語彙数
        pre_train=False,
        embedding=None
    ):
        super().__init__()
        self.V = V
        self.d_model = d_model
        if pre_train:
            assert(embedding is not None)
            self.embedding = self.load(embedding)
        else:
            self.embedding = nn.Embedding(V, d_model)

    def encode(
        self,
        x
    ):
        y = self.embedding(x)
        return y # 分散表現の列 (L, n_batch, d_model)

    def decode(
        self,
        x # (L, n_batch, d_model)
    ):
        y = torch.matmul(x, self.embedding.weight.T)
        return y # (L, n_batch, V)

    def load(
        self,
        embedding # 分散表現
    ):
        assert(embedding.shape == (self.V, self.d_model))
        return nn.Embedding.from_pretrained(embedding)

    def forward(self, x):
        return self.encode(x)

class SinusoidPostionalEmbedding(nn.Module):
    def __init__(
        self,
        max_L,
        dim
    ):
        super().__init__()
        self.max_L = max_L
        self.dim_model = dim
        theta_even = torch.arange(max_L)[:, None] / (10000 ** (2 * torch.arange((dim + 1) // 2)[None, :] / dim))
        theta_odd = torch.arange(max_L)[:, None] / (10000 ** (2 * torch.arange(dim // 2)[None, :] / dim))

        pos_encoding_even = torch.sin(theta_even)
        pos_encoding_odd  = torch.cos(theta_odd)

        self.pos_encoding = torch.empty((max_L, dim))
        self.pos_encoding[:, 0::2] = pos_encoding_even
        self.pos_encoding[:, 1::2] = pos_encoding_odd
        self.pos_encoding = self.pos_encoding[:, None, :]

    def forward(
        self,
        inputs
    ):
        L, _, dim = inputs.shape  # (L, n_batch, dim)
        self.pos_encoding = self.pos_encoding[:L].to(inputs.device)
        x = inputs + self.pos_encoding
        return x

class Embedding4Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        V,
        max_L,
        pos_encoding="sinusoid",
        pre_train=False,
        embedding=None
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(d_model, V, pre_train, embedding)
        if pos_encoding == "sinusoid":
            self.positional_encoding = SinusoidPostionalEmbedding(max_L, d_model)
        elif pos_encoding == "trainable":
            self.positional_encoding = nn.Linear(in_features=V, out_features=d_model)

    def encode(
        self,
        x # (L, n_batch,)
    ):
        x = self.token_embedding.encode(x)
        x = x + self.positional_encoding(x)
        return x

    def decode(
        self,
        x # (L, n_batch, d_model)
    ):
        return self.token_embedding.decode(x)

    def forward(
        self,
        x # (L, n_batch,)
    ):
        return self.encode(x)

    def load(
        self,
        embedding
    ):
        self.token_embedding.load(embedding)
