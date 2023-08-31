import torch.nn as nn
from TransformerModule.Attention import MultiHeadAttention
from TransformerModule.FeedForwardLayer import FeedForward

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim_model,
        dim_head,
        dim_mlp,
        n_head
    ):
        super().__init__()
        self.dim_model = dim_model
        self.n_head = n_head

        self.layer_norm_attention = nn.LayerNorm(dim_model)
        self.attention = MultiHeadAttention(dim_model, dim_head, dim_head, dim_head, n_head)

        self.layer_norm_mlp = nn.LayerNorm(dim_model)
        self.mlp = FeedForward(dim_model, dim_mlp, dim_model)

    def forward(
        self,
        x
    ):
        L, n_batch, dim = x.shape
        assert dim == self.dim_model

        identity = x
        x = self.layer_norm_attention(x)
        x = self.attention(x, x, x)
        x += identity

        identity = x
        x = self.layer_norm_mlp(x)
        x = self.mlp(x)
        x += identity

        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim_model,
        dim_head,
        dim_mlp,
        n_head,
        depth
    ):
        super().__init__()
        self.depth = depth
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(dim_model, dim_head, dim_mlp, n_head)
        for _ in range(depth)])

    def forward(
        self,
        x
    ):
        for i in range(self.depth):
            x = self.encoder[i](x)
        return x

