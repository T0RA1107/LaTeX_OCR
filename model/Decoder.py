import torch.nn as nn
from TransformerModule.Attention import MultiHeadAttention
from TransformerModule.FeedForwardLayer import FeedForward

class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        dim_model,
        dim_head,
        dim_mlp,
        n_head,
        dropout_rate=0.1
    ):
        super().__init__()
        self.dim_model = dim_model
        self.dim_head = dim_head
        self.dim_mlp = dim_mlp
        self.n_head = n_head

        self.layer_norm_masked_attention = nn.LayerNorm(dim_model)
        self.masked_attention = MultiHeadAttention(dim_model, dim_head, dim_head, dim_head, n_head, attention_type="CausalLinearAttention")
        self.dropout_after_masked_attn = nn.Dropout(p=dropout_rate)

        self.layer_norm_cross_attention = nn.LayerNorm(dim_model)
        self.cross_attention = MultiHeadAttention(dim_model, dim_head, dim_head, dim_head, n_head, attention_type="LinearAttention")
        self.dropout_after_cross_attn = nn.Dropout(p=dropout_rate)

        self.layer_norm_mlp = nn.LayerNorm(dim_model)
        self.mlp = FeedForward(dim_model, dim_mlp, dim_model)
        self.dropout_after_mlp = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x,
        memory,
        input_mask,
        memory_mask
    ):
        L, n_batch, dim = x.shape
        assert(dim == self.dim_model)

        # Masked Multi-head Attention
        identity = x
        x = self.layer_norm_masked_attention(x)
        x = self.masked_attention(x, x, x, input_mask)
        x += identity
        x = self.dropout_after_masked_attn(x)

        # Multi-head Cross Attention
        identity = x
        x = self.layer_norm_cross_attention(x)
        x = self.cross_attention(x, memory, memory, memory_mask)
        x += identity
        x = self.dropout_after_cross_attn(x)

        identity = x
        x = self.layer_norm_mlp(x)
        x = self.mlp(x)
        x += identity
        x = self.dropout_after_mlp(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dim_model,
        dim_head,
        dim_mlp,
        n_head,
        depth,
        dropout_rate=0.1
    ):
        super().__init__()
        self.depth = depth
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(dim_model, dim_head, dim_mlp, n_head, dropout_rate=dropout_rate)
        for _ in range(depth)])

    def forward(
        self,
        x,
        memory,
        input_mask,
        memory_mask
    ):
        for i in range(self.depth):
            x = self.decoder[i](x, memory, input_mask, memory_mask)
        return x
