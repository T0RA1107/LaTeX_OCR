import torch.nn as nn
import sys
sys.path.append("/Users/tora/Desktop/DL/LaTeX_OCR/model")
from TransformerModule.Embedding import SinusoidPostionalEmbedding, Embedding4Transformer
from Encoder import TransformerEncoder
from Decoder import TransformerDecoder

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViTLaTeXOCR(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        dim_emb,
        depth,
        n_head,
        dim_mlp,
        vocab_size,
        max_L=100,
        channels=1,
        dim_head=64
    ):
        super().__init__()
        self.H, self.W = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)
        assert self.H % self.patch_height == 0 and self.W % self.patch_width == 0, "Impossible value pairs to make patch"
        self.h, self.w = self.H // self.patch_height, self.W // self.patch_width

        self.channels = channels
        self.patch_num = self.h * self.w
        self.patch_dim = self.patch_height * self.patch_width * self.channels
        self.dim_emb = dim_emb

        # Image Embeddings
        self.image_embedding = nn.Sequential(
            SinusoidPostionalEmbedding(self.patch_num, self.patch_dim),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim_emb),
            nn.LayerNorm(dim_emb)
        )
        # Word Embeddings
        self.word_embedding = Embedding4Transformer(dim_emb, vocab_size, max_L)
        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(dim_emb, dim_head, dim_mlp, n_head, depth)
        # Transformer Decoder
        self.transformer_decoder = TransformerDecoder(dim_emb, dim_head, dim_mlp, n_head, depth)

    def forward(
        self,
        image,
        tgt,
        tgt_mask,
        memory_mask
    ):
        n_batch, H, W, C = image.shape
        assert self.H == H and self.W == W and self.channels == C, f"Invalid shape of the images: the shape must be ({self.H}, {self.W}, {self.channels}), and yours are ({H}, {W}, {C})"
        x = image.reshape(n_batch, self.h, self.patch_height, self.w, self.patch_width, C)\
                 .permute(0, 1, 3, 2, 4, 5)\
                 .reshape(n_batch, self.patch_num, self.patch_dim)\
                 .permute(1, 0, 2)
        # Embedding
        x = self.image_embedding(x)  # (self.patch_num, n_batch, self.dim_emb)

        # Transformer Encoder
        memory = self.transformer_encoder(x)

        # Transformer Decoder
        tgt_emb = self.word_embedding(tgt)
        tgt_out = self.transformer_decoder(tgt_emb, memory, tgt_mask, memory_mask)
        tgt_out = self.word_embedding.decode(tgt_out)
        return tgt_out

    def generate(
        self,
        image
    ):
        pass
