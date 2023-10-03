import torch.nn as nn
import torch
import sys
sys.path.append("/Users/tora/Desktop/DL/LaTeX_OCR/model")
from TransformerModule.Embedding import SinusoidPostionalEmbedding, Embedding4Transformer
from Encoder import TransformerEncoder
from Decoder import TransformerDecoder
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, dynamic_ncols=True)

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
        dim_head=64,
        pre_train_word_embedding=False,
        embedding=None
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
        self.max_L = max_L

        # Image Embeddings
        self.image_embedding = nn.Sequential(
            SinusoidPostionalEmbedding(self.patch_num, self.patch_dim),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim_emb),
            nn.LayerNorm(dim_emb)
        )
        # Word Embeddings
        self.word_embedding = Embedding4Transformer(dim_emb, vocab_size, max_L, pre_train=pre_train_word_embedding, embedding=embedding)
        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(dim_emb, dim_head, dim_mlp, n_head, depth)
        # Transformer Decoder
        self.transformer_decoder = TransformerDecoder(dim_emb, dim_head, dim_mlp, n_head, depth)

    def forward(
        self,
        image,
        tgt
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
        # tgt_mask = self.generate_mask(tgt_emb).to(tgt_emb.device)
        tgt_out = self.transformer_decoder(
            tgt_emb,
            memory,
            None, # tgt_mask,
            None
        )
        tgt_out = self.word_embedding.decode(tgt_out)
        return tgt_out

    def generate(
        self,
        image
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

        tgt_in = tgt = torch.full(size=(1, n_batch), fill_value=0).to(memory.device)
        # tgt_mask = torch.tril(torch.ones(self.max_L, self.max_L, dtype=int), diagonal=0)
        # tgt_mask = torch.where(tgt_mask == 1, float(0.0), float("-inf")).to(memory.device)
        for i in tqdm(range(self.max_L), desc="Auto-regressive Generation", leave=False):
            tgt = self.word_embedding.encode(tgt_in)
            tgt = self.transformer_decoder(
                tgt,
                memory,
                None, # tgt_mask[:i + 1, :i + 1],
                None
            )
            tgt = self.word_embedding.decode(tgt[[-1], :, :])
            _, tgt = torch.max(tgt, dim=-1)
            tgt_in = torch.cat([ tgt_in, tgt ], dim=0)
        tgt_out = tgt_in[1:, :]
        return tgt_out

    def generate_mask(self, tgt):
        L_tgt = tgt.shape[0]
        tgt_mask = torch.tril(torch.ones(L_tgt, L_tgt, dtype=int), diagonal=0)
        tgt_mask = torch.where(tgt_mask == 1, float(0.0), float("-inf"))
        return tgt_mask
