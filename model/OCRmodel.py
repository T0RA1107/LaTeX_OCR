import torch.nn as nn
import torch
import sys
sys.path.append("./model")
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
        dropout_rate=0.1,
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
        self.dropout_after_image_emb = nn.Dropout(p=dropout_rate)
        # Word Embeddings
        self.word_embedding = Embedding4Transformer(dim_emb, vocab_size, max_L, pre_train=pre_train_word_embedding, embedding=embedding)
        self.dropout_after_word_emb = nn.Dropout(p=dropout_rate)
        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(dim_emb, dim_head, dim_mlp, n_head, depth, dropout_rate=dropout_rate)
        # Transformer Decoder
        self.transformer_decoder = TransformerDecoder(dim_emb, dim_head, dim_mlp, n_head, depth, dropout_rate=dropout_rate)

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
        x = self.dropout_after_image_emb(x)

        # Transformer Encoder
        memory = self.transformer_encoder(x) # (self.patch_num, n_batch, self.dim_emb)

        # Transformer Decoder
        tgt_emb = self.word_embedding(tgt)
        tgt_emb = self.dropout_after_word_emb(tgt_emb)
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
        image,
        beam_search=False,
        beam_width=10
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

        if beam_search:
            probs = torch.full(size=(n_batch, beam_width), fill_value=torch.log(torch.tensor(1.0 / beam_width))).to(memory.device)
            paths = torch.full(size=(1, n_batch, beam_width), fill_value=0).to(memory.device)
            for _ in range(self.max_L):
                next_probs = torch.empty(size=(n_batch, 0)).to(memory.device)
                next_tokens = torch.empty_like(next_probs).to(memory.device)
                for i in range(beam_width):
                    tgt = self.word_embedding.encode(paths[:, :, i])
                    tgt = self.transformer_decoder(
                        tgt,
                        memory,
                        None,
                        None
                    )
                    tgt = self.word_embedding.decode(tgt[-1, :, :])  # (n_batch, V)
                    p, tgt = tgt.topk(k=beam_width, dim=1)  # (n_batch, beam_width)
                    log_p = torch.log(p) + probs[:, [i]]
                    next_probs = torch.concat([ next_probs, log_p ], dim=1)
                    next_tokens = torch.concat([ next_tokens, tgt ], dim=1)
                scores, idx = next_probs.topk(k=beam_width, dim=1)
                probs = scores
                next_paths = torch.empty(size=(paths.shape[0] + 1, n_batch, beam_width), dtype=torch.long).to(memory.device)
                for b in range(n_batch):
                    for w in range(beam_width):
                        c, next_token_id = idx[b, w] // beam_width, idx[b, w] % beam_width
                        next_paths[:, b, w] = torch.concat([ paths[:, b, c], next_tokens[b, next_token_id][None, ...] ], dim=0)
                paths = next_paths
            res = torch.empty((self.max_L - 1, 0)).to(memory.device)
            idx = probs.argmax(dim=1)
            for b in range(n_batch):
                res = torch.concat([ res, paths[1:, b, idx[b]]])
            return res
        else:
            tgt_in = tgt = torch.full(size=(1, n_batch), fill_value=0).to(memory.device)
            # tgt_mask = torch.tril(torch.ones(self.max_L, self.max_L, dtype=int), diagonal=0)
            # tgt_mask = torch.where(tgt_mask == 1, float(0.0), float("-inf")).to(memory.device)
            for i in range(self.max_L): # tqdm(range(self.max_L), desc="Auto-regressive Generation", leave=False):
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

class BeamPath:
    def __init__(
        self,
        path,  # (l, n_batch)
        probability # (1, n_batch)
    ):
        self.path = path
        self.probability = probability

    def next_token_probabilities(
        self,
        next_prob  # (1, n_batch, V)
    ):
        return self.probability[:, :, None] * next_prob

    def update_path(
        self,
        next_token,  # (1, n_batch)
        probability  # (1, n_batch)
    ):
        self.path = torch.cat([ self.path, next_token ], dim=0)
        self.probability = probability

class BeamSearcher:
    def __init__(self, n_batch, beam_width, device):
        self.n_batch = n_batch
        self.beam_width = beam_width
        self.paths = [BeamPath(torch.full(size=(1, n_batch), fill_value=0).to(device), 1.0 / beam_width) for _ in range(beam_width)]

    def determine_path(
        self,
        next_probabilities  # [(1, n_batch, V); beam_width]
    ):
        l, n_batch, V = next_probabilities.shape
        assert l == 1 and n_batch == self.n_batch
        prob_stack = torch.cat(next_probabilities, dim=2) # (1, n_batch, V * beam_width)
        prob, idx = torch.topk(prob_stack, k=self.beam_width, dim=-1)
