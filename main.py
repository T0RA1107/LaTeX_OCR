import os
from hydra import initialize_config_dir, compose
from dataset.dataset import LaTeXDataset
from model.OCRmodel import ViTLaTeXOCR
from train import train
import numpy as np
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

config_path = os.path.dirname(__file__)

def main():
    initialize_config_dir(config_dir=config_path, version_base=None)
    config = compose(config_name="config.yaml")
    train_dataset = LaTeXDataset(
        config.dataset.train_images_dir,
        config.dataset.all_formulae_lst,
        config.dataset.vocab_txt,
        config.dataset.max_seq_length,
        (config.dataset.input_shape.x, config.dataset.input_shape.y)
    )
    valid_dataset = LaTeXDataset(
        config.dataset.valid_images_dir,
        config.dataset.all_formulae_lst,
        config.dataset.vocab_txt,
        config.dataset.max_seq_length,
        (config.dataset.input_shape.x, config.dataset.input_shape.y)
    )

    embedding = None
    if config.model.pre_train_word_embedding:
        embedding_model = KeyedVectors.load_word2vec_format(config.model.embedding_path, binary=True)
        embedding = np.empty((train_dataset.vocab_size, config.model.dim_emb))
        for token, idx in train_dataset.word2id.items():
            embedding[idx] = embedding_model[token]
        embedding = torch.from_numpy(embedding).to(torch.float32)

    model = ViTLaTeXOCR(
        (config.model.image_size.x, config.model.image_size.y),
        config.model.patch_size,
        config.model.dim_emb,
        config.model.depth,
        config.model.n_head,
        config.model.dim_mlp,
        train_dataset.vocab_size,
        config.model.max_L,
        pre_train_word_embedding=config.model.pre_train_word_embedding,
        embedding=embedding
    )
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        drop_last=True
    )
    dataloader_valid = DataLoader(
        valid_dataset,
        batch_size=config.dataset.batch_size
    )
    loss_function = nn.CrossEntropyLoss()

    if config.model.pre_train_word_embedding:
        param_pre_train = []
        param_model = []
        for name, param in model.named_parameters():
            if name == "word_embedding.token_embedding.embedding.weight":
                param_pre_train.append(param)
            else:
                param_model.append(param)
        assert len(param_pre_train) >= 1
        optimizer = optim.AdamW([
            {'params': param_pre_train, 'lr': config.train.embedding_lr},
            {'params': param_model, 'lr': config.train.lr}
        ])
    else:
        optimizer = optim.AdamW(model.parameters())

    train(model, dataloader_train, dataloader_valid, loss_function, optimizer, config)

if __name__ == "__main__":
    main()
