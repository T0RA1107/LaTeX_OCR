import os
from hydra import initialize_config_dir, compose
from dataset.dataset import LaTeXDataset
from model.OCRmodel import ViTLaTeXOCR
from train import train
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

config_path = os.path.dirname(__file__)

def main():
    initialize_config_dir(config_dir=config_path, version_base=None)
    config = compose(config_name="config.yaml")
    # print(config)
    train_dataset = LaTeXDataset(
        config.dataset.train_data_path_lst,
        config.dataset.all_images_dir,
        config.dataset.all_formulae_lst,
        config.dataset.vocab_txt,
        config.dataset.max_seq_length,
        (config.dataset.input_shape.x, config.dataset.input_shape.y)
    )
    valid_dataset = LaTeXDataset(
        config.dataset.valid_data_path_lst,
        config.dataset.all_images_dir,
        config.dataset.all_formulae_lst,
        config.dataset.vocab_txt,
        config.dataset.max_seq_length,
        (config.dataset.input_shape.x, config.dataset.input_shape.y)
    )
    model = ViTLaTeXOCR(
        (config.model.image_size.x, config.model.image_size.y),
        config.model.patch_size,
        config.model.dim_emb,
        config.model.depth,
        config.model.n_head,
        config.model.dim_mlp,
        train_dataset.vocab_size,
        config.model.max_L
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
    train(model, dataloader_train, dataloader_valid, loss_function, optim.AdamW, config)

if __name__ == "__main__":
    main()
