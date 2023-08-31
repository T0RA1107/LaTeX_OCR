import numpy as np
import torch
import wandb
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, dynamic_ncols=True)

def train(model, dataloader_train, dataloader_valid, loss_fn, optimizer, config):
    if config.wandb:
        wandb.init(project=config.project, name=config.exp_name)
        wandb.config.n_epochs = n_epochs
    n_epochs = config.train.n_epochs
    device = torch.device(config.train.device)
    model.to(device)
    optimizer = optimizer(model.parameters(), lr=config.train.lr)
    train_losses = []
    valid_losses = []
    for epoch in tqdm(range(1, n_epochs + 1), desc="EPOCHS"):
        n_train = 0
        losses_train = []

        model.train()

        for img, tgt in tqdm(dataloader_train, leave=False, desc="train"):
            n_train += tgt.size()[0]
            model.zero_grad()

            img = img.permute(0, 2, 3, 1)
            tgt = tgt.transpose(1, 0)
            img = img.to(device)
            tgt = tgt.to(device)

            y = model(img, tgt, None, None)

            loss = loss_fn(y, tgt)

            loss.backward()

            optimizer.step()

            losses_train.append(loss.tolist())

        n_valid = 0
        losses_valid = []

        model.eval()
        for img, tgt in tqdm(dataloader_valid, leave=False, desc="valid"):
            n_valid += tgt.size()[0]
            img = img.permute(0, 2, 3, 1)
            tgt = tgt.transpose(1, 0)
            img = img.to(device)
            tgt = tgt.to(device)

            y = model(img, tgt, None, None)

            loss = loss_fn(y, tgt)

            loss.backward()

            losses_valid.append(loss.tolist())

        loss_train = np.mean(losses_train)
        loss_valid = np.mean(losses_valid)
        train_losses.append(loss_train)
        valid_losses.append(loss_valid)
        if config.wandb:
            wandb.log({
                'train loss': loss_train,
                'valid loss': loss_valid
            })

    if config.wandb:
        wandb.finish()
