import numpy as np
import torch
import wandb
import datetime
import os
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, dynamic_ncols=True)

import glob
import subprocess
import random


import torch.nn.functional as F

class TrainManager:
    def __init__(
        self,
        train_loss_init,
        valid_loss_init,
        update_rate=0.9
    ):
        self.train_losses = []
        self.valid_losses = []
        self.train_loss_best = train_loss_init
        self.valid_loss_best = valid_loss_init
        self.update_rate = update_rate

    def check_and_save_weight(self, loss_train, loss_valid):
        self.train_losses.append(loss_train)
        self.valid_losses.append(loss_valid)
        if loss_train < self.train_loss_best * self.update_rate and loss_valid < self.valid_loss_best * self.update_rate:
            self.train_loss_best = loss_train
            self.valid_loss_best = loss_valid
            return True
        else:
            return False

def train(model, dataloader_train, dataloader_valid, loss_fn, optimizer, config):
    n_epochs = config.train.n_epochs
    if config.wandb:
        wandb.init(project=config.project, name=config.exp_name)
        wandb.config.n_epochs = n_epochs
    if config.train.save_weight:
        time_stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M')
        save_path_dir = os.path.join(config.train.weight_path, time_stamp)
        os.makedirs(save_path_dir)
    device = torch.device(config.train.device)
    model.to(device)
    train_manager = TrainManager(2.5, 2.5)
    for epoch in tqdm(range(1, n_epochs + 1), desc="EPOCHS"):
        n_train = 0
        losses_train = []

        model.train()

        for img, tgt in dataloader_train: # tqdm(dataloader_train, leave=False, desc="train"):
            n_train += tgt.size()[0]
            model.zero_grad()

            img = img.permute(0, 2, 3, 1)
            tgt = tgt.transpose(1, 0)
            img = img.to(device)
            tgt = tgt.to(device)
            tgt_in = tgt[:-1, :]
            tgt_out = tgt[1:, :]

            y = model(img, tgt_in)

            V = y.shape[2]

            loss = loss_fn(y.reshape(-1, V), tgt_out.reshape(-1))

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
            tgt_in = tgt[:-1, :]
            tgt_out = tgt[1:, :]

            y = model(img, tgt_in)

            V = y.shape[2]

            loss = loss_fn(y.reshape(-1, V), tgt_out.reshape(-1))

            losses_valid.append(loss.tolist())

        loss_train = np.mean(losses_train)
        loss_valid = np.mean(losses_valid)
        if train_manager.check_and_save_weight(loss_train, loss_valid) and config.train.save_weight:
            torch.save(model, os.path.join(save_path_dir, f'{epoch}.pth'))
            print("\rsave weight on", os.path.join(save_path_dir, f'{epoch}.pth'))
        if config.wandb:
            wandb.log({
                'train loss': loss_train,
                'valid loss': loss_valid
            })
        # TODO: Loggerに直す
        # with open(config.train.mem_file) as f:
        #     f.write("EPOCH: {}\ntrain loss: {:.4f} valid loss: {:.4f}\n".format(epoch, loss_train, loss_valid))

    if config.wandb:
        wandb.finish()
