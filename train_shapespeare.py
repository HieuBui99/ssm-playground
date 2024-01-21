import argparse
import json
import random

import numpy as np
import torch
from fastai.learner import Learner
from fastai.callback.core import Callback
from fastai.callback.schedule import Learner, fit_flat_cos
from fastai.data.core import DataLoaders
from fastai.losses import CrossEntropyLossFlat
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch.utils.data import DataLoader, Dataset


class LMDataset(Dataset):
    def __init__(self, tokens, block_size=256):
        self.tokens = tokens
        self.block_size = block_size

    def __getitem__(self, idx):
        ix = random.randint(0, len(self.tokens) - self.block_size - 2)

        x = torch.from_numpy(self.tokens[ix : ix + self.block_size].astype(np.int64))
        y = torch.from_numpy(
            self.tokens[ix + 1 : ix + 1 + self.block_size].astype(np.int64)
        )

        return x, y

    def __len__(self):
        return len(self.tokens) // self.block_size


class MambaCallback(Callback):
    def after_pred(self):
        self.learn.pred = self.learn.pred.logits


def train(model, dataloaders, save_dir):
    learn = Learner(
        dataloaders,
        model,
        loss_func=CrossEntropyLossFlat(),
        cbs=[MambaCallback()],
    )
    learn.fit_flat_cos(10, 5e-4, wd=1e-5, pct_start=0.3)
    learn.model.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--save-dir", type=str)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config_dict = json.load(f)

    config = MambaConfig(**config_dict)
    mamba = MambaLMHeadModel(config)

    train_ds = LMDataset(np.load(f"{args.data_path}/train.npy"))
    val_ds = LMDataset(np.load(f"{args.data_path}/val.npy"))

    dls = DataLoaders(
        DataLoader(
            train_ds,
            batch_size=32,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=8,
        ),
        DataLoader(
            val_ds,
            batch_size=32,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=8,
        ),
    )

    train(mamba, dls, args.save_dir)
