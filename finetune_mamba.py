import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd
import torch
import transformers
from fastai.distributed import *
from fastai.callback.fp16 import *
from fastai.callback.core import Callback
from fastai.callback.schedule import Learner, fit_flat_cos
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat
from mamba_ssm.models.mixer_seq_simple import MambaConfig, MambaLMHeadModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


class QADataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super(QADataset, self).__init__()

        df = pd.read_parquet(data_path)

        print(f"Got {len(df)} examples, preprocess...")

        self.data, self.labels = self.preprocess(df, tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        inputs, labels =  self.data[i], self.labels[i]
        inputs = torch.nn.functional.pad(inputs, (0, 512-inputs.shape[0]), value=0)
        labels = torch.nn.functional.pad(labels, (0, 512-labels.shape[0]), value=-100)
        # print(inputs.shape, labels.shape)
        return inputs, labels
        # return dict(inputs=self.data[i], labels=self.labels[i])

    def preprocess(self, df, tokenizer):
        """
        Preprocess the data by tokenizing.
        """
        data = []
        labels = []
        cut = 20000
        contexts = df["context"].to_list()[:cut]
        questions = df["question"].to_list()[:cut]
        answers = df["answers"].to_list()[:cut]
        print("Tokenizing dataset...")
        for i, (c, q, a) in tqdm(enumerate(zip(contexts, questions, answers))):
            # Add a positive example
            text = f"{c}\n\nQ: {q}\nA: {a}\n<|endoftext|>"
            tokenized = torch.LongTensor(tokenizer.encode(text))
            data.append(tokenized[:-1])
            labels.append(tokenized[1:])
            # Generate a negative example
            random_idx = random.randint(0, len(contexts) - 1)
            text = f"{contexts[random_idx]}\n\nQ: {q}\nA: I don't know.\n<|endoftext|>"
            tokenized = torch.LongTensor(tokenizer.encode(text))
            data.append(tokenized[:-1])
            labels.append(tokenized[1:])

        return data, labels


class DataCollatorForSFTDataset(object):
    """
    Collator for supervised fine-tuning.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        inputs, labels = tuple([b[key] for b in batch] for key in ("inputs", "labels"))
        # inputs = torch.nn.utils.rnn.pad_sequence(
        #     inputs, batch_first=True, padding_value=self.tokenizer.eos_token_id
        # )
        # labels = torch.nn.utils.rnn.pad_sequence(
        #     labels, batch_first=True, padding_value=-100
        # )
        assert inputs.shape == labels.shape
        return inputs, labels


class MambaCallback(Callback):
    def after_pred(self):
        self.learn.pred = self.learn.pred.logits

def train(model, dls, n_epochs, output_dir):
    learn = Learner(
        dls,
        model,
        loss_func=CrossEntropyLossFlat(),
        cbs=[MambaCallback()],
    )
    with learn.distrib_ctx(gradient_accumulation_steps=2):
        learn.fit_flat_cos(n_epochs, 1e-4, wd=1e-5, pct_start=0.2)
    learn.model.save_pretrained(output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--save-dir", type=str, default="output")
    parser.add_argument("--config-path", type=str, default="config/mamba_large.json")
    parser.add_argument("--pretrained_path", type=str, default="pretrained/pytorch_model.bin")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=5)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config_dict = json.load(f)
    print("Initializing model")
    config = MambaConfig(**config_dict)
    mamba = MambaLMHeadModel(config)
    mamba.load_state_dict(torch.load(args.pretrained_path, map_location="cpu"))


    data_path = Path(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b", add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    # collator = DataCollatorForSFTDataset(tokenizer)

    train_ds = QADataset(data_path / "squad_train.parquet", tokenizer)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        # collate_fn=collator,
        pin_memory=True,
        num_workers=0,
        drop_last=True
    )
    val_ds = QADataset(data_path / "squad_valid.parquet", tokenizer)
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=True,
        # collate_fn=collator,
        pin_memory=True,
        num_workers=0,
        drop_last=True
    )
    dls = DataLoaders(train_dl, val_dl)

    train(mamba, dls, args.num_epochs, args.output)
if __name__ == "__main__":
    main()


