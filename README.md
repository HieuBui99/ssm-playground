# ssm-playground
Finetuning Mamba with fastai & pytorch
> **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**\
> Albert Gu*, Tri Dao*\
> Paper: https://arxiv.org/abs/2312.00752

## About

Mamba is a new state space model architecture showing promising performance on information-dense data such as language modeling, where previous subquadratic models fall short of Transformers.
It is based on the line of progress on [structured state space models](https://github.com/state-spaces/s4),
with an efficient hardware-aware design and implementation in the spirit of [FlashAttention](https://github.com/Dao-AILab/flash-attention).

## Installation

```
pip install torch fastai transformers accelerate
```

## Toy example
Run `prepare_shakespeare.py` to download and prepare the Shakespeare dataset
```python
python prepare_shakespeare.py
```

Train a Mamba language model from scratch
```python
python train_shapespeare.py --data-path <output-of-last-step> --config-path config/mamba_config.json --save-dir output
```

## Finetuning with fastai & accelerate
First download the [SQUAD](https://huggingface.co/datasets/squad) dataset for question answering.

Config accelerate
```
accelerate config
```
or run this snippet to auto config your environment
```python
from accelerate.utils import write_basic_config
write_basic_config()
```
If you only have 1 GPU:
```
python finetune_qa.py --data-path <path-to-parquet-file> --batch-size 2 --num-epochs 5 --output output
```
Multi-gpu training
```
accelerate launch finetune_qa.py --data-path <path-to-parquet-file> --batch-size 2 --num-epochs 5 --output output
```
A 1.4B params Mamba model takes about 20gb of VRAM with a batch size of 2.
