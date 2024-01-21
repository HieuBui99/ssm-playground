import json
import argparse
import tiktoken
import numpy as np
import torch
from pathlib import Path
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# generattion config
max_length = 256
top_k = 1
top_p = 0.0
temperature = 1.0


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained-path", type=str)
args = parser.parse_args()

pretrained_path = Path(args.pretrained_path)

with open(pretrained_path / "config.json", "r") as f:
    config_dict = json.load(f)
config = MambaConfig(**config_dict)
mamba = MambaLMHeadModel(config)

mamba.load_state_dict(torch.load("/mnt/sdc/Hieu/learn_ssm/pretrained/pytorch_model.bin"))
mamba.cuda()

tokenizer = tiktoken.get_encoding("gpt2")
text = "A great man once said"
input_ids = torch.LongTensor([tokenizer.encode_ordinary(text)]).cuda(0)

out = mamba.generate(input_ids=input_ids, max_length=len(input_ids) + 256)
out = tokenizer.decode_batch(out.cpu().numpy())[0].split("\n\n")

for line in out:
    print(line)
