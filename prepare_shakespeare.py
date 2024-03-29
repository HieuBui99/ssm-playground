import os
import requests
import tiktoken
import numpy as np
from transformers import AutoTokenizer

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tokenizer
# enc = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.int32)
val_ids = np.array(val_ids, dtype=np.int32)

np.save("train.npy", train_ids)
np.save("val.npy", val_ids)

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens