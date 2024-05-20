import torch
import einops
import sys
sys.path.append('../src')

from circuit_lens import get_model_encoders
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int
from torch import Tensor
import torch
import einops
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import List, Dict, TypedDict, Any, Union, Tuple, Optional
from tqdm import trange
from pprint import pprint
from transformer_lens.utils import get_act_name, to_numpy
from enum import Enum
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm 


model, z_saes, transcoders = get_model_encoders(device='cpu')
print("Model, z_saes, transcoders loaded.")

dataset = load_dataset("NeelNanda/pile-10k")
print("Dataset loaded.")

# Split the huggingface dataset up into seq_len text
seq_len = 128
batch_size = 4096
model_name = 'gpt2-small'

model = HookedTransformer.from_pretrained(model_name, device='cpu')

tokenized_dataset = []
# Concat all the text together
text = " ".join(dataset['train']['text'])

# Tokenize the text
for i in trange(0, len(text)//100, 2500):
    tokens = model.to_tokens(text[i:i+2500]).squeeze()
    # Split into seq_len chunks
    for j in range(0, len(tokens), seq_len):
        tokenized_dataset.append(tokens[j:j+seq_len]) # tests

# Keep only examples with seq_len 128
tokenized_dataset = [x for x in tokenized_dataset if len(x) == seq_len]

print(f"Length of tokenised dataset = {len(tokenized_dataset)}")

# Assert all tensors have shape seq_length
for i, tokens in enumerate(tokenized_dataset):
    assert tokens.shape[0] == seq_len, f"Token {i} has shape {tokens.shape}"


# Save to data as a torch tensor
tokenized_dataset = torch.stack(tokenized_dataset)
torch.save(tokenized_dataset, 'data/tokenized_dataset.pt')
print(f"Tokenized dataset with {len(tokenized_dataset)} examples with {seq_len} tokens.")