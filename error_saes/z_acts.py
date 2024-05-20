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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, z_saes, transcoders = get_model_encoders(device=device)

batch_size = 16

class TokenizedDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        return self.tokenized_dataset[idx]
    
tokenized_dataset = torch.load('data/tokenized_dataset.pt')
dataset = TokenizedDataset(tokenized_dataset)
print(f"Length of tokenised dataset = {len(tokenized_dataset)}")

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(next(iter(dataloader)).shape)

# Disable torch grad
torch.set_grad_enabled(False)
layer = 9
sae = z_saes[layer]

# Get all z activations
z_acts = []
for batch in tqdm(dataloader):
    logits, cache = model.run_with_cache(batch)
    z = cache["z", layer] # batch_size x seq_len x n_heads x d_head
    del logits
    del cache
    z = einops.rearrange(
        z, 
        "b s n d -> (b s) (n d)"
    )
    z_acts.append(z)

# Stack all z activations along first dimension
z_acts = torch.cat(z_acts, dim=0)
print(f"Z acts shape = {z_acts.shape}")

torch.save(z_acts, 'data/z_acts.pt')