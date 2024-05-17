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

_, z_saes, _ = get_model_encoders(device='cpu')
layer = 9
sae = z_saes[layer]
del z_saes


class SAEDataset(Dataset):
    def __init__(self, z_acts):
        self.z_acts = z_acts

    def __len__(self):
        return len(self.z_acts)

    def __getitem__(self, idx):
        return self.z_acts[idx]
    
# Load z_acts
z_acts = torch.load('data/z_acts.pt')
    
sae_dataset = SAEDataset(z_acts)

# Create SAE dataloader
batch_size = 64
sae_dataloader = DataLoader(sae_dataset, batch_size=batch_size, shuffle=True)

print(next(iter(sae_dataloader)).shape)

# Get SAE errors on each z_acts - we need to store the errors, and the original z_acts
sae_errors = []
original_z = []
for z_batch in tqdm(sae_dataloader):
    _, z_recon, z_acts, _, _ = sae(z_batch)
    sae_error = z_batch - z_recon
    sae_errors.append(sae_error)
    original_z.append(z_batch)
    
# Stack all sae errors along first dimension
sae_errors = torch.cat(sae_errors, dim=0)
original_z = torch.cat(original_z, dim=0)

# Print shapes
print(sae_errors.shape)
print(original_z.shape)

# Save both
torch.save(sae_errors, 'data/sae_errors.pt')
torch.save(original_z, 'data/original_z.pt')