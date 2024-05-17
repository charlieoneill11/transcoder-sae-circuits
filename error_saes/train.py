from gated_sae import GatedSAE
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
import torch.optim as optim
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

# Load in the sae errors and original z
sae_errors = torch.load('data/sae_errors.pt')
original_z = torch.load('data/original_z.pt')

# Create GatedSAE
n_input_features = 768
projection_up = 4
gated_sae = GatedSAE(n_input_features=768, n_learned_features=n_input_features*projection_up)


# Create GatedSAE dataset
class GatedSAEDataset(Dataset):
    def __init__(self, original_z, sae_errors):
        self.original_z = original_z
        self.sae_errors = sae_errors

    def __len__(self):
        return len(self.original_z)

    def __getitem__(self, idx):
        return self.original_z[idx], self.sae_errors[idx]
    
gated_sae_dataset = GatedSAEDataset(original_z, sae_errors)

# Create GatedSAE train dataloader and test dataloader
train_size = int(0.8 * len(gated_sae_dataset))
test_size = len(gated_sae_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(gated_sae_dataset, [train_size, test_size])

batch_size=32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

torch.set_grad_enabled(True)

n_epochs = 10
gated_sae = GatedSAE(n_input_features=768, n_learned_features=768*4)
optimizer = optim.Adam(gated_sae.parameters(), lr=0.001)

for epoch in range(n_epochs):
    print(f"Epoch {epoch}")
    for i, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        sae_out, loss = gated_sae(x, y)
        loss.backward()
        print(f"Batch {i} Loss {loss.item()}")
        optimizer.step()
        if i % (n_epochs // 10) == 0:
            # Evaluate on test set
            test_loss = 0
            for x, y in test_dataloader:
                sae_out, loss = gated_sae(x, y)
                test_loss += loss.item()

# Save to data folder
torch.save(gated_sae, 'data/gated_sae.pt')