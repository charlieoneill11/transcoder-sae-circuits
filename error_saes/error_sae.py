# %%
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

# %%
model, z_saes, transcoders = get_model_encoders(device='cpu')

# %% [markdown]
# ## Load the Pile dataset we'll use for activations

# %%

dataset = load_dataset("NeelNanda/pile-10k")

# %%
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
        tokenized_dataset.append(tokens[j:j+seq_len])

# %%
# Keep only examples with seq_len 128
tokenized_dataset = [x for x in tokenized_dataset if len(x) == seq_len]

# %%
len(tokenized_dataset)

# %%
# Assert all tensors have shape seq_length
for i, tokens in enumerate(tokenized_dataset):
    assert tokens.shape[0] == seq_len, f"Token {i} has shape {tokens.shape}"

# %%
# Turn tokenized_dataset (a list of tensors) into a Pytorch Dataset
from torch.utils.data import Dataset

batch_size = 16

class TokenizedDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        return self.tokenized_dataset[idx]
    
dataset = TokenizedDataset(tokenized_dataset)

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(next(iter(dataloader)).shape)

# %%
# Disable torch grad
torch.set_grad_enabled(False)

# %%
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
z_acts.shape

# %%
torch.save(z_acts, 'z_acts.pt')

# %%
# Load z_acts
z_acts = torch.load("z_acts.pt")

# %% [markdown]
# ## Get SAE reconstructions and errors

# %%
z_acts.shape

# %%
# Create SAE dataset
class SAEDataset(Dataset):
    def __init__(self, z_acts):
        self.z_acts = z_acts

    def __len__(self):
        return len(self.z_acts)

    def __getitem__(self, idx):
        return self.z_acts[idx]
    
sae_dataset = SAEDataset(z_acts)

# Create SAE dataloader
sae_dataloader = DataLoader(sae_dataset, batch_size=batch_size, shuffle=True)

print(next(iter(sae_dataloader)).shape)

# %%
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

# %%
sae_errors.shape, original_z.shape

# %%
# Save both
torch.save(sae_errors, 'sae_errors.pt')
torch.save(original_z, 'original_z.pt')

# %% [markdown]
# ## Train a gated SAE to predict the errors


# %%
# Load in the sae errors and original z
sae_errors = torch.load('sae_errors.pt')
original_z = torch.load('original_z.pt')

# %%
# Gated SAE
class GatedSAE(nn.Module):

    def __init__(self, n_input_features, n_learned_features, l1_coefficient=0.01):

        super().__init__()

        self.n_input_features = n_input_features
        self.n_learned_features = n_learned_features
        self.l1_coefficient = l1_coefficient

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.n_input_features, self.n_learned_features))   
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.n_learned_features, self.n_input_features))   
        )

        self.r_mag = nn.Parameter(
            torch.zeros(self.n_learned_features)
        )
        self.b_mag = nn.Parameter(
            torch.zeros(self.n_learned_features)
        )
        self.b_gate = nn.Parameter(
            torch.zeros(self.n_learned_features)
        )
        self.b_dec = nn.Parameter(
            torch.zeros(self.n_input_features)
        )

        self.activation_fn = nn.ReLU()

    def forward(self, x_act, y_error):
        # Assert x_act (original z activations i.e. the input) and the y_error (SAE error i.e. the target) have the same shape
        assert x_act.shape == y_error.shape, f"x_act shape {x_act.shape} does not match y_error shape {y_error.shape}"

        hidden_pre = einops.einsum(x_act, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")

        # Gated SAE
        hidden_pre_mag = hidden_pre * torch.exp(self.r_mag) + self.b_mag
        hidden_post_mag = self.activation_fn(hidden_pre_mag)  
        hidden_pre_gate = hidden_pre + self.b_gate
        hidden_post_gate = (torch.sign(hidden_pre_gate) + 1) / 2
        hidden_post = hidden_post_mag * hidden_post_gate

        sae_out = einops.einsum(hidden_post, self.W_dec, "... d_sae, d_sae d_in -> ... d_in") + self.b_dec

        # Now we need to handle all the loss stuff
        # Reconstruction loss
        per_item_mse_loss = self.per_item_mse_loss_with_target_norm(sae_out, y_error)
        mse_loss = per_item_mse_loss.mean()
        # L1 loss
        via_gate_feature_magnitudes = F.relu(hidden_pre_gate)
        sparsity = via_gate_feature_magnitudes.norm(p=1, dim=1).mean(dim=(0,))
        l1_loss = self.l1_coefficient * sparsity
        # Auxiliary loss
        via_gate_reconstruction = einops.einsum(via_gate_feature_magnitudes, self.W_dec.detach(), "... d_sae, d_sae d_in -> ... d_in") + self.b_dec.detach()
        aux_loss = F.mse_loss(via_gate_reconstruction, y_error, reduction="mean")
        
        loss = mse_loss + l1_loss + aux_loss

        return sae_out, loss

    def per_item_mse_loss_with_target_norm(self, preds, target):
        return torch.nn.functional.mse_loss(preds, target, reduction='none')

# %%
n_input_features = 768
projection_up = 4
gated_sae = GatedSAE(n_input_features=768, n_learned_features=n_input_features*projection_up)

# %%
# Test the forward pass
x = original_z[:16, :]
y = sae_errors[:16, :]
sae_out, loss = gated_sae(x, y)
sae_out.shape, loss

# %%
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

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# %%
torch.set_grad_enabled(True)

# %%
# Training loop
import torch.optim as optim

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

# %%



