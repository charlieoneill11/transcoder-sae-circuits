import os
import sys
import torch
import einops
import argparse
import wandb
from torch import Tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Union
from tqdm import trange, tqdm
from huggingface_hub import HfApi
from transformer_lens import HookedTransformer
import json
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from circuit_lens import get_model_encoders

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_z_activations(model, batch, layer):
    with torch.no_grad():
        logits, cache = model.run_with_cache(batch)
        z = cache["z", layer]
        del logits
        del cache
        z = einops.rearrange(z, "b s n d -> (b s) (n d)")
    return z

def get_zsae_acts(z_acts, sae):
    with torch.no_grad():
        zsae_acts = sae.encode(z_acts)
    return zsae_acts

def get_transcoder_acts(mlp_acts, transcoder):
    with torch.no_grad():
        transcoder_acts = transcoder.encode(mlp_acts)
    return transcoder_acts

def get_activation_dictionary(cache, z_saes, transcoders, tokens):

    activation_dict = {'tokens': tokens}

    for layer in range(12):

        # Attention Z
        z_acts = cache["z", layer]
        # print(f"Z acts shape = {z_acts.shape}")
        z_acts = einops.rearrange(z_acts, "b s n d -> b s (n d)")
        #print(f"Z acts shape = {z_acts.shape}")
        sae = z_saes[layer]
        zsae_acts = get_zsae_acts(z_acts, sae)
        #print(f"ZSAE acts shape = {zsae_acts.shape}")
        activation_dict[f"z_{layer}"] = zsae_acts

        # MLP
        mlp_input = cache["normalized", layer, "ln2"]
        #print(f"MLP input shape = {mlp_input.shape}")
        transcoder = transcoders[layer]
        transcoder_acts = get_transcoder_acts(mlp_input, transcoder)
        #print(f"Transcoder acts shape = {transcoder_acts.shape}")
        activation_dict[f"mlp_{layer}"] = transcoder_acts

    return activation_dict




def get_activation_store(batch_size: int = 64):
    # Load the transformer model and activation store
    hook_point = "blocks.8.hook_resid_pre" # this doesn't matter
    saes, _ = get_gpt2_res_jb_saes(hook_point)
    sparse_autoencoder = saes[hook_point]
    sparse_autoencoder.to(device)
    sparse_autoencoder.cfg.device = device
    sparse_autoencoder.cfg.hook_point = f"blocks.8.attn.hook_z"
    sparse_autoencoder.cfg.store_batch_size = batch_size

    loader = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg)

    print(f"Loader cfg batch size = {sparse_autoencoder.cfg.store_batch_size}")

    # don't overwrite the sparse autoencoder with the loader's sae (newly initialized)
    _, _, activation_store = loader.load_sae_training_group_session()
    
    del sparse_autoencoder
    del loader
    del saes
    del hook_point

    return activation_store

    


def main(total_tokens: int = 1_000_000, batch_size: int = 64, seq_len: int = 128):
    torch.set_grad_enabled(False)

    tl_model, z_saes, transcoders = get_model_encoders(device=device)

    # Load the transformer model and activation store
    activation_store = get_activation_store(batch_size=batch_size)

    # For a total of total_tokens, we will have total_tokens // (batch_size * seq_len) batches
    n_batches = total_tokens // (batch_size * seq_len)

    all_activations = []

    for batch in trange(n_batches):
        # Get the batch
        batch = activation_store.get_batch_tokens()

        # Get the activations
        _, cache = tl_model.run_with_cache(batch)
        activation_dict = get_activation_dictionary(cache, z_saes, transcoders, batch)
        all_activations.append(activation_dict)

    # Save the activations with torch
    torch.save(all_activations, 'maxact_activations.pt')

    print("Activations saved to activations.pt")


if __name__ == "__main__":
    total_tokens = 10_000
    batch_size = 4
    seq_len = 128
    main(total_tokens, batch_size, seq_len)