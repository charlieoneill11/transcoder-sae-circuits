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
from gated_sae import GatedSAE
from vanilla_sae import SparseAutoencoder
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

sys.path.append('../src')
from circuit_lens import get_model_encoders

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to track neuron activity
def track_neuron_activity(hidden_post, neuron_activity, step):
    with torch.no_grad():
        fired_neurons = (hidden_post != 0).float().sum(dim=0)
        neuron_activity[step % 25000] = fired_neurons

# Function to resample neurons
def resample_neurons(model, neuron_activity, optimizer):
    with torch.no_grad():
        total_steps = neuron_activity.shape[0]
        fired_counts = neuron_activity.sum(dim=0)
        firing_rates = fired_counts / total_steps
        dead_neurons = firing_rates < 1e-7

        if dead_neurons.sum() == 0:
            return

        # Kaiming Uniform initialization for dead neurons
        kaiming_init = torch.nn.init.kaiming_uniform_

        # Reinitialize encoder weights for dead neurons
        model.W_enc[:, dead_neurons] = kaiming_init(model.W_enc[:, dead_neurons])

        # Reinitialize decoder weights for dead neurons
        model.W_dec.weight[dead_neurons, :] = kaiming_init(model.W_dec.weight[dead_neurons, :])

        # Reset biases for dead neurons
        model.b_mag[dead_neurons] = 0
        model.b_gate[dead_neurons] = 0
        model.W_dec.bias[dead_neurons] = 0

        # Reset optimizer parameters for dead neurons
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = optimizer.state[p]
                    if 'exp_avg' in state:
                        state['exp_avg'][dead_neurons] = 0
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'][dead_neurons] = 0

def get_z_activations(model, batch, layer):
    with torch.no_grad():
        logits, cache = model.run_with_cache(batch)
        z = cache["z", layer]
        del logits
        del cache
        z = einops.rearrange(z, "b s n d -> (b s) (n d)")
    return z

def train_regular_sae(model: Union[GatedSAE, SparseAutoencoder], tl_model, sae, n_batches: int, lr: float, repo_name: str, layer: int, l1_coefficient: float, batch_size: int, projection_up: int, activation_store) -> float:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    sae = sae.to(device)

    # Initialize neuron activity tracker
    neuron_activity = torch.zeros((25000, model.n_learned_features), device=device)
    total_steps = 0

    initial_test_loss, initial_recon_loss, initial_l0_loss, initial_dead_neurons = evaluate_model(model, tl_model, activation_store, layer, sae)
    print(f"Initial Test Loss {initial_test_loss:.4f} | Initial Reconstruction Error {initial_recon_loss:.4f} | Initial L0 Loss {initial_l0_loss:.4f} | Initial Dead Neurons: {initial_dead_neurons:.2f}%")
    
    # Log initial metrics
    wandb.log({
        "test_loss": initial_test_loss,
        "recon_loss": initial_recon_loss,
        "l0_loss": initial_l0_loss,
        "dead_neurons": initial_dead_neurons,
        "training_type": "regular_sae"
    })

    for batch_num in trange(n_batches, desc='Training Batches'):
        model.train()
        batch_tokens = activation_store.get_batch_tokens().to(device)
        z_acts = get_z_activations(tl_model, batch_tokens, layer)
        optimizer.zero_grad()
        sae_out, loss, _ = model(z_acts, z_acts)
        loss.backward()
        optimizer.step()

        # Track neuron activity
        track_neuron_activity(model.encoder(z_acts), neuron_activity, total_steps)
        total_steps += batch_size

        # Clear cache
        del z_acts
        del batch_tokens
        del sae_out
        torch.cuda.empty_cache()
        
        if (batch_num + 1) % 25 == 0:
            test_loss, recon_loss, l0_loss, dead_neurons = evaluate_model(model, tl_model, activation_store, layer, sae)
            print(f"Batch {batch_num + 1} Test Loss {test_loss:.4f} | Reconstruction Error {recon_loss:.4f} | L0 Loss {l0_loss:.4f} | Dead Neurons: {dead_neurons:.2f}%")

            # Log metrics
            wandb.log({
                "test_loss": test_loss,
                "recon_loss": recon_loss,
                "l0_loss": l0_loss,
                "dead_neurons": dead_neurons
            })

        # Resample neurons at specific training steps
        if total_steps in {50000, 100000, 150000, 200000}:
            resample_neurons(model, neuron_activity, optimizer)

        # Save and upload the model every 1000 batches
        if (batch_num + 1) % 5000 == 0 or batch_num == n_batches - 1:
            local_model_path = f'./{repo_name}/sae_layer_{layer}.pt'
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            torch.save(model.state_dict(), local_model_path)

            # Create config dictionary
            test_loss, recon_loss, l0_loss, dead_neurons = evaluate_model(model, tl_model, activation_store, layer, sae)
            config = {
                "layer": layer,
                "model_type": type(model).__name__,
                "n_batches": batch_num+1,
                "l1_coefficient": l1_coefficient,
                "projection_up": projection_up,
                "batch_size": batch_size,
                "learning_rate": lr,
                "test_loss": test_loss,
                "reconstruction_error": recon_loss,
                "l0_loss": l0_loss,
                "dead_neurons_percentage": dead_neurons
            }

            # Save config to JSON file
            config_path = f'./{repo_name}/config_layer_{layer}_{projection_up}.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            # Upload model and config to Hugging Face
            api = HfApi(token="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")
            api.upload_file(
                path_or_fileobj=local_model_path,
                path_in_repo=f'sae_layer_{layer}_{projection_up}.pt',
                repo_id=f'charlieoneill/{repo_name}',
                token="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"
            )
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo=f'config_layer_{layer}_{projection_up}.json',
                repo_id=f'charlieoneill/{repo_name}',
                token="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"
            )
            print(f"Model and config saved and uploaded for batch {batch_num + 1}")

    return recon_loss, l0_loss

def evaluate_model(model: Union[GatedSAE, SparseAutoencoder], tl_model, activation_store, layer, sae) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss, total_recon_loss, l0_loss = 0.0, 0.0, 0.0
    total_neurons = None
    total_batches = 0

    with torch.no_grad():
        for _ in range(10):  # evaluate on 10 batches
            batch_tokens = activation_store.get_batch_tokens().to(device)
            z_acts = get_z_activations(tl_model, batch_tokens, layer)
            _, loss, recon_loss = model(z_acts, z_acts)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            learned_activations = model.encoder(z_acts)
            l0_loss += (learned_activations != 0).float().sum(dim=1).mean().item()

            # Initialize total_neurons with zeros on the first batch
            if total_neurons is None:
                total_neurons = torch.zeros_like(learned_activations[0], dtype=torch.float, device=learned_activations.device)
            
            # Sum the activations for each neuron across all batches
            total_neurons += (learned_activations != 0).float().sum(dim=0)
            total_batches += learned_activations.shape[0]

    avg_loss = total_loss / 10  # average over 10 batches
    avg_recon_loss = total_recon_loss / 10
    avg_l0_loss = l0_loss / 10

    # Calculate the proportion of neurons that are zero across all test examples
    dead_neurons = (total_neurons == 0).sum().item()
    total_neurons_count = total_neurons.numel()
    percentage_dead_neurons = (dead_neurons / total_neurons_count) * 100

    return avg_loss, avg_recon_loss, avg_l0_loss, percentage_dead_neurons

def main_regular_sae(layer: int, model_type: str, n_batches: int, l1_coefficient: float, batch_size: int, lr: float, projection_up: int, repo_name: str):
    torch.set_grad_enabled(False)
    print(f"Running on {device}...")

    # Set the W&B run name
    run_name = f"{layer}_{l1_coefficient}_{projection_up}_regular"

    # Initialize W&B
    wandb.init(project=repo_name, name=run_name, config={
        "layer": layer,
        "model_type": model_type,
        "n_batches": n_batches,
        "l1_coefficient": l1_coefficient,
        "batch_size": batch_size,
        "learning_rate": lr,
        "projection_up": projection_up,
        "training_type": "regular_sae"
    })

    # Load SAE model as before
    _, z_saes, _ = get_model_encoders(device=device)
    sae = z_saes[layer]

    # Load the transformer model and activation store
    hook_point = "blocks.8.hook_resid_pre" # this doesn't matter
    saes, _ = get_gpt2_res_jb_saes(hook_point)
    sparse_autoencoder = saes[hook_point]
    sparse_autoencoder.to(device)
    sparse_autoencoder.cfg.device = device
    sparse_autoencoder.cfg.hook_point = f"blocks.{layer}.attn.hook_z"
    sparse_autoencoder.cfg.store_batch_size = batch_size

    loader = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg)

    print(f"Loader cfg batch size = {sparse_autoencoder.cfg.store_batch_size} (batch size = {batch_size})")

    # don't overwrite the sparse autoencoder with the loader's sae (newly initialized)
    tl_model, _, activation_store = loader.load_sae_training_group_session()

    n_input_features = 768

    torch.set_grad_enabled(True)
    if model_type == 'gated':
        model = GatedSAE(n_input_features=n_input_features, n_learned_features=n_input_features * projection_up, l1_coefficient=l1_coefficient)
    else:
        model = SparseAutoencoder(n_input_features=n_input_features, n_learned_features=n_input_features * projection_up, l1_coefficient=l1_coefficient)
    
    model = model.to(device)

    final_recon_loss, final_l0_loss = train_regular_sae(model=model, tl_model=tl_model, sae=sae, 
                                                        n_batches=n_batches, lr=lr, repo_name=repo_name, 
                                                        layer=layer, l1_coefficient=l1_coefficient, batch_size=batch_size, 
                                                        projection_up=projection_up, activation_store=activation_store)

    print(f"Final Reconstruction Error: {final_recon_loss:.4f}")
    print(f"Final L0 Loss: {final_l0_loss:.4f}")

    # Finish W&B run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and Save Regular SAE for Transformer Layers")
    parser.add_argument('--layer', type=int, required=True, help='Layer number to train the SAE on')
    parser.add_argument('--model_type', type=str, default='gated', choices=['gated', 'vanilla'], help='Type of SAE model to train')
    parser.add_argument('--n_batches', type=int, default=2500, help='Number of training batches')
    parser.add_argument('--l1_coefficient', type=float, default=2, help='L1 regularisation coefficient')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--projection_up', type=int, default=16, help='Factor to increase the number of learned features by')
    parser.add_argument('--repo_name', type=str, default="regular-sae", help='HuggingFace repository name to save the model')

    args = parser.parse_args()

    # Print total number of tokens we will train for = batch_size * n_batches * 128
    print(f"Total number of tokens we will train for: {args.batch_size * args.n_batches * 128}")

    main_regular_sae(args.layer, args.model_type, args.n_batches, args.l1_coefficient, args.batch_size, args.lr, args.projection_up, args.repo_name)
