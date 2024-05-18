import os
import sys
import torch
import einops
import argparse
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union
from tqdm import trange, tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import HfApi, HfFolder, Repository, login
from transformer_lens import HookedTransformer
from gated_sae import GatedSAE
from vanilla_sae import SparseAutoencoder

sys.path.append('../src')
from circuit_lens import get_model_encoders

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TokenizedDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        return self.tokenized_dataset[idx]

class SAEDataset(Dataset):
    def __init__(self, z_acts):
        self.z_acts = z_acts

    def __len__(self):
        return len(self.z_acts)

    def __getitem__(self, idx):
        return self.z_acts[idx]

class GatedSAEDataset(Dataset):
    def __init__(self, original_z: Tensor, sae_errors: Tensor):
        self.original_z = original_z
        self.sae_errors = sae_errors

    def __len__(self):
        return len(self.original_z)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.original_z[idx], self.sae_errors[idx]

def create_tokenized_dataset():
    print("Creating tokenized dataset...")
    model_name = 'gpt2-small'
    dataset = load_dataset("NeelNanda/pile-10k")
    print("Dataset loaded.")
    
    seq_len = 128
    model = HookedTransformer.from_pretrained(model_name, device='cpu')
    
    tokenized_dataset = []
    text = " ".join(dataset['train']['text'])
    
    for i in trange(0, len(text)//100, 2500):
        tokens = model.to_tokens(text[i:i+2500]).squeeze()
        for j in range(0, len(tokens), seq_len):
            tokenized_dataset.append(tokens[j:j+seq_len])
    
    tokenized_dataset = [x for x in tokenized_dataset if len(x) == seq_len]
    print(f"Length of tokenised dataset = {len(tokenized_dataset)}")
    
    for i, tokens in enumerate(tokenized_dataset):
        assert tokens.shape[0] == seq_len, f"Token {i} has shape {tokens.shape}"
    
    tokenized_dataset = torch.stack(tokenized_dataset)
    torch.save(tokenized_dataset, 'data/tokenized_dataset.pt')
    print(f"Tokenized dataset with {len(tokenized_dataset)} examples with {seq_len} tokens saved to 'data/tokenized_dataset.pt'.")

def get_z_activations(model, dataloader, layer):
    z_acts = []
    for batch in tqdm(dataloader, desc='Getting z activations'):
        logits, cache = model.run_with_cache(batch)
        z = cache["z", layer]  # batch_size x seq_len x n_heads x d_head
        del logits
        del cache
        z = einops.rearrange(z, "b s n d -> (b s) (n d)")
        z_acts.append(z)
    return torch.cat(z_acts, dim=0)

def get_sae_errors(sae, z_acts, batch_size):
    sae_dataset = SAEDataset(z_acts)
    sae_dataloader = DataLoader(sae_dataset, batch_size=batch_size, shuffle=True)
    
    sae_errors = []
    original_z = []
    for z_batch in tqdm(sae_dataloader, desc='Getting SAE errors'):
        _, z_recon, _, _, _ = sae(z_batch)
        sae_error = z_batch - z_recon
        assert torch.allclose(z_recon + sae_error, z_batch, rtol=1e-4, atol=1e-4)
        sae_errors.append(sae_error)
        original_z.append(z_batch)
    
    return torch.cat(sae_errors, dim=0), torch.cat(original_z, dim=0)

def create_dataloaders(dataset: Dataset, batch_size: int, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def train_model(model: Union[GatedSAE, SparseAutoencoder], train_dataloader: DataLoader, test_dataloader: DataLoader, n_epochs: int, lr: float) -> float:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    initial_test_loss, initial_recon_loss, initial_l0_loss = evaluate_model(model, test_dataloader)
    print(f"Initial Test Loss {initial_test_loss:.4f} | Initial Reconstruction Error {initial_recon_loss:.4f} | Initial L0 Loss {initial_l0_loss:.4f}")

    for epoch in trange(n_epochs, desc='Training Epochs'):
        model.train()
        for x, y in tqdm(train_dataloader, desc='Training'):
            optimizer.zero_grad()
            _, loss, _ = model(x, y)
            loss.backward()
            optimizer.step()
        
        test_loss, recon_loss, l0_loss = evaluate_model(model, test_dataloader)
        print(f"Epoch {epoch + 1} Test Loss {test_loss:.4f} | Reconstruction Error {recon_loss:.4f} | L0 Loss {l0_loss:.4f}")

    return recon_loss, l0_loss

def evaluate_model(model: Union[GatedSAE, SparseAutoencoder], dataloader: DataLoader) -> Tuple[float, float]:
    model.eval()
    total_loss, total_recon_loss, l0_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Evaluating'):
            _, loss, recon_loss = model(x, y)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            learned_activations = model.encoder(x)
            l0_loss += (learned_activations != 0).float().sum(dim=1).mean().item()
    
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_l0_loss = l0_loss / len(dataloader)
    return avg_loss, avg_recon_loss, avg_l0_loss

def main(layer: int, model_type: str, n_epochs: int, l1_coefficient: float, batch_size: int, lr: float, repo_name: str):
    print(f"Running on {device}...")

    if not os.path.exists('data/tokenized_dataset.pt'):
        create_tokenized_dataset()
    
    model, z_saes, _ = get_model_encoders(device=device)
    sae = z_saes[layer]

    tokenized_dataset = torch.load('data/tokenized_dataset.pt')
    print(f"Length of tokenised dataset = {len(tokenized_dataset)}")
    dataset = TokenizedDataset(tokenized_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    z_acts_path = f'data/z_acts_layer_{layer}.pt'
    sae_errors_path = f'data/sae_errors_layer_{layer}.pt'
    original_z_path = f'data/original_z_layer_{layer}.pt'

    if not os.path.exists(z_acts_path):
        print(f"Z activations for layer {layer} not found, generating...")
        torch.set_grad_enabled(False)
        z_acts = get_z_activations(model, dataloader, layer)
        torch.save(z_acts, z_acts_path)
        torch.set_grad_enabled(True)
    else:
        print(f"Loading existing Z activations for layer {layer}...")
        z_acts = torch.load(z_acts_path)

    if not os.path.exists(sae_errors_path) or not os.path.exists(original_z_path):
        print(f"SAE errors and/or original Z activations not found for layer {layer}, generating...")
        sae_errors, original_z = get_sae_errors(sae, z_acts, batch_size)
        torch.save(sae_errors, sae_errors_path)
        torch.save(original_z, original_z_path)
    else:
        print(f"Loading existing SAE errors and original Z activations for layer {layer}...")
        sae_errors = torch.load(sae_errors_path)
        original_z = torch.load(original_z_path)

    dataset = GatedSAEDataset(original_z, sae_errors)
    train_dataloader, test_dataloader = create_dataloaders(dataset, batch_size)

    n_input_features = 768
    projection_up = 8

    if model_type == 'gated':
        model = GatedSAE(n_input_features=n_input_features, n_learned_features=n_input_features * projection_up, l1_coefficient=l1_coefficient)
    else:
        model = SparseAutoencoder(n_input_features=n_input_features, n_learned_features=n_input_features * projection_up, l1_coefficient=l1_coefficient)

    final_recon_loss, final_l0_loss = train_model(model, train_dataloader, test_dataloader, n_epochs, lr)

    local_model_path = f'./{repo_name}/sae_layer_{layer}.pt'
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
    torch.save(model, local_model_path)
    print(f"Final Reconstruction Error: {final_recon_loss:.4f}")
    print(f"Final L0 Loss: {final_l0_loss:.4f}")

    # Upload the model to HuggingFace Hub
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_model_path,
        path_in_repo=f'sae_layer_{layer}.pt',
        repo_id=f'charlieoneill/{repo_name}'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and Save Gated SAE for Transformer Layers")
    parser.add_argument('--layer', type=int, required=True, help='Layer number to train the SAE on')
    parser.add_argument('--model_type', type=str, default='gated', choices=['gated', 'vanilla'], help='Type of SAE model to train')
    parser.add_argument('--n_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--l1_coefficient', type=float, default=1e-4, help='L1 regularisation coefficient')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--repo_name', type=str, default="error-saes", help='HuggingFace repository name to save the model')

    args = parser.parse_args()

    main(args.layer, args.model_type, args.n_epochs, args.l1_coefficient, args.batch_size, args.lr, args.repo_name)
