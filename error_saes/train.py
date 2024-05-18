import torch
import sys
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union
from tqdm import trange, tqdm
import os

sys.path.append('../src')

from gated_sae import GatedSAE
from vanilla_sae import SparseAutoencoder

class GatedSAEDataset(Dataset):
    def __init__(self, original_z: Tensor, sae_errors: Tensor):
        self.original_z = original_z
        self.sae_errors = sae_errors

    def __len__(self):
        return len(self.original_z)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.original_z[idx], self.sae_errors[idx]

def create_dataloaders(dataset: Dataset, batch_size: int = 64, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Save the datasets
    torch.save(train_dataset, 'data/train_dataset.pt')
    torch.save(test_dataset, 'data/test_dataset.pt')
    
    return train_dataloader, test_dataloader

def train_model(model: Union[GatedSAE, SparseAutoencoder], train_dataloader: DataLoader, test_dataloader: DataLoader, n_epochs: int = 1, lr: float = 0.001) -> float:
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Print initial test loss
    initial_test_loss, initial_recon_loss, initil_l0_loss = evaluate_model(model, test_dataloader)
    print(f"Initial Test Loss {initial_test_loss:.4f} | Initial Reconstruction Error {initial_recon_loss:.4f} | Initial L0 Loss {initil_l0_loss:.4f}")

    for epoch in trange(n_epochs):
        print(f"Epoch {epoch + 1}")
        model.train()
        for x, y in tqdm(train_dataloader, desc='Training...'):
            optimizer.zero_grad()
            _, loss, _ = model(x, y)
            loss.backward()
            optimizer.step()
        
        test_loss, recon_loss, l0_loss = evaluate_model(model, test_dataloader)
        print(f"Test Loss {test_loss:.4f} | Reconstruction Error {recon_loss:.4f} | L0 Loss {l0_loss:.4f}")

    return recon_loss, l0_loss

def evaluate_model(model: Union[GatedSAE, SparseAutoencoder], dataloader: DataLoader) -> Tuple[float, float]:
    model.eval()
    total_loss, total_recon_loss, l0_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Evaluating...'):
            _, loss, recon_loss = model(x, y)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            # Calculate L0 loss on learned activations
            learned_activations = model.encoder(x)
            # L0 loss is the number of non-zero activations in the second dimension, averaged over the batch (first dimension)
            l0_loss += (learned_activations != 0).float().sum(dim=1).mean().item()
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_l0_loss = l0_loss / len(dataloader)
    return avg_loss, avg_recon_loss, avg_l0_loss

def main(model_type: str, n_epochs: int = 1, l1_coefficient: float = 1e-4, batch_size: int = 64, lr: float = 0.001) -> float:
    sae_errors = torch.load('data/sae_errors.pt')
    original_z = torch.load('data/original_z.pt')
    
    dataset = GatedSAEDataset(original_z, sae_errors)
    # If dataloader paths don't exist
    if not os.path.exists('data/train_dataset.pt') and not os.path.exists('data/test_dataset.pt'):
        print("Creating new dataloaders...")
        train_dataloader, test_dataloader = create_dataloaders(dataset, batch_size)
    else:
        print("Loading existing dataloaders...")
        train_dataset = torch.load('data/train_dataset.pt')
        test_dataset = torch.load('data/test_dataset.pt')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    n_input_features = 768  
    projection_up = 8

    if model_type == 'gated':
        model = GatedSAE(n_input_features=n_input_features, n_learned_features=n_input_features * projection_up, l1_coefficient=l1_coefficient)
    else:
        model = SparseAutoencoder(n_input_features=n_input_features, n_learned_features=n_input_features * projection_up, l1_coefficient=l1_coefficient)
    
    final_recon_loss, l0_loss = train_model(model, train_dataloader, test_dataloader, n_epochs, lr)

    model_path = 'data/gated_sae.pt' if model_type == 'gated' else 'data/sparse_sae.pt'
    torch.save(model, model_path)

    return final_recon_loss, l0_loss

if __name__ == '__main__':
    final_recon_loss, final_l0_loss = main(model_type='gated', n_epochs=5, l1_coefficient=1e-4)
    print(f"Final Reconstruction Error: {final_recon_loss:.4f}")
    print(f"Final L0 Loss: {final_l0_loss:.4f}")


