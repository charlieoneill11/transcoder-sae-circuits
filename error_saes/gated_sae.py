import torch
import einops
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import List, Dict, TypedDict, Any, Union, Tuple, Optional

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

        return sae_out, loss, mse_loss
    
    def encoder(self, x_act):
        hidden_pre = einops.einsum(x_act, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")

        # Gated SAE
        hidden_pre_mag = hidden_pre * torch.exp(self.r_mag) + self.b_mag
        hidden_post_mag = self.activation_fn(hidden_pre_mag)  
        hidden_pre_gate = hidden_pre + self.b_gate
        hidden_post_gate = (torch.sign(hidden_pre_gate) + 1) / 2
        hidden_post = hidden_post_mag * hidden_post_gate

        return hidden_post

    def per_item_mse_loss_with_target_norm(self, preds, target):
        return torch.nn.functional.mse_loss(preds, target, reduction='none')