# import torch
# import einops
# from torch import Tensor
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
# from typing import List, Dict, TypedDict, Any, Union, Tuple, Optional

# class GatedSAE(nn.Module):

#     def __init__(self, n_input_features, n_learned_features, l1_coefficient=0.01):

#         super().__init__()

#         self.n_input_features = n_input_features
#         self.n_learned_features = n_learned_features
#         self.l1_coefficient = l1_coefficient

#         self.W_enc = nn.Parameter(
#             torch.nn.init.kaiming_uniform_(torch.empty(self.n_input_features, self.n_learned_features))   
#         )
#         self.W_dec = nn.Parameter(
#             torch.nn.init.kaiming_uniform_(torch.empty(self.n_learned_features, self.n_input_features))   
#         )

#         self.r_mag = nn.Parameter(
#             torch.zeros(self.n_learned_features)
#         )
#         self.b_mag = nn.Parameter(
#             torch.zeros(self.n_learned_features)
#         )
#         self.b_gate = nn.Parameter(
#             torch.zeros(self.n_learned_features)
#         )
#         self.b_dec = nn.Parameter(
#             torch.zeros(self.n_input_features)
#         )

#         self.activation_fn = nn.ReLU()

#     def forward(self, x_act, y_error):
#         # Assert x_act (original z activations i.e. the input) and the y_error (SAE error i.e. the target) have the same shape
#         assert x_act.shape == y_error.shape, f"x_act shape {x_act.shape} does not match y_error shape {y_error.shape}"

#         hidden_pre = einops.einsum(x_act, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")

#         # Gated SAE
#         hidden_pre_mag = hidden_pre * torch.exp(self.r_mag) + self.b_mag
#         hidden_post_mag = self.activation_fn(hidden_pre_mag)  
#         hidden_pre_gate = hidden_pre + self.b_gate
#         hidden_post_gate = (torch.sign(hidden_pre_gate) + 1) / 2
#         hidden_post = hidden_post_mag * hidden_post_gate

#         sae_out = einops.einsum(hidden_post, self.W_dec, "... d_sae, d_sae d_in -> ... d_in") + self.b_dec

#         # Now we need to handle all the loss stuff
#         # Reconstruction loss
#         per_item_mse_loss = self.per_item_mse_loss_with_target_norm(sae_out, y_error)
#         mse_loss = per_item_mse_loss.mean()
#         # L1 loss
#         via_gate_feature_magnitudes = F.relu(hidden_pre_gate)
#         sparsity = via_gate_feature_magnitudes.norm(p=1, dim=1).mean(dim=(0,))
#         l1_loss = self.l1_coefficient * sparsity
#         # Auxiliary loss
#         via_gate_reconstruction = einops.einsum(via_gate_feature_magnitudes, self.W_dec.detach(), "... d_sae, d_sae d_in -> ... d_in") + self.b_dec.detach()
#         aux_loss = F.mse_loss(via_gate_reconstruction, y_error, reduction="mean")
        
#         loss = mse_loss + l1_loss + aux_loss

#         return sae_out, loss, mse_loss
    
#     def encoder(self, x_act):
#         hidden_pre = einops.einsum(x_act, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")

#         # Gated SAE
#         hidden_pre_mag = hidden_pre * torch.exp(self.r_mag) + self.b_mag
#         hidden_post_mag = self.activation_fn(hidden_pre_mag)  
#         hidden_pre_gate = hidden_pre + self.b_gate
#         hidden_post_gate = (torch.sign(hidden_pre_gate) + 1) / 2
#         hidden_post = hidden_post_mag * hidden_post_gate

#         return hidden_post

#     def per_item_mse_loss_with_target_norm(self, preds, target):
#         return torch.nn.functional.mse_loss(preds, target, reduction='none')

import torch
import einops
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
from typing import List, Dict, TypedDict, Any, Union, Tuple, Optional


class ConstrainedUnitNormLinear(nn.Module):
    DIMENSION_CONSTRAIN_UNIT_NORM: int = -1

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # Register backward hook to remove any gradient information parallel to the dictionary
        # vectors (columns of the weight matrix) before applying the gradient step.
        self.weight.register_hook(self._weight_backward_hook)

    def reset_parameters(self) -> None:
        self.weight.data = torch.nn.init.kaiming_uniform_(self.weight)

        # Scale so that each column has unit norm
        with torch.no_grad():
            normalized_weight = torch.nn.functional.normalize(self.weight, dim=self.DIMENSION_CONSTRAIN_UNIT_NORM)
            self.weight.data.copy_(normalized_weight)

        # Initialise the bias
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _weight_backward_hook(self, grad):
        dot_product = einops.einsum(
            grad, self.weight, "out_features in_features, out_features in_features -> out_features"
        )

        normalized_weight = self.weight / torch.norm(self.weight, dim=self.DIMENSION_CONSTRAIN_UNIT_NORM, keepdim=True)

        projection = einops.einsum(
            dot_product, normalized_weight, "out_features, out_features in_features -> out_features in_features"
        )

        return grad - projection

    def constrain_weights_unit_norm(self) -> None:
        with torch.no_grad():
            normalized_weight = torch.nn.functional.normalize(self.weight, dim=self.DIMENSION_CONSTRAIN_UNIT_NORM)
            self.weight.data.copy_(normalized_weight)

    def forward(self, x):
        self.constrain_weights_unit_norm()
        return torch.nn.functional.linear(x, self.weight, self.bias)


class GatedSAE(nn.Module):

    def __init__(self, n_input_features, n_learned_features, l1_coefficient=0.01):
        super().__init__()

        self.n_input_features = n_input_features
        self.n_learned_features = n_learned_features
        self.l1_coefficient = l1_coefficient

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.n_input_features, self.n_learned_features))
        )

        self.W_dec = ConstrainedUnitNormLinear(n_learned_features, n_input_features, bias=True)

        self.r_mag = nn.Parameter(torch.zeros(self.n_learned_features))
        self.b_mag = nn.Parameter(torch.zeros(self.n_learned_features))
        self.b_gate = nn.Parameter(torch.zeros(self.n_learned_features))
        self.b_dec = nn.Parameter(torch.zeros(self.n_input_features))

        self.activation_fn = nn.ReLU()

    def forward(self, x_act, y_error):
        assert x_act.shape == y_error.shape, f"x_act shape {x_act.shape} does not match y_error shape {y_error.shape}"

        hidden_pre = einops.einsum(x_act, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")

        hidden_pre_mag = hidden_pre * torch.exp(self.r_mag) + self.b_mag
        hidden_post_mag = self.activation_fn(hidden_pre_mag)
        hidden_pre_gate = hidden_pre + self.b_gate
        hidden_post_gate = (torch.sign(hidden_pre_gate) + 1) / 2
        hidden_post = hidden_post_mag * hidden_post_gate

        sae_out = self.W_dec(hidden_post) + self.b_dec

        per_item_mse_loss = self.per_item_mse_loss_with_target_norm(sae_out, y_error)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        via_gate_feature_magnitudes = F.relu(hidden_pre_gate)
        #sparsity = via_gate_feature_magnitudes.norm(p=1, dim=1).mean(dim=(0,))
        sparsity = via_gate_feature_magnitudes.norm(p=1, dim=-1).mean()
        l1_loss = self.l1_coefficient * sparsity

        via_gate_reconstruction = self.W_dec(via_gate_feature_magnitudes) + self.b_dec.detach()
        aux_loss = self.per_item_mse_loss_with_target_norm(via_gate_reconstruction, y_error).sum(dim=-1).mean() #F.mse_loss(via_gate_reconstruction, y_error, reduction="mean")

        loss = mse_loss + l1_loss + aux_loss

        return sae_out, loss, mse_loss

    def encoder(self, x_act):
        hidden_pre = einops.einsum(x_act, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")

        hidden_pre_mag = hidden_pre * torch.exp(self.r_mag) + self.b_mag
        hidden_post_mag = self.activation_fn(hidden_pre_mag)
        hidden_pre_gate = hidden_pre + self.b_gate
        hidden_post_gate = (torch.sign(hidden_pre_gate) + 1) / 2
        hidden_post = hidden_post_mag * hidden_post_gate

        return hidden_post

    def per_item_mse_loss_with_target_norm(self, preds, target):
        return torch.nn.functional.mse_loss(preds, target, reduction='none')
