import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import (
    utils,
)
from dtypes import DTYPES


RUN_DICT = {
    0: "gpt2-small_L0_Hcat_z_lr1.20e-03_l11.80e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    1: "gpt2-small_L1_Hcat_z_lr1.20e-03_l18.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v5",
    2: "gpt2-small_L2_Hcat_z_lr1.20e-03_l11.00e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v4",
    3: "gpt2-small_L3_Hcat_z_lr1.20e-03_l19.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    4: "gpt2-small_L4_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v7",
    5: "gpt2-small_L5_Hcat_z_lr1.20e-03_l11.00e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    6: "gpt2-small_L6_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    7: "gpt2-small_L7_Hcat_z_lr1.20e-03_l11.10e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    8: "gpt2-small_L8_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v6",
    9: "gpt2-small_L9_Hcat_z_lr1.20e-03_l11.20e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    10: "gpt2-small_L0_Hcat_z_lr1.20e-03_l11.80e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    11: "gpt2-small_L11_Hcat_z_lr1.20e-03_l13.00e+00_ds24576_bs4096_dc3.16e-06_rsanthropic_rie25000_nr4_v9",
}


class ZSAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg["act_size"], d_hidden, dtype=dtype)
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(d_hidden, cfg["act_size"], dtype=dtype)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        self.dtype = dtype
        self.device = cfg["device"]

        self.version = 0
        self.to(cfg["device"])

    def forward(self, x, per_token=False):
        # Move x to cfg device
        x = x.to(self.device)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)  # [batch_size, d_hidden]
        x_reconstruct = acts @ self.W_dec + self.b_dec  # [batch_size, act_size]
        if per_token:
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1)  # [batch_size]
            l1_loss = self.l1_coeff * (acts.float().abs().sum(dim=-1))  # [batch_size]
            loss = l2_loss + l1_loss  # [batch_size]
        else:
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)  # []
            l1_loss = self.l1_coeff * (acts.float().abs().sum(dim=-1).mean(dim=0))  # []
            loss = l2_loss + l1_loss  # []
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    def encode(self, x):
        x = x.to(self.device)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        return acts
    
    def decode(self, acts):
        x_reconstruct = acts @ self.W_dec + self.b_dec
        return x_reconstruct

    @classmethod
    def load_from_hf(cls, version, hf_repo="ckkissane/tinystories-1M-SAES"):
        """
        Loads the saved autoencoder from HuggingFace.
        """

        cfg = utils.download_file_from_hf(hf_repo, f"{version}_cfg.json")
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf(hf_repo, f"{version}.pt", force_is_torch=True))  # type: ignore
        return self

    @classmethod
    def load_zsae_for_layer(cls, layer: int):
        auto_encoder_run = RUN_DICT[layer]
        return cls.load_from_hf(
            auto_encoder_run, hf_repo="ckkissane/attn-saes-gpt2-small-all-layers"
        )
