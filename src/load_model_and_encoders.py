import torch
import einops

from z_sae import ZSAE
from mlp_transcoder import SparseTranscoder
from transformer_lens import HookedTransformer
from typing import Any, Tuple, Optional
from tqdm import trange

model_encoder_cache: Optional[Tuple[HookedTransformer, Any, Any]] = None

model_cache: Optional[HookedTransformer] = None


def get_model(device):
    global model_cache

    if model_cache is not None:
        return model_cache

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model_cache = model

    return model


def get_model_encoders(device):
    global model_encoder_cache

    if model_encoder_cache is not None:
        return model_encoder_cache

    model = get_model(device)

    print()
    print("Loading SAEs...")
    z_saes = [ZSAE.load_zsae_for_layer(i) for i in trange(model.cfg.n_layers)]

    print()
    print("Loading Transcoders...")
    mlp_transcoders = [
        SparseTranscoder.load_from_hugging_face(i) for i in trange(model.cfg.n_layers)
    ]

    model_encoder_cache = (model, z_saes, mlp_transcoders)

    return model_encoder_cache
