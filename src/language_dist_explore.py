# %%
%load_ext autoreload
%autoreload 2

# %%

import circuitsvis as cv
import torch
import einops

from circuit_discovery import CircuitDiscovery
from lang_dist_graph_extract import LanguageDistributionGraphExtract
from discovery_strategies import create_filter, create_simple_greedy_strategy
from pprint import pprint
from time import time
from open_web_text import open_web_text_tokens

# %%

torch.set_grad_enabled(False)


# %%
strat = create_simple_greedy_strategy(passes=1)
fil = create_filter(no_sae_error=True)

ld = LanguageDistributionGraphExtract(strategy=strat, component_filter=fil, num_seqs=100, head_layer_threshold=2)

# %%
start = time()
vecs, seq_pos = ld.run()

print(f"Time: {time() - start}")

# %%
cd.model.tokenizer.decode(open_web_text_tokens[9])




# %%
cd = ld.get_circuit_discovery_for_seq(9, -7)
cd.print_attn_heads_and_mlps_in_graph()

# %%
seq_pos[-10:]

# %%
t = 104
print(cd.model.tokenizer.decode(open_web_text_tokens[9, t]))

cd.model.tokenizer.decode(open_web_text_tokens[9, t-2:t+2])



# %%

cd.visualize_graph()

# %%
cd.component_lens_at_loc([4, 0])


# %%

cd = CircuitDiscovery(
    # prompt="''The quick brown fox jumps over the lazy dog.",
    prompt=' volunteers was in stark contrast',
    seq_index=-1,
    token=" to",
    allowed_components_filter=fil,
)

strat(cd)



# %%
a = torch.zeros(12, 12)

# %%
a[[1, 2], 1] = 1

# %%
a

# %%

v = cd.get_graph_feature_vec()

# %%
v.shape


# %%

num_head_sources = cd.model.cfg.n_heads * (cd.model.cfg.n_layers - 1)
num_mlp_sources = cd.model.cfg.n_layers - 1

num_head_targets = 3 * cd.model.cfg.n_heads * (cd.model.cfg.n_layers - 1)
num_mlp_targets = cd.model.cfg.n_layers - 1


def head_source_index(layer: int, head: int):
    return (layer * cd.model.cfg.n_heads) + head

def mlp_source_index(layer: int):
    return num_head_sources + layer

def head_target_index(layer: int, head: int, head_type: str):
    if head_type == "q":
        type_i = 0
    elif head_type == "k":
        type_i = 1
    elif head_type == "v":
        type_i = 2

    # Head 0 isn't a target
    return ((layer - 1) * cd.model.cfg.n_heads * 3) + (head * 3) + type_i

def mlp_target_index(layer: int):
    # MLP 0 isn't a target
    return num_head_targets + layer - 1


# %%
ff = einops.rearrange(v, "(n h) -> n h", n=num_head_sources + num_mlp_sources)

# %%
num_head_sources + num_mlp_sources

# %%
58201 / 143

# %%
ff[head_source_index(9, 11), head_target_index(10, 1, 'v')]

# %%
ff.sum()

# %%

# %%
pprint(cd.get_heads_and_mlps_in_graph())

# %%
cd.are_heads_above_layer(1)

# %%
