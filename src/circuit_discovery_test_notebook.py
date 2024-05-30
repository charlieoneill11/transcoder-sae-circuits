# %%
%load_ext autoreload
%autoreload 2

# %%
from example_prompts import SUCCESSOR_EXAMPLE_PROMPT, IOI_EXAMPLE_PROMPT

from circuit_discovery import CircuitDiscovery, only_feature
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
import torch.nn.functional as F
from plotly_utils import *
from circuit_lens import CircuitComponent

from rich import print as rprint
from rich.table import Table

import torch
import time
import einops

# %%
torch.set_grad_enabled(False)

# %%
IOI_EXAMPLE_PROMPT = "Mary and Jeff went to the store, and Mary gave an apple to Jeff"
IOI_COUNTER = "Mary and Jeff went to the store, and Mary gave an apple to Mary"

# %%
cd = CircuitDiscovery(IOI_EXAMPLE_PROMPT, -2, allowed_components_filter=only_feature)
counter = CircuitDiscovery(IOI_COUNTER, -2, allowed_components_filter=only_feature)

# %%
cd.component_lens_at_loc([0, 0, 'q'])



# %%
pass_based = True

passes = 5
node_contributors = 1
first_pass_minimal = True

sub_passes = 3
do_sub_pass = False
layer_thres = 9
minimal = True


num_greedy_passes = 20
k = 1
N = 30

thres = 4

def strategy(cd: CircuitDiscovery):
    if pass_based:
        for _ in range(passes):
            cd.add_greedy_pass(contributors_per_node=node_contributors, minimal=first_pass_minimal)

            if do_sub_pass:
                for _ in range(sub_passes):
                    cd.add_greedy_pass_against_all_existing_nodes(contributors_per_node=node_contributors, skip_z_features=True, layer_threshold=layer_thres, minimal=minimal)
    else:
        for _ in range(num_greedy_passes):
            cd.greedily_add_top_contributors(k=k, reciever_threshold=thres)



cd.reset_graph()
counter.reset_graph()

strategy(cd)
strategy(counter)

cd.print_attn_heads_and_mlps_in_graph()
counter.print_attn_heads_and_mlps_in_graph()


# %%
## Shows a graphviz-based visual for the different nodes and edges in 
## the circuit, together with the heads and token positions on which they operate

cd.visualize_graph()

# %%
## Show the attention heads patterns for all the heads that are included in
## the circuit

cd.visualize_attn_heads_in_graph()

# %%
## Show the component lens attributions recursively from the root
cd.component_lens_at_loc([0, 1])
