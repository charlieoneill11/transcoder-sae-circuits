# %%
%load_ext autoreload
%autoreload 2

# %%
import torch

from max_act_analysis import MaxActAnalysis
from circuit_discovery import EdgeMatrixAnalyzer, EdgeMatrixMethods
from discovery_strategies import create_filter, create_simple_greedy_strategy
from tqdm import trange

# %%
torch.set_grad_enabled(False)


# %%
strat = create_simple_greedy_strategy(passes=1)
fil = create_filter(no_sae_error=True)

ma = MaxActAnalysis('attn', 9, 15647, num_sequences=5_000, strategy=strat, component_filter=fil)

# %%
ae = ma.active_examples

# %%
ma.get_circuit_discovery_for_max_activating_example

mats  = []

for i in trange(len(ma.active_examples[0])):
    cd = ma.get_circuit_discovery_for_max_activating_example(i)
    mats.append(cd.get_graph_edge_matrix())

# %%

len(ma.active_examples[0])

# %%
