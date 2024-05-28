# %%
%load_ext autoreload
%autoreload 2

# %%
import torch

from circuit_interp import CircuitInterp
from circuit_discovery import CircuitDiscovery
from data.ioi_dataset import gen_templated_prompts
from discovery_strategies import create_simple_greedy_strategy, create_filter

# %%
torch.set_grad_enabled(False)

# %%
dataset_prompts = gen_templated_prompts(template_idex=1, N=500)

# %%
strat = create_simple_greedy_strategy(passes=1)
no_sae_error_filter = create_filter(no_sae_error=True)

# %%
pi = 1

cd = CircuitDiscovery(
    dataset_prompts[pi]['text'],
    token=dataset_prompts[pi]['correct'],
    allowed_components_filter=no_sae_error_filter,
)
# %%
strat(cd)

# %%
cd.visualize_graph()

# %%
cd.root_node.sorted_contributors_in_graph[0].sorted_contributors_in_graph[0].sorted_contributors_in_graph('q')

# %%
