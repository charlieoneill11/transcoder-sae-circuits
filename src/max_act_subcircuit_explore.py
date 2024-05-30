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

ma = MaxActAnalysis('attn', 8, 16513, num_sequences=5_000, strategy=strat, component_filter=fil)

# %%
ae = ma.active_examples

# %%
ma.get_circuit_discovery_for_max_activating_example

ats  = []

for i in trange(len(ma.active_examples[0])):
    cd = ma.get_circuit_discovery_for_max_activating_example(i)
    ats.append(cd.get_graph_edge_matrix(only_include_head_edges=True))

# %%
len(ma.active_examples[0])

# %%
ma.active_examples[0]


# %%
mm = EdgeMatrixMethods(cd.model)
anal = EdgeMatrixAnalyzer(mm, torch.stack(ats), ma.active_examples[0])

# %%
anal.imshow_totals(no_clip=False, zmax=100)

# %%
ats = torch.stack(ats)

# %%
ind = ats[:, mm.head_source_index(5, 5), mm.head_target_index(8, 6, 'v')].nonzero().squeeze()

# %%
ind
seq_pos = ma.active_examples[0][ind]

# %%
cd = ma.get_circuit_discovery_for_max_activating_example(ind[2])
cd.print_attn_heads_and_mlps_in_graph()

# %%
cd.visualize_graph()


# %%
