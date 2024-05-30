# %%
%load_ext autoreload
%autoreload 2

# %%

import circuitsvis as cv
import torch
import einops


from circuit_discovery import CircuitDiscovery, EdgeMatrixMethods, EdgeMatrixAnalyzer
from lang_dist_graph_extract import LanguageDistributionGraphExtract
from discovery_strategies import create_filter, create_simple_greedy_strategy
from pprint import pprint
from time import time
from open_web_text import open_web_text_tokens
from load_model_and_encoders import get_model
from transformer_lens import utils
from plotly_utils import *
from circuit_lens import CircuitComponent


# %%

torch.set_grad_enabled(False)
model = get_model('cuda')

# %%
strat = create_simple_greedy_strategy(passes=1)
fil = create_filter()

ld = LanguageDistributionGraphExtract(strategy=strat, component_filter=fil, seq_batch_size=10, num_seqs=500, head_layer_threshold=2)
mm = EdgeMatrixMethods(model)


# %%
start = time()
gm, seq_pos = ld.run()

print(f"Time: {time() - start}")

gm = torch.stack(gm)
seq_pos = torch.tensor(seq_pos)

torch.save(gm, "graph_matrices_big_better.pt")
torch.save(seq_pos, "graph_matrices_seq_pos_big_better.pt")

# %%
# torch.save(vecs, "graph_matrices.pt")
# torch.save(seq_pos, "graph_matrices_seq_pos.pt")

# %%

# %%
gm.shape

# %%
analyzer = EdgeMatrixAnalyzer(mm, gm, seq_pos)

# %%
analyzer.imshow_totals()

# %%
tots = gm.sum(dim=0)

# %%
tots.shape

# %%
indices = gm[:, mm.head_source_index(7, 11), mm.head_target_index(9, 6, 'q')].nonzero().squeeze()

# %%
source_target = seq_pos[indices]
source_target

# %%
cd = ld.get_circuit_discovery_for_seq(40, 70, is_target_index=False)
cd.print_attn_heads_and_mlps_in_graph()

# %%
cd.visualize_graph()

# %%
cd.visualize_attn_heads_in_graph()




# %%
# %%
tots[mm.head_source_index(5, 5), mm.head_target_index(8, 6, 'v')]

# %%
analyzer.get_top_head_connections()

# %%
gm.shape




# %%
vecs.shape

# %%
seq_pos

# %%
# vecs[-1][mm.head_source_index(4, 11), mm.head_target_index(5, 1, 'k')]
vecs[-1][mm.mlp_source_index(0), mm.head_target_index(9, 8, 'q')]




# %%

cd.model.tokenizer.decode(open_web_text_tokens[9])




# %%
cd = ld.get_circuit_discovery_for_seq(9, -5)
cd.print_attn_heads_and_mlps_in_graph()

# %%
cd.visualize_graph()
# %%
cd.component_lens_at_loc([1, 0, 2])



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

v = cd.get_graph_edge_matrix()

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
analyze = analyzer
layer_thresh = 1
k = 100


source_labels = analyze.get_source_labels()
target_labels = analyze.get_target_labels()

total = analyze.graph_matrix_total[
    : analyze.methods.num_head_sources, : analyze.methods.num_head_targets
]

total[: layer_thresh * analyze.model.cfg.n_heads, :] = 0

top_vals, top_indices = torch.topk(total.flatten(), k=k)
top_indices = torch.stack(torch.unravel_index(top_indices, total.shape)).T

print('shape', top_indices.shape)

top_list = []
for v, source_target in zip(top_vals, top_indices):
    source, target = source_target
    source, target = source, target

    top_list.append((source_labels[source], target_labels[target], v))

pprint(top_list)

# %%
source = (8, 7)
target = (9, 3, 'v')


indices = gm[:, mm.head_source_index(*source), mm.head_target_index(*target)].nonzero().squeeze()
len(indices)

# %%
source_target = seq_pos[indices]

# %%
source_target



# %%
i = 3
seq, pos = source_target[i]

cd = ld.get_circuit_discovery_for_seq(seq, pos, is_target_index=False)
# cd = ld_other.get_circuit_discovery_for_seq(seq, pos, is_target_index=False)
cd.print_attn_heads_and_mlps_in_graph()

# %%
cd.visualize_graph()
# %%
cd.visualize_attn_heads_in_graph()
# %%
cd.component_lens_at_loc([0])
# %%

analyze = analyzer
layer_thresh = 1
k = 100


source_labels = analyze.get_source_labels()
target_labels = analyze.get_target_labels()

summer = gm[indices].sum(dim=0)

total = summer[
    : analyze.methods.num_head_sources, : analyze.methods.num_head_targets
]

total[: layer_thresh * analyze.model.cfg.n_heads, :] = 0

top_vals, top_indices = torch.topk(total.flatten(), k=k)
top_indices = torch.stack(torch.unravel_index(top_indices, total.shape)).T

print('shape', top_indices.shape)

top_list = []
for v, source_target in zip(top_vals, top_indices):
    source, target = source_target
    source, target = source, target

    top_list.append((source_labels[source], target_labels[target], v))

pprint(top_list)

# %%
analyzer.get_top_head_connections(source_dest_list=[
    

    # ((1, 10), (8, 7, 'v'))
])

# %%
analyzer.graph_matrices[:, mm.head_source_index(4, 4)].sum(dim=0).argmax()

# %%
analyzer.get_target_labels()[400]




# %%
analyzer.get_top_head_connections(source_dest_list=[
    # ((8, 7), (9, 3, 'v'))
])

# %%
g, sp = analyzer.get_matrices_given_source_dest(source_dest_list=[
    # ((8, 7), (9, 3, 'v'))
    ((1, 10), (8, 7, 'v')),
    # ((8, 7), (10, 9, 'q'))
    # ((1, 7), (9, 3, 'v'))
])

# %%
g = analyzer.graph_matrices
sp = analyzer.seq_pos

indices = g[:, mm.head_source_index(6, 9), mm.head_target_index(10, 1, 'q')].nonzero().squeeze()
source_dest = sp[indices]

# %%
source_dest
# %%
source_target = source_dest
i = 3
seq, pos = source_target[i]
# seq, pos = (30, 121)

cd = ld.get_circuit_discovery_for_seq(seq, pos, is_target_index=False)
# cd = ld_other.get_circuit_discovery_for_seq(seq, pos, is_target_index=False)
cd.print_attn_heads_and_mlps_in_graph()

# %%
cd.visualize_graph()
# %%
cd.visualize_attn_heads_in_graph()

# %%
g = cd.get_graph_edge_matrix()

# %%
g[mm.head_source_index(8, 7), mm.head_target_index(10, 9, 'q')]
