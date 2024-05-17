# %%
%load_ext autoreload
%autoreload 2

# %%
from circuit_lens import CircuitLens
from memory import get_gpu_memory
from plotly_utils import *
from pprint import pprint
import torch
import einops
import plotly.express as px
from transformer_lens import HookedTransformer
from graphviz import Digraph

from torch import tensor

# %%
G = Digraph()
G.graph_attr.update(rankdir='BT', newrank='true')
G.node_attr.update(shape="box", style="rounded")

with G.subgraph() as subgraph:
    subgraph.attr(rank='same')

    subgraph.node('A', shape="none")
    subgraph.node('B')  
    subgraph.edge('A', 'B', style="invis", minlen="2")


with G.subgraph() as subgraph:
    subgraph.attr(rank='same')

    subgraph.node('C', shape="none")
    subgraph.node('D')  
    subgraph.node("E")

    # subgraph.edge('C', 'D', style="invis", minlen="2")


G.edges(["AC", "AD", "BE"])

G

# %%
G = Digraph(name='feature circuit')

# Add nodes and assign them to groups
G.node('A', group='1')
G.node('B', group='1')
G.node('C', group='2')
G.node('D', group='2')

# To ensure they are all in the same rank
with G.subgraph(nm) as s:
    s.attr(rank='same')
    s.node('A')
    s.node('B')
    s.node('C')
    s.node('D')

# Add edges
G.edge('A', 'C')
G.edge('B', 'D')

# Render the graph
G






# %%
circuit_lens = CircuitLens("14. Colorado 15. Missouri 16. Illinois 17")

# circuit_lens = CircuitLens("Mary and Jeff went to the store, and Mary gave an apple to Jeff")
# circuit_lens = CircuitLens("He cut the finger right off my hand")

# %%
list(set([('a', 'b')]))
('a', 'b') in set([('a', 'b')])

# %%
a = set(['a', 'b'])
print(a)
a.discard('a')
print(a)

# %%
tuple(list((1, 2, 3)))

# %%

(1, 2, 3)[:2]












# %%
layer = 8
seq_index = 7

layer_z = einops.rearrange(
circuit_lens.cache["z", layer][0, seq_index],
"n_heads d_head -> (n_heads d_head)",
)
_, z_recon, z_acts, _, _ = circuit_lens.z_saes[layer](layer_z)

error = layer_z - z_recon

_, e_recon, e_acts, _, _ = circuit_lens.z_saes[layer](error)


l = layer_z / layer_z.norm(dim=-1, keepdim=True)
e = error / error.norm(dim=-1, keepdim=True)
z = z_recon / z_recon.norm(dim=-1, keepdim=True)



l @ z, l @ e, z @ e, layer_z.norm(), z_recon.norm(),error.norm()
# l @ e, l @ z, z @ e
# %%
error.norm(), e_recon.norm(), (error - e_recon).norm(), layer_z.norm()


# %%
# e_acts

e_acts.max(), z_acts.max()

# %%
af = circuit_lens.get_active_features(seq_index, cache=False)

# %%
vectors = af.get_attn_feature_vectors(9)

# %%
vectors /= vectors.norm(dim=-1, keepdim=True)

# %%
cos = einops.einsum(vectors, vectors, "i d, j d -> i j")

cos[range(13), range(13)] = 0

# %%
px.histogram(cos.max(dim=-1).values.cpu().detach().float()).show()

# %%
lens = circuit_lens

lens.cache['attn_scores', 0].shape

pattern = lens.cache['attn_scores', 0].softmax(dim=-1)

torch.allclose(pattern, lens.cache['pattern', 0])


# %%

model = HookedTransformer.from_pretrained("gpt2-small", center_writing_weights=False)
# %%
unembed_children = circuit_lens.get_unembed_lens_for_prompt_token(0, visualize=True)

# %%
(1, 2, (3, 4)) == (1, 2, (3, 4))





# %%
import time
import random

start = time.time()

n_tokens = circuit_lens.n_tokens

for i in range(4000):
    token = random.randint(1, 8)

    unembed_children = circuit_lens.get_unembed_lens_for_prompt_token(token, visualize=False)

print(time.time() - start)

# %%
af = circuit_lens.get_active_features(7, cache=False)

# %%
af.vectors.shape



# %%
uu = unembed_children[1]()

# %%
q = uu[0]('v')

# %%
b = q[2]()

# %%
b[0]()





# %%
l9 = unembed_children[0]()

# %%
# %%
l9_q = l9[0]('q')

# %%
l8 = l9_q[0]()

# %%
l8_q = l8[0]('q')



# %%
l8_q = l8[0]('q')
# %%
l8_k = l8[0]('k')

# %%
l8_k[4]()

# %%


l7 = l8_q[3]()

# %%
l7_k = l7[0]('k')
# %%
l7_k[2]()
