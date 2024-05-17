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

from torch import tensor

# %%
circuit_lens = CircuitLens("14. Colorado 15. Missouri 16. Illinois 17")

# circuit_lens = CircuitLens("Mary and Jeff went to the store, and Mary gave an apple to Jeff")
# circuit_lens = CircuitLens("He cut the finger right off my hand")


# %%
layer = 9
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



l @ z, l @ e, z @ e
# l @ e, l @ z, z @ e
# %%
error.norm(), e_recon.norm(), (error - e_recon).norm()


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
unembed_children = circuit_lens.get_unembed_lens_for_prompt_token(-2)
print(unembed_children)

# %%
uu = unembed_children[2]()
print(uu)

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
