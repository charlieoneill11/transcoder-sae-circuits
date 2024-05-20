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

from rich import print as rprint
from rich.table import Table

import torch
import time
import einops


# %%
torch.set_grad_enabled(False)

# %%
cd = CircuitDiscovery(IOI_EXAMPLE_PROMPT, -2, allowed_components_filter=only_feature)


# %%
pre_layer = 4
post_layer = 7
N = 1

t_pre = cd.mlp_transcoders[pre_layer]
t_post = cd.mlp_transcoders[post_layer]

# feature = 9500
feature = 51

total = None

for i in range(N):
    x = 0.5 * (2 * torch.rand_like(t_pre.W_dec[0]) - 1)
    x /= x.norm()
    # x *= 5
    x -= t_post.b_dec

    y = t_pre.W_dec
    y /= y.norm(dim=-1, keepdim=True)
    y *= 100

    feature_enc = t_post.W_enc[:, feature]


    only_noise = einops.einsum(
        x,
        feature_enc,
        "d, d -> "
    ) + t_post.b_enc[feature]

    print(only_noise)

    # with_pre = einops.einsum(
    #     y + x.unsqueeze(0),
    #     feature_enc,
    #     "pre_feature d, d -> pre_feature"
    # ) + t_post.b_enc[feature]

    with_pre = einops.einsum(
        # y ,
        y + x.unsqueeze(0),
        feature_enc,
        "pre_feature d, d -> pre_feature"
    ) + t_post.b_enc[feature]

    # diff = with_pre.relu() - only_noise.relu().unsqueeze(0)
    # diff = with_pre.relu()
    diff = with_pre
    # diff = only_noise

    if total is None:
        total = diff
    else:
        total += diff

total = total / N

line(
    # total.relu(),
    total,
)



# %%
pre_layer = 4
post_layer = 7

t_pre = cd.mlp_transcoders[pre_layer]
t_post = cd.mlp_transcoders[post_layer]

t = None
b = None
N = 1000
# N = 1
# feature = 11379
feature = 16089
# feature = 15976
# feature = 7323


for i in range(N):
    # x = t4.W_dec[:100].sum(0)
    x = 0.5 * (2 * torch.rand_like(t_pre.W_dec[0]) - 1)
    x /= x.norm()
    x *= 10


    y = t_pre.W_dec[feature]
    # y = t_pre.W_dec[feature:feature + 10000].sum(0)
    y /= y.norm()
    y *= 10 #* (t_pre.W_dec[feature] / t_pre.W_dec[feature].norm())
    # print('x norm', x.norm())
    # print('xy norm', x.norm(), y.norm())

    # x = torch.zeros_like(t4.W_dec[1])
    # x = -t5.b_dec
    x -= t_post.b_dec
    # y -= t_post.b_dec

    pre = einops.einsum(
        x,
        t_post.W_enc,
        "d_in, d_in d_sae -> d_sae"
    ) + t_post.b_enc

    pre_y = einops.einsum(
        y + x,
        t_post.W_enc,
        "d_in, d_in d_sae -> d_sae"
    ) + t_post.b_enc
    

    pre_yy = einops.einsum(
        y,
        t_post.W_enc,
        "d_in, d_in d_sae -> d_sae"
    ) + t_post.b_enc

    # new = (pre_y > 0).float() - (pre > 0).float()
    # new = pre_yy
    # new = pre
    # new = ((pre_y.relu() - pre.relu()) > 0).float()
    new = pre_y.relu() - pre.relu()


    if t is None:
        t = new
        # t = pre_y.relu() - pre.relu()
        # t = pre_y.relu() 
    else:
        t += new
        # t += pre_y.relu() - pre.relu()
        # t += pre_y.relu() 


t = t / N

line(
    t.relu(),
    # t,
    title=f"MLP {pre_layer} (Feature #{feature}) -> MLP {post_layer}",
    labels={
        'x': f"MLP {post_layer} Feature",
        'y': 'Activation'
    }
)

# %% [markdown]
# # The final lad

# %%
pre_layer = 4
post_layer = 7

t_pre = cd.mlp_transcoders[pre_layer]
t_post = cd.mlp_transcoders[post_layer]

t = None
b = None
N = 1000
# N = 1
# feature = 11379
feature = 16089
# feature = 15976
# feature = 7323

start = 0
amount = 100


for i in range(N):
    # x = t4.W_dec[:100].sum(0)
    x = 0.5 * (2 * torch.rand_like(t_pre.W_dec[0]) - 1)
    x /= x.norm()
    x *= 10


    y = t_pre.W_dec
    # y = t_pre.W_dec[feature:feature + 10000].sum(0)
    y /= y.norm(dim=-1, keepdim=True)
    y *= 20 #* (t_pre.W_dec[feature] / t_pre.W_dec[feature].norm())
    # print('x norm', x.norm())
    # print('xy norm', x.norm(), y.norm())

    # x = torch.zeros_like(t4.W_dec[1])
    # x = -t5.b_dec
    x -= t_post.b_dec
    # y -= t_post.b_dec

    pre = einops.einsum(
        x,
        t_post.W_enc[:, start:start+amount],
        "d_in, d_in d_sae -> d_sae"
    ) + t_post.b_enc[start:start+amount]

    pre_y = einops.einsum(
        y + x.unsqueeze(0),
        t_post.W_enc[:, start:start+amount],
        "d_sae_in d_in, d_in d_sae -> d_sae_in d_sae"
    ) + t_post.b_enc[start:start+amount]
    

    # pre_yy = einops.einsum(
    #     y,
    #     t_post.W_enc,
    #     "d_in, d_in d_sae -> d_sae"
    # ) + t_post.b_enc

    # new = (pre_y > 0).float() - (pre > 0).float()
    # new = pre_yy
    # new = pre
    # new = ((pre_y.relu() - pre.relu()) > 0).float()
    new = pre_y.relu() - pre.unsqueeze(0).relu()


    if t is None:
        t = new
        # t = pre_y.relu() - pre.relu()
        # t = pre_y.relu() 
    else:
        t += new
        # t += pre_y.relu() - pre.relu()
        # t += pre_y.relu() 


t = t / N

(t.sum(dim=0) > 0).sum()

# %%
mlp_7_feature = 52


line(
    t[:, mlp_7_feature].relu(),
    # t,
    # title=f"MLP {pre_layer} (Feature #{feature}) -> MLP {post_layer}",
    title=f"MLP {pre_layer} -> MLP {post_layer} (Feature #{mlp_7_feature}) ",
    labels={
        'x': f"MLP {pre_layer} Feature",
        'y': 'Activation'
    }
)

# %%
z_i = 2
mlp_i = 3
l_feature = 16579

z = cd.z_saes[z_i]
t = cd.mlp_transcoders[mlp_i]

better_z = einops.rearrange(z.W_dec, 'sae (n d) -> sae n d', n=12)
better_z = einops.einsum(better_z, cd.model.W_O[z_i], "sae n d, n d d_model -> sae d_model")

b = einops.einsum(
    better_z, t.W_enc[:, l_feature], "sae d_model, d_model -> sae"   
)

line(b)

# %%
