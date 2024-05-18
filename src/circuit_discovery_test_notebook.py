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

# %%
inverse = "Mary and Jeff went to the store, and Jeff gave an apple to Mary"

# cd = CircuitDiscovery(SUCCESSOR_EXAMPLE_PROMPT, -2, allowed_components_filter=only_feature)
cd = CircuitDiscovery(IOI_EXAMPLE_PROMPT, -2, allowed_components_filter=only_feature)

# %%
# passes = 3

cd.reset_graph()
# for _ in range(passes):
#     cd.add_greedy_pass()
cd.add_greedy_pass(contributors_per_node=2)
# %%

cd.print_attn_heads_and_mlps_in_graph()


# %%

cd.visualize_current_graph_performance(
    head_ablation_type="bos", 
    # head_ablation_type="zero", 
    # include_all_heads=True,
    # include_all_mlps=True
)

# %%
cd.visualize_graph()

# %%
# cd.visualize_current_graph_performance(head_ablation_style="zero", include_all_heads=True, include_all_mlps=True)

# %%
a = [1,2, 3]
b = [4, 5, 6]

a.extend(b)
a



# %%


# %%

cd.print_attn_heads_and_mlps_in_graph()

# %%
cd.visualize_attn_heads_in_graph(value_weighted=False)

# %%



# %%
list(reversed(range(0, 10)))

# %%
cache['v', 0].shape




# %%
gl = cd.get_logits_for_current_graph()

# %%
a = [4, 1, 2, 3]
print('asdf', a.sort())
a




# %%
gl.shape




# %%
model: HookedTransformer = cd.model


# %%
logits = model(IOI_EXAMPLE_PROMPT, return_type="logits")
logits.shape

# %%
tokens  = model.to_tokens(IOI_EXAMPLE_PROMPT)

# %%
loss = utils.lm_cross_entropy_loss(logits, tokens, per_token=True)


# %%
loss.shape

# %%
model(IOI_EXAMPLE_PROMPT, return_type="loss")

# %%
table = Table(show_header=True, header_style="bold yellow", show_lines=True, title="Who de boss")
table.add_column("Token")
table.add_column("Loss")

table.add_row(" Hey", "0.1")
table.add_row(" You", ".2")

rprint(table)


# %%
model.tokenizer.decode(35)

# %%
tokens.squeeze().squeeze().shape



# %%
def visualize_top_tokens(tokens, logits, seq_index, token=None,k=10, model=model):
    tokens = tokens.squeeze()
    logits = logits.squeeze()

    if seq_index < 0:
        seq_index += tokens.size(0)

    if token is None:
        token_title = "Correct Token"

        token = tokens[seq_index + 1]
    else:
        token_title = "Selected Token"
    
    log_probs = F.log_softmax(logits, dim=-1)[seq_index]
    softmax = F.softmax(logits, dim=-1)[seq_index]

    indices = softmax.topk(k).indices.tolist()

    token_rank = (softmax > softmax[token]).sum().item()


    selected_token_table = Table(show_header=True, header_style="bold yellow", show_lines=True, title=token_title)
    selected_token_table.add_column("Rank")
    selected_token_table.add_column("Token")
    selected_token_table.add_column("Prob")
    selected_token_table.add_column("-Log Prob")
    selected_token_table.add_column("Token Index")

    selected_token_table.add_row(
        str(token_rank),
        f"'{model.tokenizer.decode(token)}'",
        f"{softmax[token].item() * 100:.3g}%",
        f"{-log_probs[token].item():.3g}",
        str(int(token))
    )

    top_table = Table(show_header=True, header_style="bold yellow", show_lines=True, title="Top Tokens")
    top_table.add_column("Rank")
    top_table.add_column("Token")
    top_table.add_column("Prob")
    top_table.add_column("-Log Prob")
    top_table.add_column("Token Index")

    for i, index in enumerate(indices):
        top_table.add_row(
            str(i),
            f"'{model.tokenizer.decode(index)}'",
            f"{softmax[index].item() * 100:.3g}%",
            f"{-log_probs[index].item():.3g}",
            str(index)
        )

    rprint(selected_token_table)
    rprint(top_table)

# %%
# visualize_top_tokens(tokens, logits, -2, token=606)
visualize_top_tokens(tokens, logits, -2) #, token=262)



# %%
list(zip([1, 2], [3, 4], [5, 6]))

# %%
def hook(
    acts,
    hook: HookPoint
):
    print(hook.name, hook.layer())

ftr = lambda x: x.endswith('z')

_ =  model.run_with_hooks(IOI_EXAMPLE_PROMPT,
                     fwd_hooks=[(ftr, hook)])

# %%
logits, cache = model.run_with_cache(IOI_EXAMPLE_PROMPT)

# %%
cache['v', 0].shape

# %%
list(cache.keys())

# %%
logits.shape


# %%
line(cd.mlp_transcoders[0].W_dec[0].relu())

# %%
pre_layer = 4
post_layer = 7

t_pre = cd.mlp_transcoders[pre_layer]
t_post = cd.mlp_transcoders[post_layer]

t = None
b = None
N = 1000
# N = 1
feature = 26


for i in range(N):
    # x = t4.W_dec[:100].sum(0)
    x = 0.5 * (2 * torch.rand_like(t_pre.W_dec[0]) - 1)
    x /= x.norm()
    x *= 10


    y = 10 * (t_pre.W_dec[feature] / t_pre.W_dec[feature].norm())
    # print('x norm', x.norm())

    # x = torch.zeros_like(t4.W_dec[1])
    # x = -t5.b_dec
    x -= t_post.b_dec

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
    new = pre
    # new = ((pre_y.relu() - pre.relu()) > 0).float()
    # new = pre_y.relu() - pre.relu()


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

# %%
(pre > 0).float().max()


# %%
t_pre.W_dec[5].norm()

# %%
(t_pre.W_dec.norm(dim=-1) == 1).sum()

# %%
t_post.b_dec.norm()



# %%
# x = t4.W_dec[:100].sum(0)
# x /= x.norm()
# x *= 10000
# x = 0.5 * (2 * torch.rand_like(t4.W_dec[0]) - 1)


x = t_pre.W_dec[:100].sum(0)
# x = torch.zeros_like(t4.W_dec[1])
# x = -t5.b_dec
# x -= t5.b_dec

# x = t4.W_dec[1]

(print(x.shape))

pre = einops.einsum(
    x,
    t_post.W_enc,
    "d_in, d_in d_sae -> d_sae"
) + t_post.b_enc

line(pre.relu())


# %%
line(t_post.b_enc.relu())

# %%
x.shape

# %%
torch.rand_like(x).max()


# %%
