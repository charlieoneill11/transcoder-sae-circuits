# %%
%load_ext autoreload
%autoreload 2

# %%
from example_prompts import SUCCESSOR_EXAMPLE_PROMPT, IOI_EXAMPLE_PROMPT
from circuit_discovery import CircuitDiscovery, only_feature
from transformer_lens import HookedTransformer, utils
import torch.nn.functional as F

from rich import print as rprint
from rich.table import Table

import torch
import time


# %%
torch.set_grad_enabled(False)

# %%

# %%
start = time.time()


inverse = "Mary and Jeff went to the store, and Jeff gave an apple to Mary"

cd = CircuitDiscovery(IOI_EXAMPLE_PROMPT, -2, allowed_components_filter=only_feature)

cd.add_greedy_first_pass()

print("\n Elapsed", time.time() - start)

# %%

cd.visualize_graph()

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
    selected_token_table.add_column("Neg Log Prob")
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
    top_table.add_column("Neg Log Prob")
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
visualize_top_tokens(tokens, logits, -2, token=606)
# visualize_top_tokens(tokens, logits, -2) #, token=262)



# %%
