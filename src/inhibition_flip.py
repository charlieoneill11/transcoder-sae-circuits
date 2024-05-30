# %%
# %load_ext autoreload
# %autoreload 2

# %%
import torch
import time
import plotly.express as px
import matplotlib.pyplot as plt
import einops
import pandas as pd

from discovery_strategies import create_filter, create_simple_greedy_strategy
from task_evaluation import TaskEvaluation
from data.ioi_dataset import gen_templated_prompts, BABA_TEMPLATES
from plotly_utils import *
from rich import print as rprint
from rich.table import Table
from z_sae import ZSAE
from ranking_utils import visualize_top_tokens, visualize_top_tokens_for_seq
from tqdm import trange

from transformer_lens import HookedTransformer, utils

# from data.ioi_dataset import GT_GROUND_TRUTH_HEADS
from memory import get_gpu_memory

# %%
torch.set_grad_enabled(False)
get_gpu_memory()

# %%
model = HookedTransformer.from_pretrained("gpt2-small")
z8 = ZSAE.load_zsae_for_layer(8)

# %%
baba_feature_i = 16513
abba_feature_i = 7861

baba_feature = z8.W_dec[baba_feature_i]
abba_feature = z8.W_dec[abba_feature_i]


# %%
template_idx = 10
N = 100

abba_dataset = gen_templated_prompts(
    prompt_type="abba", template_idex=template_idx, N=N
)
baba_dataset = gen_templated_prompts(
    prompt_type="baba", template_idex=template_idx, N=N
)

# %%
N = 100

data_type = "BABA"

base_correct = 0
base_counter = 0

steered_correct = 0
steered_counter = 0

if data_type == "ABBA":
    dataset = abba_dataset

    abba_coeff = -10
    baba_coeff = 15
else:
    dataset = baba_dataset
    abba_coeff = 30
    baba_coeff = -5

    # abba_coeff = 30
    # baba_coeff = -5

for i in trange(N):
    prompt_ex = dataset[i]
    prompt = prompt_ex["text"]
    tokens = model.to_tokens(prompt)
    str_tokens = model.to_str_tokens(prompt)

    correct_token = model.to_single_token(prompt_ex["correct"])
    counter_token = model.to_single_token(prompt_ex["counter"])

    base_logits = model(tokens)
    base_softmax = base_logits.softmax(dim=-1)[0, -1]

    base_correct += base_softmax[correct_token].item()
    base_counter += base_softmax[counter_token].item()

    # print(prompt)
    # visualize_top_tokens(model, base_logits)

    big_abba = abba_feature - baba_feature
    # big_abba = einops.rearrange(big_abba, "(n_heads d_head) -> n_heads d_head", n_heads=12)

    abba = einops.rearrange(
        abba_feature, "(n_heads d_head) -> n_heads d_head", n_heads=12
    )
    baba = einops.rearrange(
        baba_feature, "(n_heads d_head) -> n_heads d_head", n_heads=12
    )

    def l8_hook(acts, hook):
        acts[:, -1, :, :] += abba_coeff * abba.unsqueeze(0)
        acts[:, -1, :, :] += baba_coeff * baba.unsqueeze(0)

        return acts

    steered_logits = model.run_with_hooks(
        tokens, fwd_hooks=[(utils.get_act_name("z", 8), l8_hook)]
    )

    steered_softmax = steered_logits.softmax(dim=-1)[0, -1]

    steered_correct += steered_softmax[correct_token].item()
    steered_counter += steered_softmax[counter_token].item()

# visualize_top_tokens_for_seq(
#     model, tokens, steered_logits, -1, token=prompt_ex["correct"]
# )

steered_correct /= N
steered_counter /= N

base_correct /= N
base_counter /= N

data = {
    # 'Category': ['A', 'B', 'C', 'D'],
    # 'Values': [10, 20, 30, 40]
    "Category": [
        "Correct Name (Base)",
        "Incorrect Name (Base)",
        "Correct Name (Steered)",
        "Incorrect Name (Steered)",
    ],
    "Probability": [base_correct, base_counter, steered_correct, steered_counter],
}

df = pd.DataFrame(data)

fig = px.bar(
    df,
    x="Category",
    y="Probability",
    # x=["Base Correct", "Base Incorrect", "Steered Correct", "Steered Incorrect"],
    # y=[base_correct, base_counter, steered_correct, steered_counter],
    color="Category",
    color_discrete_sequence=["blue", "blue", "red", "red"],
    title=f"{data_type} Probability Mass Before/After Steering ({N} Samples)",
)

fig.update_layout(showlegend=False)

fig.show()

# %%
cache["z", 0].shape

model.to_single_token(" Jeff")

# %%
px.bar(x=["x", "y"], y=[9, 1]).show()


# %%
base_softmax.shape
