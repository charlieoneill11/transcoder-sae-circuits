# %%

%load_ext autoreload
%autoreload 2

# %%
import torch
import time
import plotly.express as px
import matplotlib.pyplot as plt

from discovery_strategies import create_filter, create_simple_greedy_strategy
from task_evaluation import TaskEvaluation
from data.ioi_dataset import gen_templated_prompts, BABA_TEMPLATES
from plotly_utils import *
from rich import print as rprint
from rich.table import Table

# from data.ioi_dataset import GT_GROUND_TRUTH_HEADS
from memory import get_gpu_memory

# %%
torch.set_grad_enabled(False)
get_gpu_memory()


# %%

for template_idx in [0, 1, 5, 6, 11, 14]:
    N = 500

    template = BABA_TEMPLATES[template_idx].replace("[A]", '[Noun]').replace("[B]", '[Noun]')

    abba_dataset = gen_templated_prompts(
        prompt_type="abba", template_idex=template_idx, N=N
    )
    baba_dataset = gen_templated_prompts(
        prompt_type="baba", template_idex=template_idx, N=N
    )

    component_filter = create_filter(no_sae_error=True)
    strategy = create_simple_greedy_strategy(
        passes=5
    )

    abba_eval = TaskEvaluation(
        prompts=abba_dataset,
        circuit_discovery_strategy=strategy,
        allowed_components_filter=component_filter,
        no_mean_cache=True
    )

    baba_eval = TaskEvaluation(
        prompts=baba_dataset,
        circuit_discovery_strategy=strategy,
        allowed_components_filter=component_filter,
        no_mean_cache=True
    )

    unique_N = 100

    abba_unique_features = abba_eval.get_unique_features_for_heads_over_dataset(N=unique_N)
    baba_unique_features = baba_eval.get_unique_features_for_heads_over_dataset(N=unique_N)

    inhibition_heads = [
        (7, 3),
        (7, 9),
        (8, 6),
        (8, 10)
    ]

    table = Table(
        show_header=True, header_style="bold yellow", show_lines=True, title=template
    )

    table.add_column("Head")
    table.add_column("ABBA Features")
    table.add_column("BABA Features")

    for layer, head in inhibition_heads:
        abba_set = abba_unique_features.get_unique_features_for_head(layer=layer, head=head, N=unique_N)
        baba_set = baba_unique_features.get_unique_features_for_head(layer=layer, head=head, N=unique_N)

        abba_features = ", ".join([str(f) for f in abba_set - baba_set])
        baba_features = ", ".join([str(f) for f in baba_set - abba_set])

        table.add_row(
            f"L{layer}H{head}",
            abba_features,
            baba_features
        )

    rprint(table)


# %%
layer = 8
head = 6
num_samples = 30

_ = abba_unique_features.get_feature_counts_for_head_over_range(layer=layer, head=head, N=unique_N, num_samples=num_samples)
_ = baba_unique_features.get_feature_counts_for_head_over_range(layer=layer, head=head, N=unique_N, num_samples=num_samples)


# %%
abba_set = abba_unique_features.get_unique_features_for_head(layer=layer, head=head, N=unique_N)
baba_set = baba_unique_features.get_unique_features_for_head(layer=layer, head=head, N=unique_N)

print("ABBA:", abba_set)
print("BABA:", baba_set)

# %%
print("ABBA - BABA: ", abba_set - baba_set)
print("BABA - ABBA: ", baba_set - abba_set)


# %%
for i in range(10):
    print(abba_dataset[i]['text'])

# %%
i = 6

abba_cd = abba_eval.get_circuit_discovery_for_prompt(i)
abba_cd.print_attn_heads_and_mlps_in_graph()

# abba_cd.visualize_graph(begin_layer=7)

# %%
abba_cd.component_lens_at_loc([0, 0, 'q'])


# %%
abba_eval.get_attn_head_freqs_over_dataset(N=20, return_freqs=False)

# %%
baba_eval.get_attn_head_freqs_over_dataset(N=20, return_freqs=False)

