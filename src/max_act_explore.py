# %%
%load_ext autoreload
%autoreload 2

# %%

import torch

from max_act_analysis import MaxActAnalysis,open_web_text_tokens
from aug_interp_prompts import main_aug_interp_prompt
from openai_utils import gen_openai_completion
from discovery_strategies import (
    create_filter,
    create_simple_greedy_strategy,
    create_top_contributor_strategy,
)

# %%
torch.set_grad_enabled(False)


# %%
feature = 16513
# feature = 16401
layer = 8
# feature = 15647
# num_examples = 1000
num_examples = 5_000

strategy = create_simple_greedy_strategy(
    passes=1,
    node_contributors=1,
    minimal=True,
)

analyze = MaxActAnalysis("attn", layer, feature, num_sequences=num_examples, batch_size=128, strategy=strategy)
analyze.show_top_active_examples(num_examples=5)

# %%
mini_examples = analyze.get_context_referenced_prompts_for_range(0, 10)


# %%
p = main_aug_interp_prompt(mini_examples)

print(p)
# %%
gen_openai_completion(p)

# %%


i = 70
# strategy = create_simple_greedy_strategy(
#     passes=1,
#     node_contributors=1,
#     minimal=True,
#     # do_sub_pass=True,
#     # sub_passes=1,
#     # sub_pass_layer_threshold=-1,
#     # sub_pass_minimal=True,
# )

# sae_error = True
# no_sae_error = not sae_error

# # strategy = create_top_contributor_strategy(
# #     num_greedy_passes=5,
# #     layer_threshold=4
# # )


# # comp_filter = create_filter(no_sae_error=not sae_error)
# # comp_filter = create_filter(no_sae_error=no_sae_error)
# comp_filter = create_filter()
analyze.show_example(i)

cd = analyze.get_circuit_discovery_for_max_activating_example(i)

cd.print_attn_heads_and_mlps_in_graph()

# %%
i = 30

p, ap = analyze.get_context_referenced_prompt(i, merge_nearby_context=False)

print(p)
print()
print(ap)

# %%





# %%

cd.visualize_graph()

# %%
analyze.show_example(i, no_cutoff=True)

# %%
analyze.num_examples



# %%

cd.visualize_attn_heads_in_graph()

# %%
cd.component_lens_at_loc([0, 'q'])

# %%
cd.component_lens_at_loc_on_graph([0, 'q', 0, 0, 'v', 0, 0, 0, 'v'])


# %%
cd.component_lens_at_loc([0, 'v'])

# %%
cd.get_min_max_pos_in_graph()
