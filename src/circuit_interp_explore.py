# %%
%load_ext autoreload
%autoreload 2

# %%
import torch

from circuit_interp import CircuitInterp
from circuit_discovery import CircuitDiscovery
from data.ioi_dataset import gen_templated_prompts
from discovery_strategies import create_simple_greedy_strategy, create_filter

# %%
torch.set_grad_enabled(False)

# %%

dataset_prompts = gen_templated_prompts(template_idex=1, N=500)

# %% 

strat = create_simple_greedy_strategy(passes=1)
no_sae_error_filter = create_filter(no_sae_error=True)

# %%
pi = 2

print(dataset_prompts[pi]["text"])

cd = CircuitDiscovery(
    dataset_prompts[pi]["text"],
    token=dataset_prompts[pi]["correct"],
    allowed_components_filter=no_sae_error_filter,
)
strat(cd)
cd.print_attn_heads_and_mlps_in_graph()
# %%


# %%

cd.visualize_graph()

# %%
cd.root_node.sorted_contributors_in_graph[0]

# %%

ci = CircuitInterp(cd, strat, num_max_act_seqs_in_prompt=10, num_open_web_text_seq=5000)

# %%
ex_dick = ci.interpret_heads_in_circuit(visualize=True, layer_threshold=0)

# %%

for k, v in ex_dick.items():
    _, layer, head, *_ = k

    print(f"L{layer}H{head}")
    print("Label:")
    print(v[0].strip('\n'))
    print()
    print("Head Description")
    print(v[1].strip('\n'))
    print()



# %%
res = ci.get_attn_head_interp(
    cd.root_node.sorted_contributors_in_graph[0].sorted_contributors_in_graph[0],
    "This neuron causes the model to strongly predict ' Richard' as the next token.",
    visualize=True,
)

# %%
i = 2
items = list(ci.child_max_acts_cache.items())
item = items[2]

print(item[0])
item[1]['v'].show_top_active_examples()



# %%
print(res)


# %%
print(res[-1])

# %%
