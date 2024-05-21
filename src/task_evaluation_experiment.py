# %%
%load_ext autoreload
%autoreload 2


# %%
import torch
import time

from task_evaluation import TaskEvaluation
from data.ioi_dataset import gen_templated_prompts
from circuit_discovery import CircuitDiscovery, only_feature
from circuit_lens import CircuitComponent

# %%

torch.set_grad_enabled(False)


# %%
dataset_prompts = gen_templated_prompts(template_idex=1)

# %%
def component_filter(component: str):
    return component in [
        CircuitComponent.Z_FEATURE,
        CircuitComponent.MLP_FEATURE,
        CircuitComponent.ATTN_HEAD,
        CircuitComponent.UNEMBED,
        # CircuitComponent.UNEMBED_AT_TOKEN,
        CircuitComponent.EMBED,
        CircuitComponent.POS_EMBED,
        # CircuitComponent.BIAS_O,
        CircuitComponent.Z_SAE_ERROR,
        # CircuitComponent.Z_SAE_BIAS,
        # CircuitComponent.TRANSCODER_ERROR,
        # CircuitComponent.TRANSCODER_BIAS,
    ]

passes = 1
node_contributors = 1

def strategy(cd: CircuitDiscovery):
    for _ in range(passes):
        cd.add_greedy_pass(contributors_per_node=node_contributors)
        cd.add_greedy_pass_against_all_existing_nodes(contributors_per_node=node_contributors, skip_z_features=True, layer_threshold=5)

task_eval = TaskEvaluation(prompts=dataset_prompts, eval_index=-2, circuit_discovery_strategy=strategy, allowed_components_filter=component_filter)


cd = task_eval.get_circuit_discovery_for_prompt(10)
cd.print_attn_heads_and_mlps_in_graph()

# %%
attn_heads,mlps = cd.get_heads_and_mlps_in_graph_at_seq()

mlps

# %%
_ = task_eval.evalute_logit_diff_on_task(N=10, edge_based_graph_evaluation=False, include_all_heads=True, include_all_mlps=False)

# %%
start = time.time()

# %%
_, cache = cd.model.run_with_cache("hey there i'm danny and I like to eat pie")

# %%
cache['mlp_out', 0].shape
# %%




# cd = task_eval.evaluate_circuit_discovery_for_prompt(1, k=5)

cd = task_eval.get_circuit_discovery_for_prompt(0)
cd.print_attn_heads_and_mlps_in_graph()

print(f"Time taken: {time.time() - start}")

# %%

# %%
cd.component_lens_at_loc([0])


# %%
task_eval.evaluate_circuit_discovery_for_prompt(1, k=5)


# %%
# cd.visualize_graph(begin_layer=5)
# cd.visualize_graph(begin_layer=3)
cd.visualize_graph(begin_layer=0)

# %%
cd.print_attn_heads_and_mlps_in_graph()

# %%
cd.visualize_attn_heads_in_graph()

# %%
l = cd.model(cd.tokens)

# %%
l.shape

# %%
list(task_eval.mean_cache.keys())







# %%
cd = task_eval.get_circuit_discovery_for_prompt(0)

# %%
cd.visualize_graph()

# %%
cd.visualize_graph_performance_against_mean_ablation(task_eval.mean_cache, include_all_mlps=True, k=10)

# %%
task_eval.evaluate_circuit_discovery_for_prompt(0, k=10)

# %%
a = ['as', 'asdfasdf', 'dd']
sorted(a, key=lambda x: len(x))

# %%
r = cd.root_node

id_value = {node.tuple_id: value for (node, value) in cd.root_node.top_k_contributors}

# %%
id_value

# %%
r.contributors_in_graph

# %%
sorted_contributors = sorted(r.contributors_in_graph, key=lambda x: -id_value[x.tuple_id])

# %%

[c.tuple_id for c in sorted_contributors]

# %%

[c.tuple_id for c in r.contributors_in_graph]

# %%
cd

# %%
cd.component_lens_at_loc(loc=[])

# %%
