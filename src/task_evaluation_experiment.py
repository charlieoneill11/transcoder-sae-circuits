# %%

%load_ext autoreload
%autoreload 2


# %%
import torch
import time
import plotly.express as px
import matplotlib.pyplot as plt

from task_evaluation import TaskEvaluation
from data.ioi_dataset import gen_templated_prompts
from data.greater_than_dataset import generate_greater_than_dataset
from circuit_discovery import CircuitDiscovery, only_feature
from circuit_lens import CircuitComponent
from plotly_utils import *
from data.ioi_dataset import IOI_GROUND_TRUTH_HEADS
# from data.ioi_dataset import GT_GROUND_TRUTH_HEADS
from memory import get_gpu_memory
from sklearn import metrics
from tqdm import trange

from utils import get_attn_head_roc


# %%
torch.set_grad_enabled(False)
get_gpu_memory()
# %%
dataset_prompts = gen_templated_prompts(template_idex=1, N=500)


# dataset_prompts = generate_greater_than_dataset(N=100)


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


pass_based = True

passes = 5
node_contributors = 1
first_pass_minimal = True

sub_passes = 3
do_sub_pass = False
layer_thres = 9
minimal = True


num_greedy_passes = 20
k = 1
N = 30

thres = 4


# # Danny and Charlie... Charlie gave shit to Danny
# # Danny and Charlie... Charlie gave shit to Charlie
# # Danny and Charlie... Danny gave shit to Danny
# #

def strategy(cd: CircuitDiscovery):
    if pass_based:
        for _ in range(passes):
            cd.add_greedy_pass(contributors_per_node=node_contributors, minimal=first_pass_minimal)

            if do_sub_pass:
                for _ in range(sub_passes):
                    cd.add_greedy_pass_against_all_existing_nodes(contributors_per_node=node_contributors, skip_z_features=True, layer_threshold=layer_thres, minimal=minimal)
    else:
        for _ in range(num_greedy_passes):
            cd.greedily_add_top_contributors(k=k, reciever_threshold=thres)



task_eval = TaskEvaluation(prompts=dataset_prompts, circuit_discovery_strategy=strategy, allowed_components_filter=component_filter)


# a = task_eval.get_attn_head_freqs_over_dataset(N=N, return_freqs=True)

# %%
ground = task_eval.get_faithfulness_curve_over_data(N=20, attn_head_freq_n=10, faithfulness_intervals=30, rand=False, ioi_ground=True)
base = task_eval.get_faithfulness_curve_over_data(N=20, attn_head_freq_n=10, faithfulness_intervals=30, rand=False, ioi_ground=False)

radd = []
for _ in trange(20):
    radd.append(task_eval.get_faithfulness_curve_over_data(N=20, attn_head_freq_n=10, faithfulness_intervals=30, rand=True, ioi_ground=False, visualize=False))


# %%
big_rad = {}
for rad in radd:
    for k, v in rad.items():
        if k not in big_rad:
            big_rad[k] = 0
        big_rad[k] += v

for k in big_rad:
    big_rad[k] /= 20

rad = big_rad

# %%
plt.plot([float(k) for k in ground.keys()], ground.values(), label="Ground Truth")
plt.plot([float(k) for k in rad.keys()], base.values(), label="Circuit Discovery")
plt.plot([float(k) for k in base.keys()], rad.values(), label="Random")
plt.legend()
plt.grid(True)
plt.grid(color='gray', linestyle='--', linewidth=0.2)
plt.title("IOI Faithfulness (Mean Ablation)")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.margins(0)
plt.ylabel("Normalized KL")
plt.xlabel("# Heads")
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
plt.show()


# %%
line(torch.tensor([[1, 2, 3], [4, 5, 6]]))

# %%
IOI_GROUND_TRUTH_HEADS.sum()





# %%
cd = task_eval.get_circuit_discovery_for_prompt(20)
cd.print_attn_heads_and_mlps_in_graph()

# %%
cd.visualize_graph()

# %%
# f = task_eval.get_features_at_heads_over_dataset(N=30)

# %%
N = 300

counts = task_eval.get_unique_features_for_heads_over_dataset(N=N)
# %%
layer = 9
head = 9
num_samples = 50

data = counts.get_feature_counts_for_head_over_range(layer, head, N, num_samples=num_samples)
layer_data = counts.get_feature_counts_for_layer_over_range(layer, N, num_samples=num_samples)


px.line(
    x=list(data.keys()),
    y=list(data.values()),
    labels={"x": "# IOI Sample Prompts", "y": "# Unique Features"},
    title=f"Unique features associated with L{layer}H{head}"
).show()


px.line(
    x=list(layer_data.keys()),
    y=list(layer_data.values()),
    labels={"x": "# IOI Sample Prompts", "y": "# Unique Features"},
    title=f"Unique features associated with Layer {layer}"
).show()

# %%
counts.get_feature_set_for_head(layer, head)


# %%
i = 3
cd = task_eval.get_circuit_discovery_for_prompt(i)
cd.print_attn_heads_and_mlps_in_graph()

# %%
cd.visualize_graph()

# %%
cd.component_lens_at_loc([0, 0, 'q'])

# %%
cd.visualize_attn_heads_in_graph()



# %%
cd.component_lens_at_loc([1, 0, 'q', 4, 0, 'q', 1])


# %%
imshow(a)





# %%
f

# %%
of

# %%
f[6][9]

# %%
imshow(a)

# %%
import random

random.sample([2, 3, 4], k=2)







# %%
attn_freqs = task_eval.get_attn_head_freqs_over_dataset(N=N, subtract_counter_factuals=True, return_freqs=True)

# %%
af = attn_freqs * 30


# %%
ground_truth = IOI_GROUND_TRUTH_HEADS

# fp, tp, thresh = get_attn_head_roc(ground_truth, a.flatten().softmax(dim=-1), "IOI", visualize=True, additional_title="(No Counterfactuals)")
_ = get_attn_head_roc(ground_truth, attn_freqs.flatten().softmax(dim=-1), "GT", visualize=True, additional_title="(No Counterfactuals)")

# %%
thresh.shape


# %%
px.imshow(
    attn_freqs.flatten().softmax(dim=-1).reshape(12, 12),
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0
).show()

# %%
imshow(IOI_GROUND_TRUTH_HEADS)
# imshow(attn_freqs)



# %%
score, fp, tp, thresh = get_attn_head_roc(ground_truth, a.flatten().softmax(dim=-1), "GT", visualize=True, additional_title="(No Counterfactuals)")



# %%
fp, tp, *_ = metrics.roc_curve(IOI_GROUND_TRUTH_DATA, attn_freqs.flatten().softmax(dim=-1))

# %%
px.line(
    x=fp,
    y=tp,
).show()


# %%
attn_freqs.flatten().softmax(dim=-1).numpy()




# %%
task_eval.get_faithfulness_curve_over_data(N=10, attn_head_freq_n=10)


# %%
flattened_indices = torch.argsort(a.view(-1), descending=True)

# Step 3: Convert flattened indices back to multidimensional indices
cc = torch.stack(torch.unravel_index(flattened_indices, a.shape)).T

# %%
[0, 1] in cc.tolist()[:10]
# for i in multi_dim_indices.[:10]:
#     print(i)

# %%
model.run_with_cache(model.to_tokens('ey there'))[1]['z', 0].shape



# %%
model = task_eval.model

# %%

answer_tokens = []
for i in range(len(prompt_format)):
    for j in range(2):
        answers.append((names[i][j], names[i][1 - j]))
        answer_tokens.append(
            (
                model.to_single_token(answers[-1][0]),
                model.to_single_token(answers[-1][1]),
            )
        )
        # Insert the *incorrect* answer to the prompt, making the correct answer the indirect object.
        prompts.append(prompt_format[i].format(answers[-1][1]))
answer_tokens = torch.tensor(answer_tokens)









# %%
_ = task_eval.evalute_logit_diff_on_task(N=20, edge_based_graph_evaluation=False, include_all_heads=False, include_all_mlps=False)

# %%
(a.flatten() == 0).sum()

# %%
a




# %%


# %%
i = 21


task_eval.evaluate_circuit_discovery_for_prompt(i)

cd = task_eval.get_circuit_discovery_for_prompt(i)
# %%

cd.print_attn_heads_and_mlps_in_graph()

# %%
cd.visualize_attn_heads_in_graph()

# %%

cd.get_top_next_contributors()



# %%
cd.visualize_graph(begin_layer=6)
# %%
dataset_prompts[i]['text']





# %%
attn_heads,mlps = cd.get_heads_and_mlps_in_graph_at_seq()

mlps

# %%

# %%
task_eval.evalute_circuit_discovery_logit_diff(0)


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
cd.component_lens_at_loc_on_graph([0])


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
cd.component_lens_at_loc_on_graph(loc=[])

# %%
