import yaml
import torch
import numpy as np
import plotly.express as px
import einops
import html
from IPython.display import HTML, display
from sklearn import metrics
from tqdm import trange, tqdm
from z_sae import ZSAE
from mlp_transcoder import SparseTranscoder
from openai import AzureOpenAI
from transformer_lens.utils import to_numpy
from collections import defaultdict, Counter

from datasets import load_dataset
from huggingface_hub import HfApi
from transformer_lens import HookedTransformer, utils
from circuit_lens import get_model_encoders, CircuitComponent
from task_evaluation import TaskEvaluation
from data.ioi_dataset import gen_templated_prompts, IOI_GROUND_TRUTH_HEADS
from data.greater_than_dataset import generate_greater_than_dataset, GT_GROUND_TRUTH_HEADS
from circuit_discovery import CircuitDiscovery
from plotly_utils import *
from autointerp_prompts import get_opening_prompt

from termcolor import colored
from tabulate import tabulate


class CircuitPrediction:
    def __init__(self, attn_freqs, mlp_freqs, features_for_heads, features_for_mlps, co_occurrence_dict):
        self.attn_freqs = attn_freqs
        self.mlp_freqs = mlp_freqs
        self.features_for_heads = features_for_heads
        self.features_for_mlps = features_for_mlps
        self.component_labels = self.get_component_labels()
        self.circuit_hypergraph = self.create_circuit_hypergraph()

        self.co_occurrence_dict = self.symmetric_co_occurrence_dict(self.clean_co_occurrence_dict(co_occurrence_dict))

    def clean_co_occurrence_dict(self, co_occurrence_dict):
        cleaned_dict = {}

        for component, co_occurrences in co_occurrence_dict.items():
            cleaned_sub_dict = {}

            for other_component, feature_tuples in co_occurrences.items():
                filtered_tuples = [t for t in feature_tuples if -1 not in t]

                if filtered_tuples:  # only add if there are remaining tuples
                    cleaned_sub_dict[other_component] = filtered_tuples

            if cleaned_sub_dict: 
                cleaned_dict[component] = cleaned_sub_dict

        return cleaned_dict
    
    def symmetric_co_occurrence_dict(self, co_occurrence_dict):
        symmetric_dict = {}

        for component, co_occurrences in co_occurrence_dict.items():
            for other_component, feature_tuples in co_occurrences.items():
                set_key = (component, other_component)

                reversed_feature_tuples = [(t[1], t[0]) for t in feature_tuples]

                if other_component in co_occurrence_dict and component in co_occurrence_dict[other_component]:
                    assert co_occurrence_dict[other_component][component] == reversed_feature_tuples, "Symmetric lists do not match!"

                if set_key not in symmetric_dict and (set_key[1], set_key[0]) not in symmetric_dict:
                    symmetric_dict[set_key] = feature_tuples

        return symmetric_dict
    
    def parse_component(self, component_str):
        """
        Parse the component string to return the appropriate tuple key.
        """
        if component_str.startswith("MLP"):
            layer = int(component_str[3:])
            return ('mlp_feature', layer)
        elif component_str.startswith("L") and "H" in component_str:
            layer, head = map(int, component_str[1:].split("H"))
            return ('attn_head', layer, head)
        else:
            raise ValueError(f"Invalid component format: {component_str}")

    def get_cooccurrences(self, component1_str, component2_str):
        """
        Retrieve co-occurrences between two components given their string representations.
        Ensures symmetry in retrieval.
        """
        component1 = self.parse_component(component1_str)
        component2 = self.parse_component(component2_str)
        
        # Create a sorted tuple to ensure symmetry
        key = tuple(sorted((component1, component2)))
        
        if key in self.co_occurrence_dict:
            return self.co_occurrence_dict[key], key
        elif (key[1], key[0]) in self.co_occurrence_dict:
            return self.co_occurrence_dict[(key[1], key[0])], (key[1], key[0])
        else:
            return [], None
    
    def display_co_occurrences(self):
        for layer, data in self.co_occurrence_dict.items():
            print(f"Layer {layer}:")
            
            # Display heads
            print("  Heads:")
            for head_data in data['heads']:
                for (layer, head), info in head_data.items():
                    print(f"    Head {head}: Count = {info['count']}, Features = {info['features']}")
            
            # Display mlps
            mlp_info = data['mlps']
            print(f"  MLP: Count = {mlp_info['count']}, Features = {mlp_info['features']}")

    def create_circuit_hypergraph(self):
        circuit_hypergraph = {label: {"freq": 0, "features": []} for label in self.component_labels}
        for layer, freq in enumerate(self.attn_freqs):
            for head, freq_head in enumerate(freq):
                label = f"L{layer}_H{head}"
                circuit_hypergraph[label]["freq"] += freq_head.item()
                circuit_hypergraph[label]["features"].extend(self.features_for_heads[layer][head])
        for i, freq in enumerate(self.mlp_freqs):
            label = f"MLP{i}"
            circuit_hypergraph[label]["freq"] += freq.item()
            circuit_hypergraph[label]["features"].extend(self.features_for_mlps[i])
        return circuit_hypergraph
            
    def get_component_labels(self):
        head_labels = [f"L{layer}_H{head}" for layer in range(12) for head in range(12)]
        mlp_labels = [f"MLP{i}" for i in range(12)]
        labels = []
        for i in range(12):
            labels.extend(head_labels[i*12:(i+1)*12])
            labels.append(mlp_labels[i])
        return labels
    
    def get_all_features_from_attn_layer(self, layer: int):
        features = []
        for head in range(12):
            features.extend(self.features_for_heads[layer][head])
        return features

    def get_circuit_at_threshold(self, threshold: float, visualize: bool = False):
        circuit = np.zeros(len(self.component_labels))
        for i, label in enumerate(self.component_labels):
            if self.circuit_hypergraph[label]["freq"] > threshold:
                circuit[i] = 1
        if visualize:
            self.visualize_circuit(circuit)
        return circuit
    
    def visualize_circuit(self, circuit: np.ndarray, additional_title=""):
        circuit_array = np.zeros((12, 13))
        labels = [f"A{i}" for i in range(1, 13)] + ["MLP"]
        for i, pred in enumerate(circuit):
            layer = i // 13
            head = i % 13
            circuit_array[layer, head] = pred
        fig = px.imshow(circuit_array, labels=dict(x="Attention Head", y="Layer"), width=500,
                        title=additional_title, x=labels, y=list(range(12)), color_continuous_scale="blues")
        fig.show()

    def component_frequency_array(self, visualize: bool = False):
        frequency_array = np.zeros((12, 13))
        labels = [f"A{i}" for i in range(1, 13)] + ["MLP"]
        for i, label in enumerate(self.component_labels):
            layer = i // 13
            head = i % 13
            frequency_array[layer, head] = self.circuit_hypergraph[label]["freq"]
        if visualize:
            fig = px.imshow(frequency_array, labels=dict(x="Attention Head", y="Layer"), width=450,
                        title="Frequency of components", x=labels, y=list(range(12)), color_continuous_scale="blues")
            fig.show()
        return frequency_array

    def unique_feature_array(self, visualize: bool = False):
        unique_features_array = np.zeros((12, 13))
        labels = [f"A{i}" for i in range(1, 13)] + ["MLP"]
        for i, label in enumerate(self.component_labels):
            layer = i // 13
            head = i % 13
            unique_features_array[layer, head] = len(set(self.circuit_hypergraph[label]["features"]))
        if visualize:
            fig = px.imshow(unique_features_array, labels=dict(x="Attention Head", y="Layer"), width=450,
                        title="Unique features", x=labels, y=list(range(12)), color_continuous_scale="blues")
            fig.show()
        return unique_features_array
    
    def get_top_k_feature_tuples(self, k=5):
        # Use a Counter to count the occurrences of each tuple
        global_counter = Counter()

        # Iterate through each component set and their feature lists
        for component_set, feature_list in self.co_occurrence_dict.items():
            for feature_tuple in feature_list:
                # Create a sorted tuple to ensure (x, y) and (y, x) are treated the same
                sorted_tuple = tuple(sorted(feature_tuple))
                global_counter[(component_set, sorted_tuple)] += 1

        # Get the top-k tuples by count
        top_k_tuples = global_counter.most_common(k)
        
        # Create a dictionary to store the results
        top_k_dict = defaultdict(dict)
        
        for (component_set, feature_tuple), count in top_k_tuples:
            top_k_dict[component_set][feature_tuple] = count

        return top_k_dict
    
    def visualize_co_occurrences(self):
        num_layers = 12
        num_heads = 12
        num_components = num_layers * (num_heads + 1)
        
        # Create a mapping from components to indices
        component_to_index = {}
        index = 0
        for layer in range(num_layers):
            for head in range(num_heads):
                component_to_index[('attn_head', layer, head)] = index
                index += 1
            component_to_index[('mlp_feature', layer)] = index
            index += 1

        # Initialize the co-occurrence matrix
        co_occurrence_matrix = np.zeros((num_components, num_components))

        # Populate the co-occurrence matrix
        for comp_pair, co_occurrences in self.co_occurrence_dict.items():
            comp1, comp2 = list(comp_pair)
            index1 = component_to_index[comp1]
            index2 = component_to_index[comp2]
            co_occurrence_matrix[index1, index2] = len(co_occurrences)
            co_occurrence_matrix[index2, index1] = len(co_occurrences)

        # Create the labels for the heatmap
        labels = []
        for layer in range(num_layers):
            for head in range(num_heads):
                labels.append(f"L{layer}H{head}")
            labels.append(f"MLP{layer}")

        # Plot the heatmap using Plotly
        fig = px.imshow(
            co_occurrence_matrix,
            labels=dict(x="Component", y="Component", color="Co-occurrence Count"),
            x=labels,
            y=labels,
            color_continuous_scale="blues",
            title="Component Co-occurrences Heatmap"
        )
        
        fig.update_layout(width=800, height=800)
        fig.show()


def tokenize_and_concatenate(dataset, tokenizer, streaming=False, max_length=1024, column_name="text", add_bos_token=True):
    for key in dataset.features:
        if key != column_name:
            dataset = dataset.remove_columns(key)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    seq_len = max_length - 1 if add_bos_token else max_length
    def tokenize_function(examples):
        text = examples[column_name]
        full_text = tokenizer.eos_token.join(text)
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
        tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)
        num_batches = num_tokens // seq_len
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len)
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=[column_name])
    return tokenized_dataset


def get_feature_scores(model, encoder, tokens_arr, feature_indices, batch_size=64, act_name=None, 
                       use_raw_scores=False, use_decoder=False, feature_post=None, ignore_endoftext=False):
    
    # Determine the type of encoder and set defaults
    if isinstance(encoder, ZSAE):
        print("ZSAE")
        act_name = act_name or 'attn.hook_z'
        layer = encoder.cfg['layer']
        name_filter = f'blocks.{layer}.attn.hook_z'
    elif isinstance(encoder, SparseTranscoder):
        print("SparseTranscoder")
        act_name = act_name or encoder.cfg.hook_point
        layer = encoder.cfg.hook_point_layer
        name_filter = act_name
    else:
        raise ValueError("Unsupported encoder type")

    scores = []
    endoftext_token = model.tokenizer.eos_token

    for i in tqdm(range(0, tokens_arr.shape[0], batch_size)):
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens_arr[i:i+batch_size], stop_at_layer=layer+1, names_filter=name_filter)
            mlp_acts = cache[name_filter]
            mlp_acts_flattened = mlp_acts.reshape(-1, encoder.W_enc.shape[0])
            
            if feature_post is None:
                if isinstance(encoder, SparseTranscoder) and use_decoder:
                    feature_post = encoder.W_dec[:, feature_indices]
                else:
                    feature_post = encoder.W_enc[:, feature_indices]
                    
            if isinstance(encoder, SparseTranscoder) and use_decoder:
                bias = -(encoder.b_dec @ feature_post)
            else:
                bias = encoder.b_enc[feature_indices] - (encoder.b_dec @ feature_post)
            
            if use_raw_scores:
                cur_scores = (mlp_acts_flattened @ feature_post) + bias
            else:
                hidden_acts = encoder.encode(mlp_acts_flattened)
                cur_scores = hidden_acts[:, feature_indices]
                del hidden_acts
            
            if ignore_endoftext:
                cur_scores[tokens_arr[i:i+batch_size].reshape(-1) == endoftext_token] = -torch.inf

        scores.append(
            to_numpy(
                einops.rearrange(cur_scores, "(b pos) n -> b n pos", pos=tokens_arr.shape[1])
            ).astype(np.float16)
        )

    return np.concatenate(scores, axis=0)

def get_feature_scores_across_layers(model, encoder_feature_pairs, tokens_arr, act_name=None, batch_size=64):
    combined_scores = []
    for encoder, feature_indices in encoder_feature_pairs:
        scores = get_feature_scores(model, encoder, tokens_arr, feature_indices, act_name=act_name, batch_size=batch_size)
        combined_scores.append(scores)

    # We want to combine the scores along the second dimension (features), as feature scores is shape (batch, features, tokens)
    return np.stack(combined_scores, axis=1).squeeze()


def get_top_k_activating_examples(feature_scores, tokens, model, k=5):
    flat_scores = feature_scores.flatten()
    top_k_indices = flat_scores.argsort()[-k:][::-1]
    top_k_scores = flat_scores[top_k_indices]
    top_k_batch_indices, top_k_seq_indices = np.unravel_index(top_k_indices, feature_scores.shape)
    top_k_tokens = [tokens[batch_idx].tolist() for batch_idx in top_k_batch_indices]
    top_k_tokens_str = [[model.to_string(x) for x in token_seq] for token_seq in top_k_tokens]
    top_k_scores_per_seq = [feature_scores[batch_idx].tolist() for batch_idx in top_k_batch_indices]
    return top_k_tokens_str, top_k_scores_per_seq, top_k_seq_indices


def highlight_scores_in_html(token_strs, scores, seq_idx, max_color='#ff8c00', zero_color='#ffffff', show_score=True):
    if len(token_strs) != len(scores):
        print("Length mismatch between tokens and scores")
        return "", ""
    scores_min = min(scores)
    scores_max = max(scores)
    scores_normalized = (np.array(scores) - scores_min) / (scores_max - scores_min)
    max_color_vec = np.array([int(max_color[1:3], 16), int(max_color[3:5], 16), int(max_color[5:7], 16)])
    zero_color_vec = np.array([int(zero_color[1:3], 16), int(zero_color[3:5], 16), int(zero_color[5:7], 16)])
    color_vecs = np.einsum('i, j -> ij', scores_normalized, max_color_vec) + np.einsum('i, j -> ij', 1 - scores_normalized, zero_color_vec)
    color_strs = [f"#{int(x[0]):02x}{int(x[1]):02x}{int(x[2]):02x}" for x in color_vecs]
    if show_score:
        tokens_html = "".join([
            f"""<span class='token' style='background-color: {color_strs[i]}'>{html.escape(token_str)}<span class='feature_val'> ({scores[i]:.2f})</span></span>"""
            for i, token_str in enumerate(token_strs)
        ])
        clean_text = " | ".join([
            f"{token_str} ({scores[i]:.2f})"
            for i, token_str in enumerate(token_strs)
        ])
    else:
        tokens_html = "".join([
            f"""<span class='token' style='background-color: {color_strs[i]}'>{html.escape(token_str)}</span>"""
            for i, token_str in enumerate(token_strs)
        ])
        clean_text = " | ".join(token_strs)
    head = """
    <style>
        span.token {
            font-family: monospace;
            border-style: solid;
            border-width: 1px;
            border-color: #dddddd;
        }
    </style>
    """
    return head + tokens_html, clean_text


def display_top_k_activating_examples(model, feature_scores, tokens, k=5, show_score=True, display_html=True):
    top_k_tokens_str, top_k_scores_per_seq, top_k_seq_indices = get_top_k_activating_examples(feature_scores, tokens, model, k=k)
    examples_html = []
    examples_clean_text = []
    for i in range(k):
        example_html, clean_text = highlight_scores_in_html(top_k_tokens_str[i], top_k_scores_per_seq[i], top_k_seq_indices[i], show_score=show_score)
        if display_html:
            display(HTML(example_html))
        examples_html.append(example_html)
        examples_clean_text.append(clean_text)
    return examples_html, examples_clean_text


def display_top_k_activating_examples_sum(model, feature_scores, tokens, feature_indices, k=5, show_score=True, display_html=True):
    top_k_tokens_str, top_k_scores_per_seq, top_k_seq_indices = get_top_k_activating_examples_sum(feature_scores, tokens, model, feature_indices, k=k)
    examples_html = []
    examples_clean_text = []
    for i in range(k):
        example_html, clean_text = highlight_scores_in_html(top_k_tokens_str[i], top_k_scores_per_seq[i], top_k_seq_indices[i], show_score=show_score)
        if display_html:
            display(HTML(example_html))
        examples_html.append(example_html)
        examples_clean_text.append(clean_text)
    return examples_html, examples_clean_text

def get_top_k_activating_examples_sum(feature_scores, tokens, model, feature_indices, k=5):
    feature_scores_summed = np.sum(feature_scores, axis=1)
    # Sum the activations for the specified features
    summed_scores_initial = np.sum(feature_scores[:, feature_indices, :], axis=-1)
    summed_scores = np.sum(summed_scores_initial, axis=-1)
    # Add extra dim at end
    summed_scores = summed_scores[:, np.newaxis]
    
    flat_scores = summed_scores.flatten()
    top_k_indices = flat_scores.argsort()[-k:][::-1]
    top_k_scores = flat_scores[top_k_indices]
    
    top_k_batch_indices, top_k_seq_indices = np.unravel_index(top_k_indices, summed_scores.shape)
    top_k_tokens = [tokens[batch_idx].tolist() for batch_idx in top_k_batch_indices]
    top_k_tokens_str = [[model.to_string(x) for x in token_seq] for token_seq in top_k_tokens]
    top_k_scores_per_seq = [feature_scores_summed[batch_idx].tolist() for batch_idx in top_k_batch_indices]

    return top_k_tokens_str, top_k_scores_per_seq, top_k_seq_indices


# def get_top_logits(model, encoder, features, act_strength=4.0, dict_size=24576):
#     hidden_acts = torch.zeros(dict_size, device='cpu')
#     if isinstance(features, list):
#         for feature in features:
#             hidden_acts[feature] = act_strength
#     else:
#         hidden_acts[features] = act_strength  # Single feature case

#     hidden_acts = hidden_acts.unsqueeze(0)
#     hidden_acts = encoder.decode(hidden_acts)
#     logits = einops.einsum(
#         hidden_acts.to('cpu'), model.W_U.to('cpu'),
#         'b h, h l -> b l'
#     )
#     return logits

# def get_top_k_tokens(model, encoder, features, dict_size=24576, act_strength=4.0, k=10):
#     logits = get_top_logits(model, encoder, features, act_strength, dict_size)
#     top_k = torch.topk(logits, k)
#     top_k_indices = top_k.indices.squeeze().tolist()
#     top_k_logits = top_k.values.squeeze().tolist()
#     top_k_tokens = [model.to_string(x) for x in top_k_indices]
#     return top_k_tokens, top_k_logits

def get_top_logits(model, encoder, features, act_strength=4.0, dict_size=24576):
    hidden_acts = torch.zeros(dict_size, device='cpu')
    if isinstance(features, list):
        for feature in features:
            hidden_acts[feature] = act_strength
    else:
        hidden_acts[features] = act_strength  # Single feature case

    hidden_acts = hidden_acts.unsqueeze(0)
    hidden_acts = encoder.decode(hidden_acts)
    logits = einops.einsum(
        hidden_acts.to('cpu'), model.W_U.to('cpu'),
        'b h, h l -> b l'
    )
    return logits

def get_combined_logits(model, encoder_feature_pairs, act_strength=4.0, dict_size=24576):
    combined_logits = torch.zeros((1, model.W_U.size(1)), device='cpu')
    for encoder, features in encoder_feature_pairs:
        logits = get_top_logits(model, encoder, features, act_strength, dict_size)
        combined_logits += logits
    return combined_logits

def get_top_k_tokens(model, encoder_feature_pairs, dict_size=24576, act_strength=4.0, k=10):
    combined_logits = get_combined_logits(model, encoder_feature_pairs, act_strength, dict_size)
    top_k = torch.topk(combined_logits, k)
    top_k_indices = top_k.indices.squeeze().tolist()
    top_k_logits = top_k.values.squeeze().tolist()
    top_k_tokens = [model.to_string(x) for x in top_k_indices]
    return top_k_tokens, top_k_logits

def pretty_print_tokens_logits(tokens, logits):
    table = [["Token", "Logit"]]
    for token, logit in zip(tokens, logits):
        token_str = colored(token, 'blue')
        logit_str = colored(f"{logit:.4f}", 'green')
        table.append([token_str, logit_str])
    
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


def get_response(llm_client, examples_clean_text, top_tokens):
    opening_prompt = get_opening_prompt(examples_clean_text, top_tokens)
    messages = [{"role": "user", "content": opening_prompt}]
    response = llm_client.chat.completions.create(
        model="gpt4_large",
        messages=messages,
    )
    return f"{response.choices[0].message.content}"

def get_circuit_prediction(task: str = 'ioi', N: int = 50):
    torch.set_grad_enabled(False)
    dataset_prompts = gen_templated_prompts(template_idex=1, N=500)

    def component_filter(component: str):
        return component in [
            CircuitComponent.Z_FEATURE,
            CircuitComponent.MLP_FEATURE,
            CircuitComponent.ATTN_HEAD,
            CircuitComponent.UNEMBED,
            CircuitComponent.EMBED,
            CircuitComponent.POS_EMBED,
            CircuitComponent.Z_SAE_ERROR,
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
    thres = 4

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

    # features_for_heads = task_eval.get_features_at_heads_over_dataset(N=N, use_set=False)
    # features_for_mlps = task_eval.get_features_at_mlps_over_dataset(N=N, use_set=False)
    # mlp_freqs = task_eval.get_mlp_freqs_over_dataset(N=N, return_freqs=True, visualize=False)
    # attn_freqs = task_eval.get_attn_head_freqs_over_dataset(N=N, subtract_counter_factuals=False, return_freqs=True, visualize=False)

    results = task_eval.process_all_metrics_over_dataset(N=N, visualize=False, subtract_counter_factuals=False, use_set=False)
    features_for_heads = results["features_for_heads"]
    features_for_mlps = results["features_for_mlps"]
    mlp_freqs = results["mlp_freqs"]
    attn_freqs = results["attn_head_freqs"]
    co_occurrence_dict = results["co_occurrence_dict"]

    # # Create CircuitDiscovery object
    # cd = task_eval.get_circuit_discovery_for_prompt(20)
    # cd.build_directed_graph(contributors_per_node=1)

    # Create CircuitPrediction object
    cp = CircuitPrediction(attn_freqs, mlp_freqs, features_for_heads, features_for_mlps, co_occurrence_dict)

    return cp
 

# def get_circuit_prediction(task: str = 'ioi', N: int = 50):
#     torch.set_grad_enabled(False)
#     dataset_prompts = gen_templated_prompts(template_idex=1, N=500)
#     def component_filter(component: str):
#         return component in [
#             CircuitComponent.Z_FEATURE,
#             CircuitComponent.MLP_FEATURE,
#             CircuitComponent.ATTN_HEAD,
#             CircuitComponent.UNEMBED,
#             CircuitComponent.EMBED,
#             CircuitComponent.POS_EMBED,
#             CircuitComponent.Z_SAE_ERROR,
#         ]
#     pass_based = True
#     passes = 5
#     node_contributors = 1
#     first_pass_minimal = True
#     sub_passes = 3
#     do_sub_pass = False
#     layer_thres = 9
#     minimal = True
#     num_greedy_passes = 20
#     k = 1
#     thres = 4
#     def strategy(cd: CircuitDiscovery):
#         if pass_based:
#             for _ in range(passes):
#                 cd.add_greedy_pass(contributors_per_node=node_contributors, minimal=first_pass_minimal)
#                 if do_sub_pass:
#                     for _ in range(sub_passes):
#                         cd.add_greedy_pass_against_all_existing_nodes(contributors_per_node=node_contributors, skip_z_features=True, layer_threshold=layer_thres, minimal=minimal)
#         else:
#             for _ in range(num_greedy_passes):
#                 cd.greedily_add_top_contributors(k=k, reciever_threshold=thres)
#     task_eval = TaskEvaluation(prompts=dataset_prompts, circuit_discovery_strategy=strategy, allowed_components_filter=component_filter)
#     # cd = task_eval.get_circuit_discovery_for_prompt(20)
#     features_for_heads = task_eval.get_features_at_heads_over_dataset(N=N, use_set=False)
#     features_for_mlps = task_eval.get_features_at_mlps_over_dataset(N=N, use_set=False)
#     mlp_freqs = task_eval.get_mlp_freqs_over_dataset(N=N, return_freqs=True, visualize=False)
#     attn_freqs = task_eval.get_attn_head_freqs_over_dataset(N=N, subtract_counter_factuals=False, return_freqs=True, visualize=False)
#     cp = CircuitPrediction(attn_freqs, mlp_freqs, features_for_heads, features_for_mlps)
#     return cp
    
