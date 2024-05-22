import torch
import plotly.express as px
import random

from circuit_discovery import CircuitDiscovery, all_allowed
from circuit_lens import get_model_encoders
from typing import List, Callable, Set
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from data.eval_dataset import EvalItem
from tqdm import trange
from plotly_utils import *
from functools import partial


Z_FILTER = lambda x: x.endswith("z")


class FeatureCountForHeads:
    def __init__(self, features_for_heads_over_dataset: List[List[List[Set]]]):
        self.featuers_for_heads_over_dataset = features_for_heads_over_dataset

    def get_feature_counts_for_head_over_range(
        self, layer, head, N, num_samples=10, visualize=True
    ):
        counts = {0: 0.0}

        for i in trange(1, N):
            total = 0

            for _ in range(num_samples):
                sample_data = random.sample(self.featuers_for_heads_over_dataset, k=i)

                sample_set = set()

                for data_point in sample_data:
                    sample_set.update(data_point[layer][head])

                total += len(sample_set)

            counts[i] = total / num_samples

        if visualize:
            px.line(
                x=list(counts.keys()),
                y=list(counts.values()),
                labels={"x": "# IOI Sample Prompts", "y": "# Unique Features"},
                title=f"Unique features associated with L{layer}H{head}",
            ).show()

        return counts

    def get_feature_counts_for_layer_over_range(
        self, layer, N, num_samples=10, visualize=True
    ):
        counts = {0: 0.0}

        for i in trange(1, N):
            total = 0

            for _ in range(num_samples):
                sample_data = random.sample(self.featuers_for_heads_over_dataset, k=i)

                sample_set = set()

                for data_point in sample_data:
                    for head_set in data_point[layer]:
                        sample_set.update(head_set)

                total += len(sample_set)

            counts[i] = total / num_samples

        if visualize:
            px.line(
                x=list(counts.keys()),
                y=list(counts.values()),
                labels={"x": "# IOI Sample Prompts", "y": "# Unique Features"},
                title=f"Unique features associated with Layer {layer}",
            ).show()

        return counts

    def get_feature_set_for_head(self, layer, head):
        feature_set = set()

        for data_point in self.featuers_for_heads_over_dataset:
            feature_set.update(data_point[layer][head])

        return feature_set


class TaskEvaluation:
    """
    Evaluates `CircuitDiscovery` on a specific task
    """

    def __init__(
        self,
        prompts: List[EvalItem],
        circuit_discovery_strategy: Callable[[CircuitDiscovery], None],
        allowed_components_filter: Callable[[str], bool] = all_allowed,
        eval_index: int = -1,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model_encoders(self.device)[0]
        self.circuit_discovery_strategy = circuit_discovery_strategy
        self.prompts = prompts

        self.tokens = self.model.to_tokens(self.str_prompts)
        self.eval_index = eval_index
        self.allowed_components_filter = allowed_components_filter

        logits, cache = self.model.run_with_cache(self.tokens, return_type="logits")

        mean_cache = {}

        for key, value in cache.items():
            mean_cache[key] = value.mean(dim=0)

        self.mean_cache = ActivationCache(mean_cache, self.model)
        self.mean_logits = logits.mean(dim=0)

    def logit_difference(self, prompt_idx: int):
        correct_token = self.prompts[prompt_idx]["correct"]
        counter_token = self.prompts[prompt_idx]["counter"]

        if correct_token is None or counter_token is None:
            raise ValueError("Prompt does not have correct or counter token")

        toks = self.model.to_tokens([correct_token, counter_token]).squeeze()
        unembed = self.model.tokens_to_residual_directions(toks)

        return unembed[0] - unembed[1]

    @property
    def str_prompts(self):
        return [p["text"] for p in self.prompts]

    def get_circuit_discovery_for_prompt(
        self, prompt_idx: int, use_counter_factual=False, **kwargs
    ):
        if use_counter_factual:
            token = self.prompts[prompt_idx]["counter"]
        else:
            token = self.prompts[prompt_idx]["correct"]
        # prompt = (
        #     self.prompts[prompt_idx]["text"] + self.prompts[prompt_idx]["correct"]
        #     if not use_counter_factual
        #     else self.prompts[prompt_idx]["text"] + self.prompts[prompt_idx]["counter"]
        # )
        # print("prompt:", prompt)

        cd = CircuitDiscovery(
            self.prompts[prompt_idx]["text"],
            seq_index=self.eval_index,
            token=token,
            allowed_components_filter=self.allowed_components_filter,
        )

        self.circuit_discovery_strategy(cd)
        return cd

    def evaluate_circuit_discovery_for_prompt(self, prompt_idx: int, **kwargs):
        print(self.prompts[prompt_idx]["text"])

        cd = self.get_circuit_discovery_for_prompt(prompt_idx, **kwargs)
        cd.visualize_graph_performance_against_mean_ablation(self.mean_cache, **kwargs)

        if kwargs.get("return_none", False):
            return None

        return cd

    def evalute_circuit_discovery_logit_diff(
        self,
        prompt_idx: int,
        edge_based_graph_evaluation=False,
        visualize=True,
        **kwargs,
    ):
        prompt = self.prompts[prompt_idx]
        cd = self.get_circuit_discovery_for_prompt(prompt_idx, **kwargs)

        if prompt["correct"] is None or prompt["counter"] is None:
            raise ValueError("Prompt does not have correct or counter token")

        return cd.get_logit_difference_on_mean_ablation(
            prompt["correct"],
            prompt["counter"],
            self.mean_cache,
            self.mean_logits,
            edge_based_graph_evaluation=edge_based_graph_evaluation,
            visualize=visualize,
            **kwargs,
        )

    def evalute_logit_diff_on_task(
        self, N=None, edge_based_graph_evaluation=False, visualize=True, **kwargs
    ):
        if N is None:
            N = len(self.prompts)

        normalized = 0

        for i in trange(N):
            example_norm, *_ = self.evalute_circuit_discovery_logit_diff(
                i,
                visualize=False,
                edge_based_graph_evaluation=edge_based_graph_evaluation,
                **kwargs,
            )

            normalized += example_norm

        normalized /= N

        if visualize:
            print(f"Average normalized logit difference: {normalized*100:.3g}%")

        return normalized

    def get_attn_head_freqs_over_dataset(
        self,
        N=None,
        visualize=True,
        return_freqs=True,
        subtract_counter_factuals=False,
        additional_title="",
        **kwargs,
    ):
        if N is None:
            N = len(self.prompts)

        head_freqs = torch.zeros(12, 12)

        for i in trange(N):
            cd = self.get_circuit_discovery_for_prompt(i, **kwargs)
            head_freqs = head_freqs + cd.attn_heads_tensor()

            if subtract_counter_factuals:
                counter = self.get_circuit_discovery_for_prompt(
                    i, use_counter_factual=True, **kwargs
                )

                head_freqs = head_freqs - counter.attn_heads_tensor()

        head_freqs = head_freqs.float() / N

        if visualize:
            imshow(
                head_freqs,
                title="Attn Head Freqs for Strategy + Task " + additional_title,
                labels={"x": "Head", "y": "Layer"},
            )

        if return_freqs:
            return head_freqs

    def get_features_at_heads_over_dataset(self, N=None):
        if N is None:
            N = len(self.prompts)

        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads

        features_for_heads = [[set() for _ in range(n_heads)] for _ in range(n_layers)]

        for i in trange(N):
            cd = self.get_circuit_discovery_for_prompt(i)

            prompt_features_for_heads = cd.get_features_at_heads_in_graph()

            for layer in range(n_layers):
                for head in range(n_heads):
                    features_for_heads[layer][head].update(
                        prompt_features_for_heads[layer][head]
                    )

        return features_for_heads

    def get_feature_count_for_heads_over_dataset(self, N=None):
        if N is None:
            N = len(self.prompts)

        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads

        data = []

        for i in trange(N):
            cd = self.get_circuit_discovery_for_prompt(i)

            data.append(cd.get_features_at_heads_in_graph())

        return FeatureCountForHeads(data)

    def get_faithfulness_curve_over_data(
        self, eval_n, attn_head_freq_n, faithfulness_intervals: int = 10, visualize=True
    ):
        answer_tokens = []
        for i in range(eval_n):
            answer_tokens.append(
                (
                    self.model.to_single_token(self.prompts[i]["correct"]),
                    self.model.to_single_token(self.prompts[i]["counter"]),
                )
            )

        answer_tokens = torch.tensor(answer_tokens).to(self.device)

        def logits_to_ave_diff(logits, answer_tokens=answer_tokens):
            logits = logits[:, self.eval_index, :]
            print(logits.shape, answer_tokens.shape)
            answer_logits = logits.gather(dim=-1, index=answer_tokens)
            answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]

            return answer_logit_diff.mean()

        attn_freqs = self.get_attn_head_freqs_over_dataset(
            return_freqs=True, N=attn_head_freq_n
        )
        assert attn_freqs is not None

        flattened_indices = torch.argsort(attn_freqs.view(-1), descending=True)

        # Step 3: Convert flattened indices back to multidimensional indices
        heads_by_importance = torch.stack(
            torch.unravel_index(flattened_indices, attn_freqs.shape)
        ).T.tolist()

        heads_per_faithfulness_interval = (
            len(heads_by_importance) // faithfulness_intervals
        )

        def z_ablate_all(acts, hook: HookPoint):
            return torch.zeros_like(acts)

        tokens = self.model.to_tokens(self.str_prompts[:eval_n])

        corrupted_logits = self.model.run_with_hooks(
            tokens, fwd_hooks=[(Z_FILTER, z_ablate_all)], return_type="logits"
        )
        base_logits = self.model(tokens, return_type="logits")

        corrupted_diff = logits_to_ave_diff(corrupted_logits)
        base_diff = logits_to_ave_diff(base_logits)

        print("base cor", base_diff, corrupted_diff)

        def logits_to_normalized_diff(logits):
            return (logits_to_ave_diff(logits) - corrupted_diff) / (
                base_diff - corrupted_diff
            )

        faithfulness_curve = {}

        def z_ablate_hook(acts, hook: HookPoint, heads_to_ablate_by_layer):
            heads_to_ablate = heads_to_ablate_by_layer[hook.layer()]

            acts[:, :, heads_to_ablate, :] = 0

            return acts

        for i in range(1, faithfulness_intervals + 1):
            num_heads = i * heads_per_faithfulness_interval
            heads = heads_by_importance[:num_heads]

            # print("heasd", heads)
            # raise ValueError("yeah")

            heads_to_ablate_by_layer = [[] for _ in range(self.model.cfg.n_layers)]

            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_heads):
                    if [layer, head] not in heads:
                        heads_to_ablate_by_layer[layer].append(head)

            hook = partial(
                z_ablate_hook, heads_to_ablate_by_layer=heads_to_ablate_by_layer
            )

            logits_for_heads = self.model.run_with_hooks(
                tokens, fwd_hooks=[(Z_FILTER, hook)], return_type="logits"
            )

            faithfulness_curve[len(heads)] = logits_to_normalized_diff(
                logits_for_heads
            ).item()

            if num_heads == len(heads_by_importance):
                break

        if visualize:
            px.line(
                x=list(faithfulness_curve.keys()),
                y=list(faithfulness_curve.values()),
                title="Faithfulness Curve",
                labels={"x": "Num Heads", "y": "Normalized Logit Difference"},
            ).show()

        return faithfulness_curve
