import torch

from circuit_discovery import CircuitDiscovery, all_allowed
from circuit_lens import get_model_encoders
from typing import List, Callable
from transformer_lens import ActivationCache
from data.eval_dataset import EvalItem
from tqdm import trange


class TaskEvaluation:
    """
    Evaluates `CircuitDiscovery` on a specific task
    """

    def __init__(
        self,
        prompts: List[EvalItem],
        eval_index: int,
        circuit_discovery_strategy: Callable[[CircuitDiscovery], None],
        allowed_components_filter: Callable[[str], bool] = all_allowed,
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

    def get_circuit_discovery_for_prompt(self, prompt_idx: int):
        cd = CircuitDiscovery(
            self.prompts[prompt_idx]["text"],
            -2,
            allowed_components_filter=self.allowed_components_filter,
        )

        self.circuit_discovery_strategy(cd)
        return cd

    def evaluate_circuit_discovery_for_prompt(self, prompt_idx: int, **kwargs):
        cd = self.get_circuit_discovery_for_prompt(prompt_idx)
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
        cd = self.get_circuit_discovery_for_prompt(prompt_idx)

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
