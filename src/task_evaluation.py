import torch

from circuit_discovery import CircuitDiscovery, all_allowed
from circuit_lens import get_model_encoders
from typing import List, Callable, Tuple
from transformer_lens import ActivationCache
from data.eval_dataset import EvalItem
from tqdm import trange
from sklearn.metrics import roc_auc_score, f1_score


class PredictionEvaluation:
    """ 
    Specific metrics for binary classification.
    """

    def __init__(
            self,
            y_true,
            y_pred,
            y_scores
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores

    def accuracy(self):
        return (self.y_true == self.y_pred).float().mean().item()
    
    def precision(self):
        true_positives = (self.y_true & self.y_pred).sum()
        false_positives = (~self.y_true & self.y_pred).sum()
        return true_positives / (true_positives + false_positives)
    
    def recall(self):
        true_positives = (self.y_true & self.y_pred).sum()
        false_negatives = (self.y_true & ~self.y_pred).sum()
        return true_positives / (true_positives + false_negatives)
    
    def f1_score(self):
        return f1_score(self.y_true, self.y_pred)
    
    def confusion_matrix(self):
        true_positives = (self.y_true & self.y_pred).sum()
        false_positives = (~self.y_true & self.y_pred).sum()
        true_negatives = (~self.y_true & ~self.y_pred).sum()
        false_negatives = (self.y_true & ~self.y_pred).sum()
        return true_positives, false_positives, true_negatives, false_negatives
    
    def roc_auc(self):
        return roc_auc_score(self.y_true, self.y_scores) if self.y_scores is not None else None


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
        # Ground truth is a list of tuples
        ground_truth: List[Tuple[str, str]] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model_encoders(self.device)[0]
        self.circuit_discovery_strategy = circuit_discovery_strategy
        self.prompts = prompts
        self.ground_truth = ground_truth

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
    
    def layer_head_to_index(self, layer: int, head: int):
        # Note: indexing starts at 0 for both
        return layer * 12 + head
    
    @property
    def ground_truth_vector(self):
        if self.ground_truth is None:
            return None

        gt = torch.zeros(self.model.cfg.n_heads * self.model.cfg.n_layers, device=self.device)

        for layer, head in self.ground_truth:
            gt[self.layer_head_to_index(layer, head)] = 1

        return gt

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

    def evaluate_circuit_discovery_logit_diff(
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

    def evaluate_logit_diff_on_task(
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
    
    def classification_metrics(self, cd: CircuitDiscovery):
        heads, _ = cd.get_heads_and_mlps_in_graph()
        # Convert list of lists of heads to vector of 144 and 0 if head is not present, 1 if head is present
        heads_vector = torch.zeros(144)
        for layer, head_list in enumerate(heads):
            for head in head_list:
                heads_vector[self.layer_head_to_index(layer, head)] = 1

        # Convert ground truth
        ground_truth_vector = self.ground_truth_vector

        pred_eval = PredictionEvaluation(
            ground_truth_vector,
            heads_vector,
            None,
        )

        # Get a dictionary of the main metrics
        metrics = {
            "accuracy": pred_eval.accuracy(),
            "f1_score": pred_eval.f1_score(),
            "roc_auc": pred_eval.roc_auc(),
        }

        return metrics
