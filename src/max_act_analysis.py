import torch
import einops
import circuitsvis as cv

from open_web_text import open_web_text_tokens
from tqdm import trange
from typing import List
from IPython.display import display
from discovery_strategies import BASIC_FILTER, BASIC_STRATEGY
from circuit_discovery import CircuitDiscovery
from circuit_lens import get_model_encoders
from aug_interp_prompts import main_aug_interp_prompt, extract_explanation
from openai_utils import gen_openai_completion


class MaxActAnalysis:
    num_sequences: int

    def __init__(
        self,
        feature_type: str,
        layer: int,
        feature: int,
        num_sequences=None,
        batch_size=128,
        component_filter=BASIC_FILTER,
        strategy=BASIC_STRATEGY,
    ):
        assert feature_type in ("attn", "mlp")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_type = feature_type

        model, z_saes, transcoders = get_model_encoders(self.device)

        self.model = model
        self.feature = feature
        self.layer = layer
        self.batch_size = batch_size
        self.component_filter = component_filter
        self.strategy = strategy

        if num_sequences is None:
            self.num_sequences = open_web_text_tokens.size(0)
        else:
            self.num_sequences = num_sequences

        if self.feature_type == "attn":
            self.encoder = z_saes[layer]
        else:
            self.encoder = transcoders[layer]

        self._context_referenced_prompt_cache = dict()

    _active_examples = None

    def update_filter_and_strategy(self, component_filter=None, strategy=None):
        if component_filter is not None:
            self.component_filter = component_filter

        if strategy is not None:
            self.strategy = strategy

    def get_circuit_discovery_for_max_activating_example(
        self, example_i: int, strategy=None, comp_filter=None
    ):
        if strategy is None:
            strategy = self.strategy

        if comp_filter is None:
            comp_filter = self.component_filter

        seq, pos, _ = self.get_active_example(example_i)

        tokens = open_web_text_tokens[seq]
        prompt = self.model.tokenizer.decode(tokens[1:])

        cd = CircuitDiscovery(
            prompt, seq_index=pos, allowed_components_filter=comp_filter
        )

        if self.feature_type == "attn":
            cd.set_root_to_z_feature(
                layer=self.layer, seq_index=pos, feature=self.feature
            )
        else:
            cd.set_root_to_mlp_feature(
                layer=self.layer, seq_index=pos, feature=self.feature
            )
        strategy(cd)

        return cd

    def get_context_referenced_prompts_for_range(
        self,
        start,
        end,
        token_lr=("<<", ">>"),
        context_lr=("[[", "]]"),
        merge_nearby_context=False,
    ):
        if (start, end) in self._context_referenced_prompt_cache:
            return self._context_referenced_prompt_cache[(start, end)]

        prompts = []

        # This is sketchy, but we include this here to not have nested tranges
        _ = self.active_examples

        for i in trange(start, end):
            p, ap = self.get_context_referenced_prompt(
                i,
                token_lr=token_lr,
                context_lr=context_lr,
                merge_nearby_context=merge_nearby_context,
            )

            prompts.append((p, ap))

        self._context_referenced_prompt_cache[(start, end)] = prompts

        return prompts

    def get_feature_auto_interp(
        self,
        max_act_start,
        max_act_end,
        token_lr=("<<", ">>"),
        context_lr=("[[", "]]"),
        merge_nearby_context=False,
        top_tokens_k=5,
        visualize=True,
        replace_neuron_with_feature=True,
    ):
        context_referenced_prompts = self.get_context_referenced_prompts_for_range(
            max_act_start, max_act_end, token_lr, context_lr, merge_nearby_context
        )

        top_tokens = self.get_top_k_tokens(k=top_tokens_k)

        prompt = main_aug_interp_prompt(
            examples=context_referenced_prompts,
            top_tokens=top_tokens,
            token_lr=token_lr,
            context_lr=context_lr,
        )

        res = gen_openai_completion(prompt, visualize_stream=visualize)

        explanation = extract_explanation(
            res, replace_neuron_with_feature=replace_neuron_with_feature
        )

        return explanation, res, prompt

    def get_top_k_tokens(self, k=5) -> List[str]:
        if self.feature_type == "attn":
            z_vec = self.encoder.W_dec[self.feature]
            z_vec = einops.rearrange(z_vec, "(n d) -> n d", n=self.model.cfg.n_heads)

            feature_vec = einops.einsum(
                z_vec, self.model.W_O[self.layer], "n d, n d o -> o"
            )
        else:
            feature_vec = self.encoder.W_dec[self.feature]

        feature_logits = einops.einsum(feature_vec, self.model.W_U, "d, d c -> c")
        top_tokens = feature_logits.topk(k).indices

        return self.model.to_str_tokens(top_tokens)  # type: ignore

    def get_context_referenced_prompt(
        self,
        i,
        token_lr=("<<", ">>"),
        context_lr=("[[", "]]"),
        merge_nearby_context=False,
    ):
        pos = self.get_active_example(i)[1]

        cd = self.get_circuit_discovery_for_max_activating_example(i)

        return cd.get_context_referenced_prompt(
            pos=pos,
            token_lr=token_lr,
            context_lr=context_lr,
            merge_nearby_context=merge_nearby_context,
        )

    def get_active_example(self, i):
        seq_pos, vals, _ = self.active_examples

        seq, pos = seq_pos[i]

        return seq, pos, vals[i]

    @property
    def active_examples(
        self,
    ):
        if self._active_examples is not None:
            return self._active_examples

        tokens = open_web_text_tokens[: self.num_sequences]

        if self.feature_type == "attn":
            name_filter = f"blocks.{self.layer}.attn.hook_z"
        else:
            name_filter = self.encoder.cfg.hook_point

        scores = []

        print("name", name_filter)

        for i in trange(0, tokens.shape[0], self.batch_size):
            with torch.no_grad():
                curr_tokens = tokens[i : i + self.batch_size]

                _, cache = self.model.run_with_cache(
                    curr_tokens,
                    stop_at_layer=self.layer + 1,
                    names_filter=name_filter,
                )
                acts = cache[name_filter]

                if self.feature_type == "attn":
                    acts_flat = einops.rearrange(acts, "b pos n d -> (b pos) (n d)")
                else:
                    acts_flat = einops.rearrange(acts, "b pos d -> (b pos) d")

                hidden_acts = self.encoder.encode(acts_flat)
                curr_scores = hidden_acts[:, self.feature]
                del hidden_acts

            scores.append(
                einops.rearrange(
                    curr_scores, "(b pos) -> b pos", b=curr_tokens.size(0)
                ).cpu()
            )

            del curr_scores

        scores = torch.concat(scores, dim=0)

        score_flat = scores.flatten()
        num_active = (score_flat > 0).sum().item()

        indices = score_flat.argsort(descending=True)[:num_active]
        vals = score_flat[indices]

        seq_pos = torch.stack(torch.unravel_index(indices, scores.shape)).T

        self._active_examples = (seq_pos, vals, scores)

        return self._active_examples

    def show_top_active_examples(self, num_examples=5, token_buffer=10):
        seq_pos, vals, scores = self.active_examples

        for i in range(num_examples):
            seq, pos = seq_pos[i]
            val = vals[i]

            str_tokens = self.model.to_str_tokens(open_web_text_tokens[seq])

            min_index = max(0, pos.item() - token_buffer)
            max_index = pos.item() + token_buffer

            print(f"Token: '{str_tokens[pos]}' | Value: {val.item():.3g}")

            display(
                cv.tokens.colored_tokens(
                    tokens=str_tokens[min_index:max_index],  # type: ignore
                    values=scores[seq, min_index:max_index],
                )
            )

    def show_example(self, example_i, token_buffer=10, no_cutoff=False):
        seq_pos, vals, scores = self.active_examples

        seq, pos = seq_pos[example_i]
        val = vals[example_i]

        str_tokens = self.model.to_str_tokens(open_web_text_tokens[seq])

        min_index = max(0, pos.item() - token_buffer)
        max_index = pos.item() + token_buffer

        if no_cutoff:
            min_index = 0
            max_index = len(str_tokens)

        print(f"Token: '{str_tokens[pos]}' | Value: {val.item():.3g}")

        display(
            cv.tokens.colored_tokens(
                tokens=str_tokens[min_index:max_index],  # type: ignore
                values=scores[seq, min_index:max_index],
            )
        )

    @property
    def num_examples(self):
        return self.active_examples[0].size(0)
