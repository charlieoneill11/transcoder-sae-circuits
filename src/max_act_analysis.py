import torch
import einops
import circuitsvis as cv

from open_web_text import open_web_text_tokens
from mlp_transcoder import SparseTranscoder
from z_sae import ZSAE
from transformer_lens import HookedTransformer
from tqdm import trange


class MaxActAnalysis:
    def __init__(self, feature_type: str, layer: int, feature: int):
        assert feature_type in ("attn", "mlp")
        self.feature_type = feature_type
        self.model = HookedTransformer.from_pretrained("gpt2")
        self.feature = feature
        self.layer = layer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if feature_type == "attn":
            self.encoder = ZSAE.load_zsae_for_layer(layer)
        else:
            self.encoder = SparseTranscoder.load_from_hugging_face(layer)

    _active_examples = None

    def get_active_examples(
        self,
        num_sequences=None,
        batch_size=64,
    ):
        if self._active_examples is not None:
            return self._active_examples

        if num_sequences is None:
            num_sequences = open_web_text_tokens.size(0)

        tokens = open_web_text_tokens[:num_sequences]

        if self.feature_type == "attn":
            name_filter = f"blocks.{self.layer}.attn.hook_z"
        else:
            name_filter = self.encoder.cfg.hook_point

        scores = []

        for i in trange(0, tokens.shape[0], batch_size):
            with torch.no_grad():
                curr_tokens = tokens[i : i + batch_size]

                _, cache = self.model.run_with_cache(
                    curr_tokens,
                    stop_at_layer=self.layer + 1,
                    names_filter=name_filter,
                )
                acts = cache[name_filter]
                # acts_flat = acts.reshape(-1, encoder.W_enc.shape[0])
                acts_flat = einops.rearrange(acts, "b pos n d -> (b pos) (n d)")

                hidden_acts = self.encoder.encode(acts_flat)
                curr_scores = hidden_acts[:, self.feature]
                del hidden_acts

            scores.append(
                einops.rearrange(curr_scores, "(b pos) -> b pos", b=curr_tokens.size(0))
            )

        scores = torch.concat(scores, dim=0)

        score_flat = scores.flatten()
        num_active = (score_flat > 0).sum().item()

        indices = score_flat.argsort(descending=True)[:num_active]
        vals = score_flat[indices]

        seq_pos = torch.stack(torch.unravel_index(indices, scores.shape))

        self._active_examples = (seq_pos, vals, scores)

        return self._active_examples

    def show_top_active_examples(self, num_examples=5, token_buffer=10):
        seq_pos, _, scores = self.get_active_examples()

        for i in range(num_examples):
            seq, pos = seq_pos[i]
            str_tokens = self.model.to_str_tokens(open_web_text_tokens[seq])

            min_index = max(0, pos.item() - token_buffer)
            max_index = pos.item() + token_buffer

            cv.tokens.colored_tokens(
                tokens=str_tokens[min_index:max_index],
                values=scores[seq, min_index:max_index],
            )
