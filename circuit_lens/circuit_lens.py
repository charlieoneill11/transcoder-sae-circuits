import torch
import einops

from z_sae import ZSAE
from mlp_transcoder import SparseTranscoder
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int
from torch import Tensor
from typing import List, Dict, TypedDict, Any, Union, Tuple, Optional
from tqdm import trange
from plotly_utils import imshow
from pprint import pprint

from dataclasses import dataclass

ATOL = 1e-4


class LayerKey(TypedDict):
    mlp: int
    attn: int


NUM_ATTN_AUXILARY_FEATURES = 3
NUM_MLP_AUXILARY_FEATURES = 2

NUM_AUXILARY_FEATURES = NUM_ATTN_AUXILARY_FEATURES + NUM_MLP_AUXILARY_FEATURES


@dataclass
class ActiveFeatures:
    vectors: Float[Tensor, "comp d_model"]
    values: Float[Tensor, "comp"]
    features: Int[Tensor, "comp"]
    keys: List[LayerKey]

    def get_total_active_features(self):
        total = 0

        for key in self.keys:
            total += key["mlp"] + key["attn"]

        return total

    def get_total_auxilary_features(self):
        return 2 + (NUM_AUXILARY_FEATURES * len(self.keys))

    def get_vectors_before_comp(self, kind: str, layer: int):
        max_index = 2  # include embed and pos_embed

        for i in range(layer):
            max_index += (
                self.keys[i]["mlp"] + self.keys[i]["attn"] + NUM_AUXILARY_FEATURES
            )  # include error and bias terms

        if kind == "mlp":
            max_index += self.keys[layer]["attn"] + NUM_ATTN_AUXILARY_FEATURES

        return self.vectors[:max_index] * self.values[:max_index].unsqueeze(-1)

    @property
    def max_active_features(self):
        lens = []

        for key in self.keys:
            lens.append(key["mlp"])
            lens.append(key["attn"])

        return max(lens)

    def get_attn_start_index(self, layer: int):
        start_i = 2

        for i, key in enumerate(self.keys):
            if i == layer:
                break

            start_i += key["attn"] + key["mlp"] + NUM_AUXILARY_FEATURES

        return start_i

    def get_mlp_start_index(self, layer: int):
        start_i = 2

        for i, key in enumerate(self.keys):
            if i == layer:
                break

            start_i += key["attn"] + key["mlp"] + NUM_AUXILARY_FEATURES

        start_i += self.keys[layer]["attn"] + NUM_ATTN_AUXILARY_FEATURES

        return start_i

    def get_reconstructed_attn_out(self, layer: int):
        attn_i = self.get_attn_start_index(layer)
        num_comps = self.keys[layer]["attn"] + NUM_ATTN_AUXILARY_FEATURES

        return self.vectors[attn_i : attn_i + num_comps].sum(dim=0)

    def get_sae_out_reconstruction(self, layer: int):
        attn_i = self.get_attn_start_index(layer)
        attn_i += 2

        return self.vectors[attn_i : attn_i + self.keys[layer]["attn"] + 1].sum(dim=0)

    def get_transcoder_reconstruction(self, layer: int):
        mlp_i = self.get_mlp_start_index(layer)
        mlp_i += 1

        return self.vectors[mlp_i : mlp_i + self.keys[layer]["mlp"] + 1].sum(dim=0)

    def get_attn_feature_vectors(self, layer: int):
        attn_i = self.get_attn_start_index(layer)
        attn_i += NUM_ATTN_AUXILARY_FEATURES

        return self.vectors[attn_i : attn_i + self.keys[layer]["attn"]]

    def get_mlp_feature_vectors(self, layer: int):
        mlp_i = self.get_mlp_start_index(layer)
        mlp_i += NUM_MLP_AUXILARY_FEATURES

        return self.vectors[mlp_i : mlp_i + self.keys[layer]["mlp"]]

    def get_reconstructed_mlp_out(self, layer: int):
        mlp_i = self.get_mlp_start_index(layer)

        num_comps = self.keys[layer]["mlp"] + NUM_MLP_AUXILARY_FEATURES

        return self.vectors[mlp_i : mlp_i + num_comps].sum(dim=0)

    def get_top_k_features(self, activations: Float[Tensor, "comp"], k=10):
        values, indices = activations.topk(k=k)

        features = []

        for v, i in zip(values.tolist(), indices.tolist()):
            if i == 0:
                features.append(("embed", 0, self.features[i], v))
                break
            elif i == 1:
                features.append(("pos_embed", 0, self.features[i], v))
                break

            start_i = 2

            for l, key in enumerate(self.keys):
                if i == start_i:
                    features.append(("O Bias", l, self.features[i], v))
                    break
                start_i += 1

                if i == start_i:
                    features.append(("Z SAE Error", l, self.features[i], v))
                    break
                start_i += 1

                if i == start_i:
                    features.append(("Z SAE Bias", l, self.features[i], v))
                    break
                start_i += 1

                if i < start_i + key["attn"]:
                    features.append(("attn", l, self.features[i], v))
                    break

                start_i += key["attn"]

                if i == start_i:
                    features.append(("Transcoder Error", l, self.features[i], v))
                    break
                start_i += 1

                if i == start_i:
                    features.append(("Transcoder Bias", l, self.features[i], v))
                    break
                start_i += 1

                if i < start_i + key["mlp"]:
                    features.append(("mlp", l, self.features[i], v))
                    break

                start_i += key["mlp"]

        return features

    def get_top_k_lens_runs(
        self,
        activations: Float[Tensor, "comp"],
        web: "CircuitLens",
        seq_index: int,
        k=10,
    ):
        features = self.get_top_k_features(activations, k=k)

        lens_runs = []

        for feature in features:
            if feature[0] == "attn":
                run_type = "z_feature_head_seq"
            elif feature[0] == "mlp":
                run_type = "mlp"
            else:
                run_type = feature[0]

            lens_runs.append(
                ComponentLens(
                    web,
                    run_data={
                        "run_type": run_type,
                        "layer": feature[1],
                        "seq_index": seq_index,
                        "feature": feature[2],
                    },
                )
            )

        return lens_runs

    def get_top_k_labels(self, activations: Float[Tensor, "comp"], k=10):
        features = self.get_top_k_features(activations, k=k)

        return [
            f"{kind.capitalize()} {layer} | Feature: {feature} | Value: {value:.3g}"
            for kind, layer, feature, value in features
        ]

    def reshape_activations_for_visualization(
        self, activations: Float[Tensor, "comp 1"]
    ):
        min_val = activations.min()

        visualization = torch.ones(
            ((12 * 4) + 1, self.max_active_features), device=activations.device
        ) * (min_val / 2)

        a_len = activations.size(0)

        visualization[0, :2] = activations[:2]

        start_i = 2

        for i, key in enumerate(self.keys):
            ii = (4 * i) + 1

            if start_i + key["attn"] + NUM_ATTN_AUXILARY_FEATURES > a_len:
                break

            visualization[ii, :NUM_ATTN_AUXILARY_FEATURES] = activations[
                start_i : start_i + NUM_ATTN_AUXILARY_FEATURES
            ]
            start_i += NUM_ATTN_AUXILARY_FEATURES

            visualization[ii + 1, : key["attn"]] = activations[
                start_i : start_i + key["attn"]
            ]
            start_i += key["attn"]

            if start_i + key["mlp"] + NUM_MLP_AUXILARY_FEATURES > a_len:
                break

            visualization[ii + 2, :NUM_MLP_AUXILARY_FEATURES] = activations[
                start_i : start_i + NUM_MLP_AUXILARY_FEATURES
            ]
            start_i += NUM_MLP_AUXILARY_FEATURES

            visualization[ii + 3, : key["mlp"]] = activations[
                start_i : start_i + key["mlp"]
            ]
            start_i += key["mlp"]

        return visualization


@dataclass
class ComponentLens:
    web: "CircuitLens"
    run_data: Dict[str, Any]

    @property
    def run_type(self):
        return self.run_data["run_type"]

    def __str__(self):
        if self.run_type == "unembed":
            return f"Unembed | Token: '{self.web.model.tokenizer.decode([self.run_data['token']])}' :: {self.run_data['token']} | Seq Index: {self.run_data['seq_index']}"
        elif self.run_type == "z_feature_head_seq":
            return f"Z Feature Head/Seq | Feature: {self.run_data['feature']} |Layer: {self.run_data['layer']} | Seq Index: {self.run_data['seq_index']}"
        elif self.run_type == "head":
            return f"Head | Layer: {self.run_data['layer']} | Head: {self.run_data['head']} | Source: {self.run_data['source_index']} | Destination: {self.run_data['destination_index']}"
        elif self.run_type == "mlp":
            return f"MLP Lens | Layer: {self.run_data['layer']} | Seq Index: {self.run_data['seq_index']} | Feature: {self.run_data['feature']}"
        else:
            return f"{self.run_type} | Layer: {self.run_data['layer']} | Seq Index: {self.run_data['seq_index']} | Feature: {self.run_data['feature']}"

    def __repr__(self):
        return str(self)

    def __call__(self, head_type=None, **kwargs):
        if self.run_type == "unembed":
            return self.web.get_unembed_lens(
                self.run_data["token_id"], self.run_data["seq_index"] ** kwargs
            )
        elif self.run_type == "z_feature_head_seq":
            return self.web.get_head_seq_activations_for_z_feature(
                self.run_data["layer"],
                self.run_data["seq_index"],
                self.run_data["feature"],
                **kwargs,
            )
        elif self.run_type == "head":
            if head_type is None:
                head_type = "q"

            if head_type == "q":
                return self.web.get_q_lens_on_head_seq(
                    self.run_data["layer"],
                    self.run_data["head"],
                    self.run_data["source_index"],
                    self.run_data["destination_index"],
                    **kwargs,
                )
            elif head_type == "k":
                return self.web.get_k_lens_on_head_seq(
                    self.run_data["layer"],
                    self.run_data["head"],
                    self.run_data["source_index"],
                    self.run_data["destination_index"],
                    **kwargs,
                )
            elif head_type == "v":
                return self.web.get_v_lens_at_seq(
                    self.run_data["layer"],
                    self.run_data["head"],
                    self.run_data["source_index"],
                    self.run_data["feature"],
                    **kwargs,
                )
        elif self.run_type == "mlp":
            return self.web.get_mlp_feature_lens_at_seq(
                self.run_data["layer"],
                self.run_data["seq_index"],
                self.run_data["feature"],
                **kwargs,
            )


model_encoder_cache: Optional[Tuple[HookedTransformer, Any, Any]] = None


def get_model_encoders():
    global model_encoder_cache

    if model_encoder_cache is not None:
        return model_encoder_cache

    model = HookedTransformer.from_pretrained("gpt2-small")

    z_saes = [ZSAE.load_zsae_for_layer(i) for i in trange(model.cfg.n_layers)]

    mlp_transcoders = [
        SparseTranscoder.load_from_hugging_face(i) for i in trange(model.cfg.n_layers)
    ]

    model_encoder_cache = (model, z_saes, mlp_transcoders)

    return model_encoder_cache


class CircuitLens:
    def __init__(self, prompt):
        (model, z_saes, mlp_transcoders) = get_model_encoders()

        self.z_saes = z_saes
        self.mlp_transcoders = mlp_transcoders
        self.model: HookedTransformer = model

        self.prompt = prompt
        self.tokens = self.model.to_tokens(prompt)

        self.logits, self.cache = self.model.run_with_cache(
            self.tokens, return_type="logits"
        )

        self._seq_activations = {}

    @property
    def n_tokens(self):
        return self.tokens.size(1)

    def test_compare_sae_out(self, layer, seq_index):
        seq_index = self.process_seq_index(seq_index)

        layer_z = einops.rearrange(
            self.cache["z", layer][0, seq_index],
            "n_heads d_head -> (n_heads d_head)",
        )

        test_sae_out = self.get_active_features(
            seq_index, cache=False
        ).get_sae_out_reconstruction(layer)

        _, z_recon, _, _, _ = self.z_saes[layer](layer_z)

        z_recon = einops.rearrange(
            z_recon,
            "(n_head d_head) -> n_head d_head",
            n_head=self.model.cfg.n_heads,
        )
        z_recon = einops.einsum(
            z_recon,
            self.model.W_O[layer],
            "n_head d_head, n_head d_head d_model -> d_model",
        )

        return (
            torch.allclose(z_recon, test_sae_out, atol=ATOL),
            (z_recon - test_sae_out).norm().item(),
        )

    def test_compare_transcoder_out(self, layer, seq_index):
        seq_index = self.process_seq_index(seq_index)

        test_transcoder_out = self.get_active_features(
            seq_index, cache=False
        ).get_transcoder_reconstruction(layer)

        mlp_input = self.cache["normalized", layer, "ln2"]
        out = self.mlp_transcoders[layer](mlp_input)[0][0, seq_index]

        return (
            torch.allclose(out, test_transcoder_out, atol=ATOL),
            (out - test_transcoder_out).norm().item(),
        )

    def test_compare_max_attn_features(self, layer: int, seq_index):
        seq_index = self.process_seq_index(seq_index)
        active_features = self.get_active_features(seq_index, cache=False)

        z_sae = self.z_saes[layer]

        max_features = active_features.get_attn_feature_vectors(layer)

        layer_z = einops.rearrange(
            self.cache["z", layer][0, seq_index],
            "n_heads d_head -> (n_heads d_head)",
        )
        _, _, z_acts, _, _ = self.z_saes[layer](layer_z)

        z_winner_count = z_acts.nonzero().numel()

        z_values, z_max_features = z_acts.topk(k=z_winner_count)

        z_contributions = z_sae.W_dec[z_max_features.squeeze(0)] * z_values.squeeze(
            0
        ).unsqueeze(-1)
        z_contributions = einops.rearrange(
            z_contributions,
            "winners (n_head d_head) -> winners n_head d_head",
            n_head=self.model.cfg.n_heads,
        )
        z_residual_vectors = einops.einsum(
            z_contributions,
            self.model.W_O[layer],
            "winners n_head d_head, n_head d_head d_model -> winners d_model",
        )

        return (
            torch.allclose(z_residual_vectors, max_features, atol=ATOL),
            (z_residual_vectors - max_features).norm().item(),
        )

    def test_compare_max_mlp_features(self, layer: int, seq_index: int):
        seq_index = self.process_seq_index(seq_index)

        mlp_transcoder = self.mlp_transcoders[layer]
        mlp_input = self.cache["normalized", layer, "ln2"][:, seq_index]

        _, mlp_acts, *_ = mlp_transcoder(mlp_input)

        mlp_acts = mlp_acts[0]

        mlp_winner_count = mlp_acts.nonzero().numel()

        mlp_values, mlp_max_features = mlp_acts.topk(k=mlp_winner_count)

        mlp_residual_vectors = mlp_transcoder.W_dec[
            mlp_max_features.squeeze(0)
        ] * mlp_values.squeeze(0).unsqueeze(-1)

        test_vectors = self.get_active_features(
            seq_index, cache=False
        ).get_mlp_feature_vectors(layer)

        return (
            torch.allclose(mlp_residual_vectors, test_vectors, atol=ATOL),
            (mlp_residual_vectors - test_vectors).norm().item(),
        )

    def test_compare_attn_out(self, layer: int, seq_index: int):
        seq_index = self.process_seq_index(seq_index)
        active_features = self.get_active_features(seq_index, cache=False)
        feature_recon = active_features.get_reconstructed_attn_out(layer)

        attn_out = self.cache["attn_out", layer][0, seq_index]

        return (
            torch.allclose(feature_recon, attn_out, atol=ATOL),
            (feature_recon - attn_out).norm().item(),
        )

    def test_compare_mlp_out(self, layer: int, seq_index: int):
        seq_index = self.process_seq_index(seq_index)

        active_features = self.get_active_features(seq_index, cache=False)
        feature_recon = active_features.get_reconstructed_mlp_out(layer)

        mlp_out = self.cache["mlp_out", layer][0, seq_index]

        return (
            torch.allclose(feature_recon, mlp_out, atol=ATOL),
            (feature_recon - mlp_out).norm(),
        )

    def test_compare_final_resid_post(self, seq_index: int):
        seq_index = self.process_seq_index(seq_index)

        cumulative_sum = self.get_active_features(seq_index, cache=False).vectors.sum(
            dim=0
        )

        resid_post = self.cache["resid_post", self.model.cfg.n_layers - 1][0, seq_index]

        return (
            torch.allclose(cumulative_sum, resid_post, atol=ATOL),
            (cumulative_sum - resid_post).norm(),
        )

    def get_active_features(self, seq_index: int, cache=True) -> ActiveFeatures:
        seq_index = self.process_seq_index(seq_index)

        act = self._seq_activations.get(seq_index, None)

        if cache and act is not None:
            return act

        component_keys: List[LayerKey] = []

        t1 = torch.tensor(1.0).to(self.cache["z", 0].device)
        t0 = torch.tensor(-1).to(self.cache["z", 0].device).int()

        # Start with embed and pos embed
        vectors = [
            torch.stack(
                [
                    self.cache["embed"][0, seq_index],
                    self.cache["pos_embed"][0, seq_index],
                ]
            )
        ]
        values = [torch.stack([t1, t1])]
        features = [torch.stack([t0, t0])]

        for layer in trange(self.model.cfg.n_layers):
            # First handle attention
            z_sae = self.z_saes[layer]

            layer_z = einops.rearrange(
                self.cache["z", layer][0, seq_index],
                "n_heads d_head -> (n_heads d_head)",
            )
            _, z_recon, z_acts, _, _ = self.z_saes[layer](layer_z)

            z_error = layer_z - z_recon
            z_bias = self.z_saes[layer].b_dec

            z_error_bias = torch.stack([z_error, z_bias])
            z_error_bias = einops.rearrange(
                z_error_bias,
                "v (n_head d_head) -> v n_head d_head",
                n_head=self.model.cfg.n_heads,
            )
            z_error_bias = einops.einsum(
                z_error_bias,
                self.model.W_O[layer],
                "v n_head d_head, n_head d_head d_model -> v d_model",
            )

            vectors.append(self.model.b_O[layer].unsqueeze(0))

            vectors.append(z_error_bias)
            values.append(torch.stack([t1, t1, t1]))
            features.append(torch.stack([t0, t0, t0]))

            z_winner_count = z_acts.nonzero().numel()

            z_values, z_max_features = z_acts.topk(k=z_winner_count)

            z_contributions = z_sae.W_dec[z_max_features.squeeze(0)] * z_values.squeeze(
                0
            ).unsqueeze(-1)
            z_contributions = einops.rearrange(
                z_contributions,
                "winners (n_head d_head) -> winners n_head d_head",
                n_head=self.model.cfg.n_heads,
            )
            z_residual_vectors = einops.einsum(
                z_contributions,
                self.model.W_O[layer],
                "winners n_head d_head, n_head d_head d_model -> winners d_model",
            )

            vectors.append(z_residual_vectors)
            values.append(z_values)
            features.append(z_max_features)

            # Now handle the transcoder
            mlp_transcoder = self.mlp_transcoders[layer]
            mlp_input = self.cache["normalized", layer, "ln2"][:, seq_index]
            mlp_output = self.cache["mlp_out", layer][:, seq_index]

            mlp_recon, mlp_acts, *_ = mlp_transcoder(mlp_input)

            mlp_error = mlp_output - mlp_recon
            mlp_bias = mlp_transcoder.b_dec_out

            vectors.append(torch.stack([mlp_error[0], mlp_bias]))
            values.append(torch.stack([t1, t1]))
            features.append(torch.stack([t0, t0]))

            mlp_winner_count = mlp_acts.nonzero().numel()

            mlp_values, mlp_max_features = mlp_acts.topk(k=mlp_winner_count)

            mlp_residual_vectors = mlp_transcoder.W_dec[
                mlp_max_features[0]
            ] * mlp_values[0].unsqueeze(-1)

            vectors.append(mlp_residual_vectors)
            values.append(mlp_values.squeeze())
            features.append(mlp_max_features.squeeze())

            component_keys.append({"mlp": mlp_winner_count, "attn": z_winner_count})

        component_vectors = torch.cat(vectors, dim=0)
        component_values = torch.cat(values, dim=0)
        component_features = torch.cat(features, dim=0)

        self._seq_activations[seq_index] = ActiveFeatures(
            vectors=component_vectors,
            values=component_values,
            features=component_features,
            keys=component_keys,
        )

        return self._seq_activations[seq_index]

    def visualize_values(self, seq_index: int):
        active_features = self.get_active_features(seq_index)

        vis = active_features.reshape_activations_for_visualization(
            active_features.values
        )

        imshow(
            vis[:, :20],
            title=f"Values for Seq Index {seq_index}",
            y=self.get_imshow_labels()[: vis.size(0)],
            height=800,
            width=600,
        )

    def get_next_lens_runs(
        self,
        active_features,
        activations,
        seq_index: int,
        title: str,
        visualize=True,
        k=None,
    ):
        if k is None:
            k = 10

        if visualize:
            vis = active_features.reshape_activations_for_visualization(activations)

            imshow(
                vis[:, :20],
                title=title,
                y=self.get_imshow_labels()[: vis.size(0)],
                height=800,
                width=600,
            )

            pprint(active_features.get_top_k_labels(activations, k=k))

        return active_features.get_top_k_lens_runs(activations, self, seq_index, k=k)

    def get_unembed_lens_for_prompt_token(self, seq_index: int, visualize=True, k=None):
        """
        Here `seq_index` refers to the seq position where the next token will be predicted
        """
        seq_index = self.process_seq_index(seq_index)

        token_i = self.tokens[0, seq_index + 1].item()

        return self.get_unembed_lens(token_i, seq_index, visualize, k)

    _labels = None

    def get_imshow_labels(self):
        if self._labels is not None:
            return self._labels

        labels = ["Embed"]
        for i in range(12):
            labels.append(f"Attn {i} Error/Bias")
            labels.append(f"Attn {i}")
            labels.append(f"Mlp {i} Error/Bias")
            labels.append(f"Mlp {i}")
        self._labels = labels

        return labels

    def process_seq_index(self, seq_index):
        if seq_index < 0:
            seq_index += self.n_tokens
        return seq_index

    def get_unembed_lens(
        self, token_i: Union[int, str], seq_index: int, visualize=True, k=None
    ):
        if isinstance(token_i, str):
            token_i = self.model.to_tokens(token_i, prepend_bos=False)[0]

        seq_index = self.process_seq_index(seq_index)
        active_features = self.get_active_features(seq_index)

        activations = (
            einops.einsum(
                active_features.vectors,
                self.model.W_U[:, token_i],
                "comp d_model, d_model -> comp",
            )
            / self.cache["ln_final.hook_scale"][0, seq_index]
        )

        activations /= self.logits[0, seq_index][token_i]

        return self.get_next_lens_runs(
            active_features=active_features,
            activations=activations,
            title=f"Unembed Lens for token '{self.model.tokenizer.decode([token_i])}' at '{self.model.tokenizer.decode([token_i])}",
            seq_index=seq_index,
            visualize=visualize,
            k=k,
        )

    def get_head_seq_activations_for_z_feature(
        self, layer: int, seq_index: int, feature: int, visualize=True, k=10
    ):
        seq_index = self.process_seq_index(seq_index)
        v = self.cache["v", layer]
        pattern = self.cache["pattern", 9]
        encoder = self.z_saes[layer]

        seq_index = self.process_seq_index(seq_index)

        pre_z = einops.einsum(
            v,
            pattern,
            "b p_seq n_head d_head, b n_head seq p_seq -> seq b p_seq n_head d_head",
        )[seq_index, 0]

        better_w_enc = einops.rearrange(
            encoder.W_enc, "(n_head d_head) feature -> n_head d_head feature", n_head=12
        )[:, :, feature]

        feature_act = einops.einsum(
            pre_z, better_w_enc, "seq n_head d_head, n_head d_head -> n_head seq"
        )
        feature_act = einops.rearrange(feature_act, "n_head seq -> (n_head seq)")

        values, indices = feature_act.topk(k=k)

        lens_runs = []
        vis_list = []

        for index, value in zip(indices, values):
            head = index // self.n_tokens
            source = index % self.n_tokens

            lens_runs.append(
                ComponentLens(
                    web=self,
                    run_data={
                        "run_type": "head",
                        "layer": layer,
                        "head": head,
                        "source_index": source,
                        "feature": feature,
                        "destination_index": seq_index,
                    },
                )
            )

            if visualize:
                vis_list.append((head, source, seq_index, value))

        if visualize:
            vis = einops.rearrange(
                feature_act, "(n_head seq) -> n_head seq", n_head=self.model.cfg.n_heads
            )

            imshow(
                vis,
                x=[
                    f"{token}/{i}"
                    for (i, token) in enumerate(self.model.to_str_tokens(self.prompt))
                ],
                title=f"Layer {layer} Head/Seq Feature {feature} at '{self.get_str_token_at_seq(seq_index)}'::{seq_index}",
                labels={"x": "Token", "y": "Head"},
            )

            pprint(
                [
                    f"Head {head} "
                    + f"| Source: {self.model.tokenizer.decode([self.tokens[0, source]])}::{source} "
                    + f"| Destination: {self.model.tokenizer.decode([self.tokens[0, dest]])}::{dest} "
                    + f"| Value: {value:.3g}"
                    for (head, source, dest, value) in vis_list
                ]
            )

        return lens_runs

    def get_mlp_feature_lens_at_seq(
        self,
        layer: int,
        seq_index: int,
        feature: int,
        visualize=True,
        k=None,
    ):
        seq_index = self.process_seq_index(seq_index)

        active_features = self.get_active_features(seq_index)
        transcoder = self.mlp_transcoders[layer]

        vectors = active_features.get_vectors_before_comp("mlp", layer)

        activation = einops.einsum(
            vectors, transcoder.W_enc[:, feature], "comp d_model, d_model -> comp"
        )

        return self.get_next_lens_runs(
            active_features,
            activation,
            seq_index,
            title=f"MLP Lens for Feature {feature} at '{self.get_str_token_at_seq(seq_index)}' :: {seq_index}",
            visualize=visualize,
            k=k,
        )

    def get_str_token_at_seq(self, seq_index: int):
        seq_index = self.process_seq_index(seq_index)
        return self.model.tokenizer.decode([self.tokens[0, seq_index]])

    def get_v_lens_at_seq(
        self,
        layer: int,
        head: int,
        seq_index: int,
        feature: int,
        visualize=True,
        k=None,
    ):
        seq_index = self.process_seq_index(seq_index)
        active_features = self.get_active_features(seq_index)
        z_sae = self.z_saes[layer]

        vectors = active_features.get_vectors_before_comp("attn", layer)

        effective_v = einops.einsum(
            vectors,
            self.model.W_V[layer, head],
            "comp d_model, d_model d_head -> comp d_head",
        )

        effective_feature = einops.rearrange(
            z_sae.W_enc[:, feature],
            "(n_head d_head) -> n_head d_head",
            n_head=self.model.cfg.n_heads,
        )[head]

        activation = einops.einsum(
            effective_v, effective_feature, "comp d_head, d_head -> comp"
        )

        return self.get_next_lens_runs(
            active_features,
            activation,
            seq_index,
            title=f"V Lens | Layer {layer} | Head {head} | Feature {feature} at  '{self.get_str_token_at_seq(seq_index)}'::{seq_index}",
            visualize=visualize,
            k=k,
        )

    def get_q_lens_on_head_seq(
        self,
        layer: int,
        head: int,
        source_index,
        destination_index,
        visualize=True,
        k=None,
    ):
        source_index = self.process_seq_index(source_index)
        destination_index = self.process_seq_index(destination_index)

        active_features = self.get_active_features(destination_index)

        vectors = active_features.get_vectors_before_comp("attn", layer)

        effective_q = einops.einsum(
            vectors,
            self.model.W_Q[layer, head],
            "comp d_model, d_model d_head -> comp d_head",
        )

        effective_k = self.cache["k", layer][0, source_index, head]

        bias_q = self.model.b_Q[layer, head]
        real_q = self.cache["q", layer][0, destination_index, head]
        bias_contrib = einops.einsum(bias_q, effective_k, "d_head, d_head -> ")
        qk = einops.einsum(real_q, effective_k, "d_head, d_head -> ")
        ln_scale = self.cache["scale", layer, "ln1"][0, destination_index]

        print("qk ", qk, "bias", bias_contrib, "ln-scale", ln_scale)

        activation = einops.einsum(
            effective_q, effective_k, "comp d_head, d_head -> comp"
        ) / (ln_scale * (qk - bias_contrib))

        return self.get_next_lens_runs(
            active_features,
            activation,
            destination_index,
            title=f"Q Lens | Layer {layer} | Head {head} | '{self.get_str_token_at_seq(source_index)}'::{source_index} -> '{self.get_str_token_at_seq(destination_index)}'::{destination_index}",
            visualize=visualize,
            k=k,
        )

    def get_k_lens_on_head_seq(
        self,
        layer: int,
        head: int,
        source_index,
        destination_index,
        visualize=True,
        k=None,
    ):
        active_features = self.get_active_features(source_index)

        vectors = active_features.get_vectors_before_comp("attn", layer)

        effective_k = einops.einsum(
            vectors,
            self.model.W_K[layer, head],
            "comp d_model, d_model d_head -> comp d_head",
        )

        effective_q = self.cache["q", layer][0, destination_index, head]

        bias_k = self.model.b_K[layer, head]
        real_k = self.cache["k", layer][0, source_index, head]

        bias_contrib = einops.einsum(bias_k, effective_q, "d_head, d_head -> ")
        qk = einops.einsum(real_k, effective_q, "d_head, d_head -> ")
        ln_scale = self.cache["scale", layer, "ln1"][0, source_index]

        activation = einops.einsum(
            effective_k, effective_q, "comp d_head, d_head -> comp"
        ) / (ln_scale * (qk - bias_contrib))

        return self.get_next_lens_runs(
            active_features,
            activation,
            destination_index,
            title=f"K Lens | Layer {layer} | Head {head} | '{self.get_str_token_at_seq(source_index)}'::{source_index} -> '{self.get_str_token_at_seq(destination_index)}'::{destination_index}",
            visualize=visualize,
            k=k,
        )
