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


class CircuitComponent:
    Z_FEATURE = "z_feature"
    MLP_FEATURE = "mlp_feature"
    ATTN_HEAD = "attn_head"
    UNEMBED = "unembed"
    UNEMBED_AT_TOKEN = "unembed_at_token"
    EMBED = "embed"
    POS_EMBED = "pos_embed"
    BIAS_O = "b_O"
    Z_SAE_ERROR = "z_sae_error"
    Z_SAE_BIAS = "z_sae_bias"
    TRANSCODER_ERROR = "transcoder_error"
    TRANSCODER_BIAS = "transcoder_bias"


class LayerKey(TypedDict):
    mlp: int
    attn: int


BIAS_OFFSET = 0
Z_ERROR_OFFSET = 1
Z_BIAS_OFFSET = 2

NUM_ATTN_AUXILARY_FEATURES = 3
NUM_MLP_AUXILARY_FEATURES = 2

NUM_AUXILARY_FEATURES = NUM_ATTN_AUXILARY_FEATURES + NUM_MLP_AUXILARY_FEATURES


def map_activation_index_to_feature(index, keys):
    current_index = 0
    
    # Embed and Pos Embed
    if index == current_index:
        return CircuitComponent.EMBED, 0
    current_index += 1
    if index == current_index:
        return CircuitComponent.POS_EMBED, 0
    current_index += 1
    
    # Iterate through each layer
    for layer, key in enumerate(keys):
        # Bias O
        if index == current_index:
            return CircuitComponent.BIAS_O, layer
        current_index += 1
        
        # Z SAE Error
        if index == current_index:
            return CircuitComponent.Z_SAE_ERROR, layer
        current_index += 1
        
        # Z SAE Bias
        if index == current_index:
            return CircuitComponent.Z_SAE_BIAS, layer
        current_index += 1
        
        # Attention Features
        if index < current_index + key["attn"]:
            return CircuitComponent.Z_FEATURE, layer, index - current_index
        current_index += key["attn"]
        
        # Transcoder Error
        if index == current_index:
            return CircuitComponent.TRANSCODER_ERROR, layer
        current_index += 1
        
        # Transcoder Bias
        if index == current_index:
            return CircuitComponent.TRANSCODER_BIAS, layer
        current_index += 1
        
        # MLP Features
        if index < current_index + key["mlp"]:
            return CircuitComponent.MLP_FEATURE, layer, index - current_index
        current_index += key["mlp"]
    
    raise ValueError("Index out of range")


def map_feature_to_activation_index(feature, keys):
    component, layer, feature_index = feature
    
    current_index = 0
    
    # Embed and Pos Embed
    if component == CircuitComponent.EMBED:
        return current_index
    current_index += 1
    if component == CircuitComponent.POS_EMBED:
        return current_index
    current_index += 1
    
    # Iterate through each layer
    for l, key in enumerate(keys):
        if l == layer:
            # Bias O
            if component == CircuitComponent.BIAS_O:
                return current_index
            current_index += 1
            
            # Z SAE Error
            if component == CircuitComponent.Z_SAE_ERROR:
                return current_index
            current_index += 1
            
            # Z SAE Bias
            if component == CircuitComponent.Z_SAE_BIAS:
                return current_index
            current_index += 1
            
            # Attention Features
            if component == CircuitComponent.Z_FEATURE:
                return current_index + feature_index
            current_index += key["attn"]
            
            # Transcoder Error
            if component == CircuitComponent.TRANSCODER_ERROR:
                return current_index
            current_index += 1
            
            # Transcoder Bias
            if component == CircuitComponent.TRANSCODER_BIAS:
                return current_index
            current_index += 1
            
            # MLP Features
            if component == CircuitComponent.MLP_FEATURE:
                return current_index + feature_index
            current_index += key["mlp"]
    
    raise ValueError("Feature not found in keys")


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

        return self.vectors[:max_index]

    def get_z_error(self, layer: int):
        error_index = self.get_attn_start_index(layer) + Z_ERROR_OFFSET

        return self.vectors[error_index]

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

    def get_reconstructed_resid_pre(self, layer: int):
        return self.get_vectors_before_comp("attn", layer).sum(dim=0)

    def get_reconstructed_resid_post(self, layer: int):
        return self.get_vectors_before_comp("attn", layer + 1).sum(dim=0)

    def get_reconstructed_resid_mid(self, layer: int):
        return self.get_vectors_before_comp("mlp", layer).sum(dim=0)

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
        k = min(k, activations.size(0))

        values, indices = activations.topk(k=k)

        features = []

        for v, i in zip(values.tolist(), indices.tolist()):
            feature_i = self.features[i].item()

            if i == 0:
                features.append((CircuitComponent.EMBED, 0, feature_i, v))
                break
            elif i == 1:
                features.append((CircuitComponent.POS_EMBED, 0, feature_i, v))
                break

            start_i = 2

            for l, key in enumerate(self.keys):
                if i == start_i:
                    features.append((CircuitComponent.BIAS_O, l, feature_i, v))
                    break
                start_i += 1

                if i == start_i:
                    # features.append(("Z SAE Error", l, self.features[i], v))
                    features.append((CircuitComponent.Z_SAE_ERROR, l, feature_i, v))
                    break
                start_i += 1

                if i == start_i:
                    # features.append(("Z SAE Bias", l, self.features[i], v))
                    features.append((CircuitComponent.Z_SAE_BIAS, l, feature_i, v))
                    break
                start_i += 1

                if i < start_i + key["attn"]:
                    # features.append(("Attn", l, self.features[i], v))
                    features.append((CircuitComponent.Z_FEATURE, l, feature_i, v))
                    break

                start_i += key["attn"]

                if i == start_i:
                    # features.append(("Transcoder Error", l, self.features[i], v))
                    features.append(
                        (
                            CircuitComponent.TRANSCODER_ERROR,
                            l,
                            feature_i,
                            v,
                        )
                    )
                    break
                start_i += 1

                if i == start_i:
                    # features.append(("Transcoder Bias", l, self.features[i], v))
                    features.append(
                        (
                            CircuitComponent.TRANSCODER_BIAS,
                            l,
                            feature_i,
                            v,
                        )
                    )
                    break
                start_i += 1

                if i < start_i + key["mlp"]:
                    # features.append(("MLP", l, self.features[i], v))
                    features.append((CircuitComponent.MLP_FEATURE, l, feature_i, v))
                    break

                start_i += key["mlp"]

        return features

    def get_top_k_lens_runs(
        self,
        activations: Float[Tensor, "comp"],
        circuit_lens: "CircuitLens",
        seq_index: int,
        k=10,
    ) -> List["ComponentLensWithValue"]:
        features = self.get_top_k_features(activations, k=k)

        lens_runs = []

        for feature in features:
            component, layer, feature_i, value = feature

            lens_runs.append(
                (
                    ComponentLens(
                        circuit_lens=circuit_lens,
                        component=component,
                        run_data={
                            "layer": layer,
                            "seq_index": seq_index,
                            "feature": feature_i,
                        },
                    ),
                    value,
                )
            )

        return lens_runs

    def get_top_k_labels(self, activations: Float[Tensor, "comp"], k=10):
        features = self.get_top_k_features(activations, k=k)

        return [
            f"{kind} | Layer: {layer} | Feature: {feature_i} | Contribution: {value*100:.3g}%"
            for kind, layer, feature_i, value in features
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


ComponentLensWithValue = Tuple["ComponentLens", float]


@dataclass
class ComponentLens:
    circuit_lens: "CircuitLens"
    component: str
    circuit_lens: "CircuitLens"
    component: str
    run_data: Dict[str, Any]

    @property
    def tuple_id(self):
        if self.component == CircuitComponent.UNEMBED:
            return (
                self.component,
                self.run_data["seq_index"],
                self.run_data["token_id"],
            )
        elif self.component == CircuitComponent.UNEMBED_AT_TOKEN:
            return (self.component, self.run_data["seq_index"])
        elif self.component == CircuitComponent.ATTN_HEAD:
            return (
                self.component,
                self.run_data["layer"],
                self.run_data["head"],
                self.run_data["source_index"],
                self.run_data["destination_index"],
                self.run_data["feature"],
            )
        else:
            return (
                self.component,
                self.run_data["layer"],
                self.run_data["seq_index"],
                self.run_data["feature"],
            )

    @classmethod
    def create_unembed_at_token_lens(cls, circuit_lens: "CircuitLens", seq_index):
        return cls(
            circuit_lens=circuit_lens,
            component=CircuitComponent.UNEMBED_AT_TOKEN,
            run_data={
                "seq_index": seq_index,
            },
        )

    @classmethod
    def create_unembed_lens(cls, circuit_lens: "CircuitLens", seq_index, token):
        if isinstance(token, str):
            token = circuit_lens.model.to_single_token(token)

        return cls(
            circuit_lens=circuit_lens,
            component=CircuitComponent.UNEMBED,
            run_data={"seq_index": seq_index, "token_id": token},
        )

    def __str__(self):
        if self.component == CircuitComponent.UNEMBED:
            return f"Unembed | Token: '{self.circuit_lens.model.tokenizer.decode([self.run_data['token']])}' :: {self.run_data['token']} | Seq Index: {self.run_data['seq_index']}"
        elif self.component == CircuitComponent.UNEMBED_AT_TOKEN:
            seq_index = self.run_data["seq_index"]

            seq_index = self.circuit_lens.process_seq_index(seq_index)

            token_i = self.circuit_lens.tokens[0, seq_index + 1].item()

            return f"Unembed at Token | Token: '{self.circuit_lens.model.tokenizer.decode([token_i])}' :: {token_i} | Seq Index: {seq_index}"
        elif self.component == CircuitComponent.Z_FEATURE:
            return f"Z Feature | Feature: {self.run_data['feature']} |Layer: {self.run_data['layer']} | Seq Index: {self.run_data['seq_index']}"
        elif self.component == CircuitComponent.ATTN_HEAD:
            return f"Head | Layer: {self.run_data['layer']} | Head: {self.run_data['head']} | Source: {self.run_data['source_index']} | Destination: {self.run_data['destination_index']}"
        elif self.component == CircuitComponent.MLP_FEATURE:
            return f"MLP Lens | Layer: {self.run_data['layer']} | Seq Index: {self.run_data['seq_index']} | Feature: {self.run_data['feature']}"
        else:
            return f"{self.component} | Layer: {self.run_data['layer']} | Seq Index: {self.run_data['seq_index']} | Feature: {self.run_data['feature']}"

    def __repr__(self):
        return str(self)

    @property
    def feature(self):
        return self.run_data.get("feature", -1)

    def __call__(
        self, head_type=None, **kwargs
    ) -> Tuple[List[ComponentLensWithValue], Optional[ActiveFeatures]]:
        if self.component == CircuitComponent.UNEMBED:
            return self.circuit_lens.get_unembed_lens(
                self.run_data["token_id"], self.run_data["seq_index"], **kwargs
            )
        elif self.component == CircuitComponent.UNEMBED_AT_TOKEN:
            seq_index = self.run_data["seq_index"]
            seq_index = self.circuit_lens.process_seq_index(seq_index)
            token_i = self.circuit_lens.tokens[0, seq_index + 1].item()

            return self.circuit_lens.get_unembed_lens(
                int(token_i), self.run_data["seq_index"], **kwargs
            )
        elif self.component == CircuitComponent.Z_FEATURE:
            return (
                self.circuit_lens.get_head_seq_lens_for_z_feature(
                    self.run_data["layer"],
                    self.run_data["seq_index"],
                    self.run_data["feature"],
                    **kwargs,
                ),
                None,
            )
        elif self.component == CircuitComponent.Z_SAE_ERROR:
            return (
                self.circuit_lens.get_head_seq_lens_for_z_error(
                    self.run_data["layer"],
                    self.run_data["seq_index"],
                    **kwargs,
                ),
                None,
            )

        elif self.component == CircuitComponent.ATTN_HEAD:
            if head_type is None:
                head_type = "q"

            if head_type == "q":
                return self.circuit_lens.get_q_lens_on_head_seq(
                    self.run_data["layer"],
                    self.run_data["head"],
                    self.run_data["source_index"],
                    self.run_data["destination_index"],
                    **kwargs,
                )
            elif head_type == "k":
                return self.circuit_lens.get_k_lens_on_head_seq(
                    self.run_data["layer"],
                    self.run_data["head"],
                    self.run_data["source_index"],
                    self.run_data["destination_index"],
                    **kwargs,
                )
            elif head_type == "v":
                return self.circuit_lens.get_v_lens_at_seq(
                    self.run_data["layer"],
                    self.run_data["head"],
                    self.run_data["source_index"],
                    self.run_data["feature"],
                    **kwargs,
                )
        elif self.component == CircuitComponent.MLP_FEATURE:
            return self.circuit_lens.get_mlp_feature_lens_at_seq(
                self.run_data["layer"],
                self.run_data["seq_index"],
                self.run_data["feature"],
                **kwargs,
            )

        return [], None


model_encoder_cache: Optional[Tuple[HookedTransformer, Any, Any]] = None


def get_model_encoders(device):
    global model_encoder_cache

    if model_encoder_cache is not None:
        return model_encoder_cache

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    print()
    print("Loading SAEs...")
    z_saes = [ZSAE.load_zsae_for_layer(i) for i in trange(model.cfg.n_layers)]

    print()
    print("Loading Transcoders...")
    mlp_transcoders = [
        SparseTranscoder.load_from_hugging_face(i) for i in trange(model.cfg.n_layers)
    ]

    model_encoder_cache = (model, z_saes, mlp_transcoders)

    return model_encoder_cache


class CircuitLens:
    def __init__(self, prompt):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        (model, z_saes, mlp_transcoders) = get_model_encoders(self.device)

        self.z_saes = z_saes
        self.mlp_transcoders = mlp_transcoders
        self.model: HookedTransformer = model

        self.prompt = prompt
        self.tokens = self.model.to_tokens(prompt)

        if self.tokens.size(0) != 1:
            raise ValueError("Can only do CircuitLens on a single prompt!")

        self.logits, self.cache = self.model.run_with_cache(
            self.tokens, return_type="logits"
        )

        self._seq_activations = {}

    @property
    def n_tokens(self):
        return self.tokens.size(1)

    def normalize_activations(self, activations):
        # activations -= activations.min()

        return activations / activations.sum()

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

        for layer in range(self.model.cfg.n_layers):
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

            z_contributions = z_sae.W_dec[z_max_features] * z_values.unsqueeze(-1)

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

            mlp_winner_count = mlp_acts.flatten().nonzero().numel()

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
        active_features: ActiveFeatures,
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

        return (
            active_features.get_top_k_lens_runs(activations, self, seq_index, k=k),
            active_features,
        )

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

        activations = einops.einsum(
            active_features.vectors,
            self.model.W_U[:, token_i],
            "comp d_model, d_model -> comp",
        )

        # activations /= activations.sum()
        activations = self.normalize_activations(activations)

        return self.get_next_lens_runs(
            active_features=active_features,
            activations=activations,
            title=f"Unembed Lens for token '{self.model.tokenizer.decode([token_i])}'/{token_i}",
            seq_index=seq_index,
            visualize=visualize,
            k=k,
        )

    def get_head_seq_activations_for_vector(
        self, layer: int, seq_index: int, vector: Float[Tensor, "n_head d_head"]
    ):
        seq_index = self.process_seq_index(seq_index)
        v = self.cache["v", layer]
        pattern = self.cache["pattern", layer]

        pre_z = einops.einsum(
            v,
            pattern,
            "b p_seq n_head d_head, b n_head seq p_seq -> seq b p_seq n_head d_head",
        )[seq_index, 0]

        # better_w_enc = einops.rearrange(
        #     encoder.W_enc, "(n_head d_head) feature -> n_head d_head feature", n_head=12
        # )[:, :, feature]

        feature_act = einops.einsum(
            pre_z, vector, "seq n_head d_head, n_head d_head -> n_head seq"
        )

        return einops.rearrange(feature_act, "n_head seq -> (n_head seq)")

    def get_head_seq_activations_for_z_feature(
        self, layer: int, seq_index: int, feature: int
    ):
        encoder = self.z_saes[layer]

        better_w_enc = einops.rearrange(
            encoder.W_enc, "(n_head d_head) feature -> n_head d_head feature", n_head=12
        )[:, :, feature]

        return self.get_head_seq_activations_for_vector(layer, seq_index, better_w_enc)

        # seq_index = self.process_seq_index(seq_index)
        # v = self.cache["v", layer]
        # pattern = self.cache["pattern", layer]
        # encoder = self.z_saes[layer]

        # pre_z = einops.einsum(
        #     v,
        #     pattern,
        #     "b p_seq n_head d_head, b n_head seq p_seq -> seq b p_seq n_head d_head",
        # )[seq_index, 0]

        # better_w_enc = einops.rearrange(
        #     encoder.W_enc, "(n_head d_head) feature -> n_head d_head feature", n_head=12
        # )[:, :, feature]

        # feature_act = einops.einsum(
        #     pre_z, better_w_enc, "seq n_head d_head, n_head d_head -> n_head seq"
        # )

        # return einops.rearrange(feature_act, "n_head seq -> (n_head seq)")

    def get_head_seq_lens_for_activations(
        self,
        feature_act,
        layer: int,
        seq_index: int,
        feature: int,
        visualize=True,
        k=10,
    ) -> List[ComponentLensWithValue]:
        """
        If `feature` is -1, then we're analyzing the SAE error
        """
        # feature_act = self.get_head_seq_activations_for_z_feature(
        #     layer, seq_index, feature
        # )

        # feature_act /= feature_act.sum()
        feature_act = self.normalize_activations(feature_act)

        values, indices = feature_act.topk(k=k)

        lens_runs = []
        vis_list = []

        for index, value in zip(indices.tolist(), values.tolist()):
            head = index // self.n_tokens
            source = index % self.n_tokens

            lens_runs.append(
                (
                    ComponentLens(
                        circuit_lens=self,
                        component=CircuitComponent.ATTN_HEAD,
                        run_data={
                            "layer": layer,
                            "head": head,
                            "source_index": source,
                            "feature": feature,
                            "destination_index": seq_index,
                        },
                    ),
                    value,
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
                title=f"Layer {layer} Head/Seq {f'Feature {feature}' if feature >= 0 else 'Z Error'} at '{self.get_str_token_at_seq(seq_index)}'::{seq_index}",
                labels={"x": "Token", "y": "Head"},
            )

            pprint(
                [
                    f"Head {head} "
                    + f"| Source: {self.model.tokenizer.decode([self.tokens[0, source]])}::{source} "
                    + f"| Destination: {self.model.tokenizer.decode([self.tokens[0, dest]])}::{dest} "
                    + f"| Contribution: {value * 100:.3g}%"
                    for (head, source, dest, value) in vis_list
                ]
            )

        return lens_runs

    def get_head_seq_lens_for_z_feature(
        self, layer: int, seq_index: int, feature: int, visualize=True, k=10
    ) -> List[ComponentLensWithValue]:

        feature_act = self.get_head_seq_activations_for_z_feature(
            layer, seq_index, feature
        )

        return self.get_head_seq_lens_for_activations(
            feature_act, layer, seq_index, feature, visualize=visualize, k=k
        )

    def get_head_seq_lens_for_z_error(
        self, layer: int, seq_index: int, visualize=True, k=10
    ):
        seq_index = self.process_seq_index(seq_index)

        error_vector = self.get_active_features(layer).get_z_error(layer)
        error_vector = einops.rearrange(
            error_vector, "(n_head d_head) -> n_head d_head", n_head=12
        )

        feature_act = self.get_head_seq_activations_for_vector(
            layer, seq_index, error_vector
        )

        return self.get_head_seq_lens_for_activations(
            feature_act, layer, seq_index, -1, visualize=visualize, k=k
        )

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

        activation = self.normalize_activations(activation)

        return self.get_next_lens_runs(
            active_features,
            activation,
            seq_index,
            title=f"MLP Lens for Feature {feature} at '{self.get_str_token_at_seq(seq_index)}'/{seq_index}",
            visualize=visualize,
            k=k,
        )

    def get_str_token_at_seq(self, seq_index: int):
        seq_index = self.process_seq_index(seq_index)
        return self.model.tokenizer.decode([self.tokens[0, seq_index]])

    def get_v_lens_activations(
        self,
        layer: int,
        head,
        seq_index: int,
        feature: int,
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

        return activation

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

        activation = self.get_v_lens_activations(layer, head, seq_index, feature)

        activation = self.normalize_activations(activation)

        return self.get_next_lens_runs(
            active_features,
            activation,
            seq_index,
            title=f"V Lens | Layer {layer} | Head {head} | Feature {feature} at  '{self.get_str_token_at_seq(seq_index)}'/{seq_index}",
            visualize=visualize,
            k=k,
        )

    def get_q_lens_activations(
        self,
        layer: int,
        head: int,
        source_index,
        destination_index,
    ):
        source_index = self.process_seq_index(source_index)
        destination_index = self.process_seq_index(destination_index)

        active_features = self.get_active_features(destination_index)

        vectors = active_features.get_vectors_before_comp("attn", layer)

        effective_k = self.cache["k", layer][0, source_index, head]

        activation = einops.einsum(
            vectors,
            self.model.W_Q[layer, head],
            effective_k,
            "comp d_model, d_model d_head, d_head -> comp",
        )

        return activation

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

        activation = self.get_q_lens_activations(
            layer, head, source_index, destination_index
        )

        activation = self.normalize_activations(activation)

        return self.get_next_lens_runs(
            active_features,
            activation,
            destination_index,
            title=f"Q Lens | Layer {layer} | Head {head} | '{self.get_str_token_at_seq(source_index)}'/{source_index} -> '{self.get_str_token_at_seq(destination_index)}'/{destination_index}",
            visualize=visualize,
            k=k,
        )

    def get_k_lens_activations(
        self,
        layer: int,
        head: int,
        source_index,
        destination_index,
    ):
        source_index = self.process_seq_index(source_index)
        destination_index = self.process_seq_index(destination_index)

        active_features = self.get_active_features(source_index)

        vectors = active_features.get_vectors_before_comp("attn", layer)
        effective_q = self.cache["q", layer][0, destination_index, head]

        activation = einops.einsum(
            vectors,
            self.model.W_K[layer, head],
            effective_q,
            "comp d_model, d_model d_head, d_head -> comp",
        )

        return activation

    def get_k_lens_on_head_seq(
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

        active_features = self.get_active_features(source_index)

        activation = self.get_k_lens_activations(
            layer, head, source_index, destination_index
        )

        activation = self.normalize_activations(activation)

        return self.get_next_lens_runs(
            active_features,
            activation,
            source_index,
            title=f"K Lens | Layer {layer} | Head {head} | '{self.get_str_token_at_seq(source_index)}'/{source_index} -> '{self.get_str_token_at_seq(destination_index)}'/{destination_index}",
            visualize=visualize,
            k=k,
        )

    def get_transcoder_ixg(
        self,
        transcoder,
        layer,
        seq_index,
        feature_vector,
        is_transcoder_post_ln=True,
        return_feature_activs=True,
    ):
        """
        Pull back the contributions from the transcoder's output to the inputs.
        """
        # Perform the matrix multiplication with the decoder weights
        pulledback_feature = transcoder.W_dec @ feature_vector

        # Determine the correct activation name based on whether layer normalization is applied
        act_name = (
            ("normalized", layer, "ln2")
            if is_transcoder_post_ln
            else ("resid_mid", layer)
        )

        # Retrieve the activations from the cache
        feature_activs = transcoder(self.cache[act_name])[1][0, seq_index]

        # Multiply pulledback_feature by the feature activations
        pulledback_feature *= feature_activs

        # Return the pulledback_feature and feature_activs
        if not return_feature_activs:
            return pulledback_feature
        else:
            return pulledback_feature, feature_activs

    def get_ln_constant(self, vector, layer, token, is_ln2=False, recip=False):
        x_act_name = ("resid_mid", layer) if is_ln2 else ("resid_pre", layer)
        y_act_name = (
            ("normalized", layer, "ln2") if is_ln2 else ("normalized", layer, "ln1")
        )

        x = self.cache[x_act_name][0, token]
        y = self.cache[y_act_name][0, token]

        if torch.dot(vector, x) == 0:
            return torch.tensor(0.0, device=vector.device)
        return (
            torch.dot(vector, y) / torch.dot(vector, x)
            if not recip
            else torch.dot(vector, x) / torch.dot(vector, y)
        )

    def get_attn_head_contribs(
        self, layer: int, seq_index: int, feature_vector: torch.Tensor
    ):
        z_sae = self.z_saes[layer]
        layer_z = einops.rearrange(
            self.cache["z", layer][0, seq_index], "n_heads d_head -> (n_heads d_head)"
        )

        # Get z_acts similar to how it's done in your current code
        _, z_recon, z_acts, _, _ = z_sae(layer_z)

        # Compute z_error and z_bias
        z_error = layer_z - z_recon
        z_bias = z_sae.b_dec

        # Stack z_error and z_bias, then rearrange
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
        return z_residual_vectors

    def get_transcoder_contribs(
        self, layer: int, seq_index: int, feature_vector: torch.Tensor, k=5
    ):
        transcoder = self.mlp_transcoders[layer]
        is_transcoder_post_ln = "ln2" in transcoder.cfg.hook_point
        act_name = (
            ("normalized", layer, "ln2")
            if is_transcoder_post_ln
            else ("resid_mid", layer)
        )

        transcoder_out = transcoder(self.cache[act_name])[0][0, seq_index]
        mlp_out = self.model.blocks[layer].mlp(self.cache[act_name])[0, seq_index]

        # Reshape feature_vector to match mlp_out and transcoder_out for dot product
        if (
            feature_vector.dim() == 2
        ):  # When feature_vector is 2D (e.g., attention heads)
            feature_vector = feature_vector.view(-1)

        error = torch.dot(feature_vector, mlp_out - transcoder_out) / torch.dot(
            feature_vector, mlp_out
        )

        pulledback_feature, feature_activs = self.get_transcoder_ixg(
            transcoder, layer, seq_index, feature_vector
        )
        top_contribs, top_indices = torch.topk(pulledback_feature, k=k)

        top_contribs_list = []
        for contrib, index in zip(top_contribs, top_indices):
            vector = transcoder.W_enc[:, index]
            vector = vector * (transcoder.W_dec @ feature_vector)[index]
            if is_transcoder_post_ln:
                vector *= self.get_ln_constant(vector, layer, seq_index)

            top_contribs_list.append(
                (vector, layer, seq_index, index.item(), contrib.item())
            )
        return top_contribs_list

    def get_top_contribs(
        self, feature_vector: torch.Tensor, layer: int, seq_index: int, k=5
    ):
        all_mlp_contribs = []
        for cur_layer in range(layer + 1):
            all_mlp_contribs += self.get_transcoder_contribs(
                cur_layer, seq_index, feature_vector, k=k
            )

        all_attn_contribs = []
        for cur_layer in range(layer + 1):
            attn_contribs = self.get_attn_head_contribs(
                cur_layer, seq_index, feature_vector
            )
            top_attn_contribs_flattened, top_attn_contrib_indices_flattened = (
                torch.topk(attn_contribs.flatten(), k=min(k, len(attn_contribs)))
            )
            top_attn_contrib_indices = torch.unravel_index(
                top_attn_contrib_indices_flattened, attn_contribs.shape
            )

            print(f"Top attn contribs flattened: {top_attn_contribs_flattened}")
            print(f"Top attn contrib indices: {top_attn_contrib_indices}")

            for contrib, (winner, head, src_token) in zip(
                top_attn_contribs_flattened, zip(*top_attn_contrib_indices)
            ):
                vector = self.model.OV[cur_layer, head] @ feature_vector
                attn_pattern = self.cache["pattern", cur_layer]
                vector *= attn_pattern[0, head, seq_index, src_token]
                vector *= self.get_ln_constant(vector, cur_layer, src_token)

                all_attn_contribs.append(
                    (vector, cur_layer, src_token, head, contrib.item())
                )

        all_contribs = all_mlp_contribs + all_attn_contribs
        all_contrib_scores = torch.tensor([x[4] for x in all_contribs])
        _, top_contrib_indices = torch.topk(
            all_contrib_scores, k=min(k, len(all_contrib_scores))
        )
        return [all_contribs[i.item()] for i in top_contrib_indices]
