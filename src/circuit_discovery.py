import torch
import einops
import circuitsvis as cv

from typing import Union, Optional, List, Dict, Tuple, Set, Callable
from circuit_lens import (
    CircuitLens,
    ComponentLens,
    CircuitComponent,
    ComponentLensWithValue,
)
from graphviz import Digraph
from functools import partial
from transformer_lens.hook_points import HookPoint
from ranking_utils import visualize_top_tokens
from jaxtyping import Float
from IPython.display import HTML

from abc import ABC


class AnalysisObject:
    circuit_discovery: "CircuitDiscovery"

    def __init__(self, circuit_discovery: "CircuitDiscovery"):
        self.circuit_discovery = circuit_discovery

    @property
    def n_tokens(self):
        return self.circuit_discovery.n_tokens

    @property
    def model(self):
        return self.circuit_discovery.model


class BasicAnalysisNode(AnalysisObject):
    def __init__(self, transformer_layer: "TransformerAnalysisLayer"):
        super().__init__(transformer_layer.circuit_discovery)

        self.transformer_layer = transformer_layer
        self.feature_nodes: Dict[int, "CircuitDiscoveryNode"] = {}

    def get_discovery_node_for_feature(
        self, feature: int, lens: Optional[ComponentLens]
    ):
        if feature not in self.feature_nodes:
            if lens is None:
                raise ValueError("Must provide lens for z feature node creation")
            self.feature_nodes[feature] = CircuitDiscoveryRegularNode(
                lens, self.transformer_layer.circuit_discovery
            )

        return self.feature_nodes[feature]

    def get_discovery_node_for_attribute(
        self, attribute: str, lens: Optional[ComponentLens]
    ):
        if getattr(self, attribute) is None:
            if lens is None:
                raise ValueError("Must provide lens for attribute node creation")
            setattr(
                self,
                attribute,
                CircuitDiscoveryRegularNode(
                    lens, self.transformer_layer.circuit_discovery
                ),
            )

        return getattr(self, attribute)


class MlpAnalysisNode(BasicAnalysisNode):
    transcoder_error: Optional["CircuitDiscoveryNode"] = None
    transcoder_bias: Optional["CircuitDiscoveryNode"] = None

    def __init__(self, transformer_layer: "TransformerAnalysisLayer"):
        super().__init__(transformer_layer)

    def get_transcoder_error_discovery_node(self, lens: Optional[ComponentLens]):
        return self.get_discovery_node_for_attribute("transcoder_error", lens)

    def get_transcoder_bias_discovery_node(self, lens: Optional[ComponentLens]):
        return self.get_discovery_node_for_attribute("transcoder_bias", lens)


class ZFeatureAnalysisNode(BasicAnalysisNode):
    sae_error: Optional["CircuitDiscoveryNode"] = None
    sae_bias: Optional["CircuitDiscoveryNode"] = None
    bias_o: Optional["CircuitDiscoveryNode"] = None

    def __init__(self, transformer_layer: "TransformerAnalysisLayer"):
        super().__init__(transformer_layer)

    def get_z_sae_error_discovery_node(self, lens: Optional[ComponentLens]):
        return self.get_discovery_node_for_attribute("sae_error", lens)

    def get_z_sae_bias_discovery_node(self, lens: Optional[ComponentLens]):
        return self.get_discovery_node_for_attribute("sae_bias", lens)

    def get_bias_o_discovery_node(self, lens: Optional[ComponentLens]):
        return self.get_discovery_node_for_attribute("bias_o", lens)


class AttnHeadAnalysisNode(AnalysisObject):
    def __init__(self, attn_head_layer: "TransformerAnalysisLayer"):
        super().__init__(attn_head_layer.circuit_discovery)

        self.attn_head_layer = attn_head_layer

        self.source_dest_feature_nodes: Dict[
            Tuple[int, int], Dict[int, "CircuitDiscoveryNode"]
        ] = {}

    def get_discovery_node_for_source_dest_feature(
        self, source: int, dest: int, feature: int, lens: Optional[ComponentLens] = None
    ):
        if (source, dest) not in self.source_dest_feature_nodes:
            if lens is None:
                raise ValueError("Must provide lens for z feature node creation")

            self.source_dest_feature_nodes[(source, dest)] = {}

        if feature not in self.source_dest_feature_nodes[(source, dest)]:
            if lens is None:
                raise ValueError("Must provide lens for z feature node creation")
            self.source_dest_feature_nodes[(source, dest)][feature] = (
                CircuitDiscoveryHeadNode(lens, self.attn_head_layer.circuit_discovery)
            )

        return self.source_dest_feature_nodes[(source, dest)][feature]


class TransformerAnalysisLayer(AnalysisObject):
    def __init__(self, transformer_model: "TransformerAnalysisModel"):
        super().__init__(transformer_model.circuit_discovery)

        self.transformer_model = transformer_model

        self.mlps_at_seq = [MlpAnalysisNode(self) for _ in range(self.n_tokens)]
        self.z_features_at_seq = [
            ZFeatureAnalysisNode(self) for _ in range(self.n_tokens)
        ]

        self.attn_heads = [
            AttnHeadAnalysisNode(self) for _ in range(self.model.cfg.n_heads)
        ]


class ComponentEdgeTracker:
    def __init__(self, transformer_model: "TransformerAnalysisModel"):
        self._reciever_to_contributors: Dict[Tuple, Set[Tuple]] = {}
        self._contributor_to_recievers: Dict[Tuple, Set[Tuple]] = {}

        self.transformer_model = transformer_model

    def add_edge(self, reciever: Tuple, contributor: Tuple):
        if reciever not in self._reciever_to_contributors:
            self._reciever_to_contributors[reciever] = set()
        if contributor not in self._contributor_to_recievers:
            self._contributor_to_recievers[contributor] = set()

        self._reciever_to_contributors[reciever].add(contributor)
        self._contributor_to_recievers[contributor].add(reciever)

    def remove_edge(self, reciever: Tuple, contributor: Tuple):
        if reciever in self._reciever_to_contributors:
            self._reciever_to_contributors[reciever].discard(contributor)
        if contributor in self._contributor_to_recievers:
            self._contributor_to_recievers[contributor].discard(reciever)

    def get_contributors(self, reciever: Tuple) -> List[Tuple]:
        return list(self._reciever_to_contributors.get(reciever, set()))

    def clear(self):
        self._reciever_to_contributors = {}
        self._contributor_to_recievers = {}


class TransformerAnalysisModel(AnalysisObject):
    def __init__(self, circuit_discovery: "CircuitDiscovery"):
        super().__init__(circuit_discovery)

        self.layers = [
            TransformerAnalysisLayer(self) for _ in range(self.model.cfg.n_layers)
        ]

        self.discovery_node_cache: Dict[Tuple, "CircuitDiscoveryNode"] = {}

        self.embed_nodes: Dict[int, "CircuitDiscoveryNode"] = {}
        self.pos_embed_nodes: Dict[int, "CircuitDiscoveryNode"] = {}

        self.unembed_nodes: Dict[int, "CircuitDiscoveryNode"] = {}

        self.edge_tracker = ComponentEdgeTracker(self)

    def add_contributor_edge(
        self, reciever_id: Tuple, contributor_lens: ComponentLens
    ) -> "CircuitDiscoveryNode":
        contributor_node = self.get_discovery_node_at_locator(contributor_lens)

        self.edge_tracker.add_edge(reciever_id, contributor_node.tuple_id)

        return contributor_node

    def get_contributors_in_graph(
        self, reciever_id: Tuple
    ) -> List["CircuitDiscoveryNode"]:
        return [
            self.get_discovery_node_at_locator(tuple_id)
            for tuple_id in self.edge_tracker.get_contributors(reciever_id)
        ]

    def _cache_discovery_node(
        self,
        tuple_id: Tuple,
        discovery_node: "CircuitDiscoveryNode",
    ):
        self.discovery_node_cache[tuple_id] = discovery_node

        return discovery_node

    def get_discovery_node_at_locator(
        self, locator: Union[ComponentLens, Tuple]
    ) -> "CircuitDiscoveryNode":
        if isinstance(locator, ComponentLens):
            lens = locator
            tuple_id: Tuple = lens.tuple_id
        else:
            lens = None
            tuple_id: Tuple = locator

        component = tuple_id[0]

        if component == CircuitComponent.ATTN_HEAD:
            tuple_id = CircuitDiscoveryHeadNode.process_tuple_id(tuple_id)

        if tuple_id in self.discovery_node_cache:
            return self.discovery_node_cache[tuple_id]

        if component == CircuitComponent.UNEMBED:
            raise NotImplementedError("Haven't implemented generic unembed yet!")
        elif component == CircuitComponent.UNEMBED_AT_TOKEN:
            seq_index = tuple_id[1]

            if seq_index not in self.unembed_nodes:
                if lens is None:
                    raise ValueError("Must provide lens for unembed node creation")
                self.unembed_nodes[seq_index] = CircuitDiscoveryRegularNode(
                    lens, self.circuit_discovery
                )

            return self._cache_discovery_node(tuple_id, self.unembed_nodes[seq_index])
        elif component == CircuitComponent.EMBED:
            seq_index = tuple_id[2]

            if seq_index not in self.embed_nodes:
                if lens is None:
                    raise ValueError("Must provide lens for embed node creation")
                self.embed_nodes[seq_index] = CircuitDiscoveryRegularNode(
                    lens, self.circuit_discovery
                )

            return self._cache_discovery_node(tuple_id, self.embed_nodes[seq_index])
        elif component == CircuitComponent.POS_EMBED:
            seq_index = tuple_id[2]

            if seq_index not in self.pos_embed_nodes:
                if lens is None:
                    raise ValueError("Must provide lens for pos embed node creation")
                self.pos_embed_nodes[seq_index] = CircuitDiscoveryRegularNode(
                    lens, self.circuit_discovery
                )

            return self._cache_discovery_node(tuple_id, self.pos_embed_nodes[seq_index])
        elif component == CircuitComponent.ATTN_HEAD:
            _, layer, head, source, dest, feature = tuple_id

            return self._cache_discovery_node(
                tuple_id,
                self.layers[int(layer)]
                .attn_heads[int(head)]
                .get_discovery_node_for_source_dest_feature(
                    source, dest, feature, lens
                ),
            )
        elif component == CircuitComponent.Z_FEATURE:
            _, layer, seq_index, feature = tuple_id
            return self._cache_discovery_node(
                tuple_id,
                (
                    self.layers[int(layer)]
                    .z_features_at_seq[int(seq_index)]
                    .get_discovery_node_for_feature(feature, lens)
                ),
            )
        elif component == CircuitComponent.Z_SAE_BIAS:
            _, layer, seq_index, *_ = tuple_id

            return self._cache_discovery_node(
                tuple_id,
                (
                    self.layers[int(layer)]
                    .z_features_at_seq[int(seq_index)]
                    .get_z_sae_bias_discovery_node(lens)
                ),
            )
        elif component == CircuitComponent.Z_SAE_ERROR:
            _, layer, seq_index, *_ = tuple_id

            return self._cache_discovery_node(
                tuple_id,
                (
                    self.layers[int(layer)]
                    .z_features_at_seq[int(seq_index)]
                    .get_z_sae_error_discovery_node(lens)
                ),
            )
        elif component == CircuitComponent.BIAS_O:
            _, layer, seq_index, *_ = tuple_id

            return self._cache_discovery_node(
                tuple_id,
                (
                    self.layers[int(layer)]
                    .z_features_at_seq[int(seq_index)]
                    .get_bias_o_discovery_node(lens)
                ),
            )
        elif component == CircuitComponent.MLP_FEATURE:
            _, layer, seq_index, feature = tuple_id
            return self._cache_discovery_node(
                tuple_id,
                (
                    self.layers[int(layer)]
                    .mlps_at_seq[int(seq_index)]
                    .get_discovery_node_for_feature(feature, lens)
                ),
            )
        elif component == CircuitComponent.TRANSCODER_BIAS:
            _, layer, seq_index, *_ = tuple_id
            return self._cache_discovery_node(
                tuple_id,
                (
                    self.layers[int(layer)]
                    .mlps_at_seq[int(seq_index)]
                    .get_transcoder_bias_discovery_node(lens)
                ),
            )
        elif component == CircuitComponent.TRANSCODER_ERROR:
            _, layer, seq_index, *_ = tuple_id
            return self._cache_discovery_node(
                tuple_id,
                (
                    self.layers[int(layer)]
                    .mlps_at_seq[int(seq_index)]
                    .get_transcoder_error_discovery_node(lens)
                ),
            )

        raise ValueError(f"Unknown component type: {component}")


class CircuitDiscoveryNode:
    def __init__(
        self, component_lens: ComponentLens, circuit_discovery: "CircuitDiscovery"
    ):
        self.circuit_discovery = circuit_discovery
        self.component_lens = component_lens

    @property
    def transformer_model(self):
        return self.circuit_discovery.transformer_model

    @property
    def k(self):
        return self.circuit_discovery.k

    @property
    def tuple_id(self):
        return self.component_lens.tuple_id

    @property
    def component(self):
        return self.tuple_id[0]


class CircuitDiscoveryRegularNode(CircuitDiscoveryNode):
    def __init__(
        self, component_lens: ComponentLens, circuit_discovery: "CircuitDiscovery"
    ):
        super().__init__(component_lens, circuit_discovery)

    _top_k_contributors: Optional[List[ComponentLensWithValue]] = None

    @property
    def layer(self):
        return self.component_lens.run_data.get("layer", 0)

    @property
    def top_k_contributors(self) -> List[ComponentLensWithValue]:
        if self._top_k_contributors is None:
            self._top_k_contributors = self.component_lens(visualize=False, k=self.k)

        return self._top_k_contributors

    _top_k_allowed_contributors: Optional[List[ComponentLensWithValue]] = None

    @property
    def top_k_allowed_contributors(self) -> List[ComponentLensWithValue]:
        if self._top_k_allowed_contributors is None:
            self._top_k_allowed_contributors = [
                contributor
                for contributor in self.top_k_contributors
                if self.circuit_discovery.allowed_components_filter(
                    contributor[0].component
                )
            ]

        return self._top_k_allowed_contributors

    @property
    def contributors_in_graph(self) -> List["CircuitDiscoveryNode"]:
        return self.transformer_model.get_contributors_in_graph(self.tuple_id)

    def add_contributor_edge(
        self, component_lens: ComponentLens
    ) -> "CircuitDiscoveryNode":
        return self.transformer_model.add_contributor_edge(
            self.tuple_id, component_lens
        )

    def get_top_unused_contributors(self) -> List[ComponentLensWithValue]:
        contributors_in_graph = self.contributors_in_graph

        contributor_ids = [c.tuple_id for c in contributors_in_graph]

        return [
            new_contributor
            for new_contributor in self.top_k_allowed_contributors
            if new_contributor[0].tuple_id not in contributor_ids
        ]


class CircuitDiscoveryHeadNode(CircuitDiscoveryNode):
    def __init__(
        self, component_lens: ComponentLens, circuit_discovery: "CircuitDiscovery"
    ):
        super().__init__(component_lens, circuit_discovery)

        self._top_k_allowed_contributors: Dict[str, List[ComponentLensWithValue]] = {}
        self._top_k_contributors: Dict[str, List[ComponentLensWithValue]] = {}

    @property
    def layer(self):
        return self.component_lens.run_data["layer"]

    @property
    def head(self):
        return self.component_lens.run_data["head"]

    @property
    def source(self):
        return self.component_lens.run_data["source_index"]

    @property
    def dest(self):
        return self.component_lens.run_data["destination_index"]

    def _validate_head_type(self, head_type: str):
        assert head_type in ["q", "k", "v"]

    def top_k_contributors(self, head_type) -> List[ComponentLensWithValue]:
        self._validate_head_type(head_type)

        if head_type not in self._top_k_contributors:
            self._top_k_contributors[head_type] = self.component_lens(
                head_type=head_type, visualize=False, k=self.k
            )

        return self._top_k_contributors[head_type]

    def top_k_allowed_contributors(
        self, head_type: str
    ) -> List[ComponentLensWithValue]:
        self._validate_head_type(head_type)

        if head_type not in self._top_k_allowed_contributors:
            self._top_k_allowed_contributors[head_type] = [
                contributor
                for contributor in self.top_k_contributors(head_type)
                if self.circuit_discovery.allowed_components_filter(
                    contributor[0].component
                )
            ]

        return self._top_k_allowed_contributors[head_type]

    def contributors_in_graph(self, head_type: str) -> List["CircuitDiscoveryNode"]:
        return self.transformer_model.get_contributors_in_graph(
            self.tuple_id_for_head_type(head_type)
        )

    def get_top_unused_contributors(
        self, head_type: str
    ) -> List[ComponentLensWithValue]:
        self._validate_head_type(head_type)

        contributors_in_graph = self.contributors_in_graph(head_type)
        contributor_ids = [c.tuple_id for c in contributors_in_graph]

        return [
            new_contributor
            for new_contributor in self.top_k_allowed_contributors(head_type)
            if new_contributor[0].tuple_id not in contributor_ids
        ]

    def tuple_id_for_head_type(self, head_type) -> Tuple:
        self._validate_head_type(head_type)

        list_id = list(self.tuple_id)
        list_id.append(head_type)

        return tuple(list_id)

    def add_contributor_edge(
        self, head_type: str, component_lens: ComponentLens
    ) -> "CircuitDiscoveryNode":
        self._validate_head_type(head_type)

        return self.transformer_model.add_contributor_edge(
            self.tuple_id_for_head_type(head_type), component_lens
        )

    @classmethod
    def process_tuple_id(cls, tuple_id):
        if tuple_id[0] != CircuitComponent.ATTN_HEAD:
            return tuple_id

        return tuple_id[:6]


def all_allowed(_: str):
    return True


def only_feature(component: str):
    return component in [
        CircuitComponent.Z_FEATURE,
        CircuitComponent.MLP_FEATURE,
        CircuitComponent.ATTN_HEAD,
        CircuitComponent.UNEMBED,
        # CircuitComponent.UNEMBED_AT_TOKEN,
        CircuitComponent.EMBED,
        CircuitComponent.POS_EMBED,
        # CircuitComponent.BIAS_O,
        # CircuitComponent.Z_SAE_ERROR,
        # CircuitComponent.Z_SAE_BIAS,
        # CircuitComponent.TRANSCODER_ERROR,
        # CircuitComponent.TRANSCODER_BIAS,
    ]


class CircuitDiscovery:
    def __init__(
        self,
        prompt,
        seq_index: int,
        allowed_components_filter: Callable[[str], bool] = all_allowed,
        k=10,
    ):
        self.lens = CircuitLens(prompt)
        self.seq_index = self.lens.process_seq_index(seq_index)
        self.allowed_components_filter = allowed_components_filter

        self.k = k

        self.transformer_model = TransformerAnalysisModel(self)

        self.root_node = self.transformer_model.get_discovery_node_at_locator(
            ComponentLens.create_root_unembed_lens(self.lens, self.seq_index),
        )

    @property
    def model(self):
        return self.lens.model

    @property
    def prompt(self):
        return self.lens.prompt

    @property
    def n_tokens(self):
        return self.lens.n_tokens

    @property
    def tokens(self):
        return self.lens.tokens

    @property
    def str_tokens(self) -> List[str]:
        return self.model.to_str_tokens(self.lens.prompt)  # type: ignore

    @property
    def z_saes(self):
        return self.lens.z_saes

    @property
    def mlp_transcoders(self):
        return self.lens.mlp_transcoders

    def reset_graph(self):
        self.transformer_model.edge_tracker.clear()

    def traverse_graph(
        self, fn: Callable[[CircuitDiscoveryNode], None], print_node_trace=False
    ):
        visited_ids = []

        queue: List[CircuitDiscoveryNode] = [self.root_node]

        while len(queue) > 0:
            node = queue.pop(0)

            if print_node_trace:
                print(node.tuple_id)

            if node.tuple_id in visited_ids:
                continue

            visited_ids.append(node.tuple_id)

            fn(node)

            if isinstance(node, CircuitDiscoveryRegularNode):
                queue.extend(node.contributors_in_graph)

            elif isinstance(node, CircuitDiscoveryHeadNode):
                for head_type in ["q", "k", "v"]:
                    queue.extend(node.contributors_in_graph(head_type))

    def add_greedy_pass(self, contributors_per_node=1):
        visited_ids = []

        queue: List[CircuitDiscoveryNode] = [self.root_node]

        while len(queue) > 0:
            node = queue.pop(0)

            if node.tuple_id in visited_ids:
                continue

            visited_ids.append(node.tuple_id)

            if isinstance(node, CircuitDiscoveryRegularNode):
                top_contribs = node.get_top_unused_contributors()
                if len(top_contribs) == 0:
                    continue

                total_contrib = 0

                for child in top_contribs[:contributors_per_node]:
                    child_discovery_node = node.add_contributor_edge(child[0])

                    total_contrib += child[1]

                    queue.append(child_discovery_node)

                print(node.tuple_id, f"Contrib: {total_contrib*100:.3g}%")
            elif isinstance(node, CircuitDiscoveryHeadNode):
                for head_type in ["q", "k", "v"]:
                    top_contribs = node.get_top_unused_contributors(head_type)
                    if len(top_contribs) == 0:
                        continue

                    total_contrib = 0

                    for child in top_contribs[:contributors_per_node]:
                        child_discovery_node = node.add_contributor_edge(
                            head_type, child[0]
                        )

                        total_contrib += child[1]

                        queue.append(child_discovery_node)

                    print(
                        node.tuple_id_for_head_type(head_type),
                        f"Contrib: {total_contrib:.3g}%",
                    )

    def get_heads_and_mlps_in_current_graph(self) -> Tuple[List[List[int]], List[int]]:
        attn_heads_set = set()
        mlps = set()

        def track_included_heads_and_mlps(
            node: CircuitDiscoveryNode,
            attn_heads_set,
            mlps,
        ):
            if isinstance(node, CircuitDiscoveryHeadNode):
                attn_heads_set.add((node.layer, node.head))
            elif isinstance(node, CircuitDiscoveryRegularNode):
                if node.component == CircuitComponent.MLP_FEATURE:
                    mlps.add(node.layer)

        fn = partial(
            track_included_heads_and_mlps, attn_heads_set=attn_heads_set, mlps=mlps
        )

        self.traverse_graph(fn)

        mlps = list(mlps)
        attn_heads = [[] for _ in range(self.model.cfg.n_layers)]

        for layer, head in attn_heads_set:
            attn_heads[layer].append(head)

        return attn_heads, mlps

    def print_attn_heads_and_mlps_in_graph(self):
        attn_heads, mlps = self.get_heads_and_mlps_in_current_graph()

        for layer in reversed(range(self.model.cfg.n_layers)):
            layer_heads = attn_heads[layer]
            layer_heads.sort()

            if layer in mlps:
                print(f"MLP{layer}")
            if layer_heads:
                print(f"L{layer}H:", layer_heads)

    def visualize_attn_heads_in_graph(self, max_width=800, value_weighted=False):
        _, cache = self.model.run_with_cache(self.tokens)

        attn_heads, _ = self.get_heads_and_mlps_in_current_graph()

        # Create the plotting data
        labels: List[str] = []
        all_patterns: List[Float[torch.Tensor, "dest_pos src_pos"]] = []

        # Assume we have a single batch item
        batch_index = 0

        for layer in range(self.model.cfg.n_layers):
            for head in attn_heads[layer]:
                labels.append(f"L{layer}H{head}")

                # Get the attention patterns for the head
                # Attention patterns have shape [batch, head_index, query_pos, key_pos]
                if value_weighted:
                    value_norms = cache["v", layer][batch_index, :, head].norm(dim=-1)

                    pattern = cache["attn", layer][batch_index][head]

                    all_patterns.append(pattern * value_norms.unsqueeze(0))
                else:
                    all_patterns.append(cache["attn", layer][batch_index, head])

        # Combine the patterns into a single tensor
        patterns: Float[torch.Tensor, "head_index dest_pos src_pos"] = torch.stack(
            all_patterns, dim=0
        )

        # Circuitsvis Plot (note we get the code version so we can concatenate with the title)
        plot = cv.attention.attention_heads(
            attention=patterns, tokens=self.str_tokens, attention_head_names=labels
        ).show_code()

        # Display the title
        title_html = f"<h2>Attention Heads in Graph</h2><br/>"

        # Return the visualisation as raw code
        return HTML(
            f"<div style='max-width: {str(max_width)}px;'>{title_html + plot}</div>"
        )

    def get_logits_for_current_graph(self, **kwargs):
        include_all_mlps = kwargs.get("include_all_mlps", False)
        include_all_heads = kwargs.get("include_all_heads", False)

        head_ablation_style = kwargs.get("head_ablation_style", "bos")

        attn_heads, mlps = self.get_heads_and_mlps_in_current_graph()

        def ablate_z_out(acts, hook: HookPoint, attn_heads):
            layer_heads = attn_heads[hook.layer()]

            if head_ablation_style == "bos":
                bos_ablation = einops.repeat(
                    acts[:, 0, ...], "B ... -> B seq ...", seq=acts.size(1)
                )

            for head in range(self.model.cfg.n_heads):
                if head not in layer_heads:
                    if head_ablation_style == "zero":
                        acts[:, :, head, :] = torch.zeros_like(acts[:, :, head, :])
                    elif head_ablation_style == "bos":
                        acts[:, :, head, :] = bos_ablation[:, :, head, :]

            return acts

        def ablate_mlp_out(acts, hook: HookPoint, mlps):
            if hook.layer() not in mlps:
                acts = torch.zeros_like(acts)

            return acts

        z_hook = partial(ablate_z_out, attn_heads=attn_heads)
        mlp_hook = partial(ablate_mlp_out, mlps=mlps)

        z_filter = lambda n: n.endswith("z")
        mlp_filter = lambda n: n.endswith("mlp_out")

        hooks = []

        if not include_all_heads:
            hooks.append((z_filter, z_hook))

        if not include_all_mlps:
            hooks.append((mlp_filter, mlp_hook))

        logits = self.model.run_with_hooks(
            self.tokens,
            fwd_hooks=hooks,
            return_type="logits",
        )

        return logits[0]

    def visualize_current_graph_performance(self, **kwargs):
        base_logits = self.model(self.tokens, return_type="logits")

        graph_logits = self.get_logits_for_current_graph(**kwargs)

        print("Base Performance:")

        visualize_top_tokens(self.model, self.tokens, base_logits, self.seq_index)

        print()
        print("Current Graph Performance:")

        visualize_top_tokens(self.model, self.tokens, graph_logits, self.seq_index)

    def visualize_graph(self):
        G = Digraph()
        G.graph_attr.update(rankdir="BT", newrank="true")
        G.node_attr.update(
            shape="box", style="rounded", fontsize="10pt", margin="0.01,0.01"
        )
        G.attr(nodesep="0.1")
        # G.attr(fontsize="12pt")
        #    , size="10,10")

        layers = [dict() for _ in range(self.model.cfg.n_layers)]
        embed = dict()

        def add_node_to_record(node: CircuitDiscoveryNode, layers, embed):
            component = node.tuple_id[0]

            if component == CircuitComponent.UNEMBED:
                raise NotImplementedError("Haven't implemented generic unembed yet!")
            if component in [
                CircuitComponent.UNEMBED_AT_TOKEN,
                CircuitComponent.EMBED,
                CircuitComponent.POS_EMBED,
            ]:
                embed.setdefault(component, set()).add(node.tuple_id)

            else:
                layer = node.tuple_id[1]

                layers[layer].setdefault(component, set()).add(node.tuple_id)

        fn = partial(add_node_to_record, layers=layers, embed=embed)

        self.traverse_graph(fn)

        def embed_id(seq_index):
            return str((CircuitComponent.EMBED, 0, seq_index, -1))

        with G.subgraph(name="words") as subgraph:
            subgraph.attr(rank="same")
            for i, str_token in enumerate(self.str_tokens):
                subgraph.node(
                    embed_id(i),
                    label=str_token,
                )

                if i:
                    subgraph.edge(
                        embed_id(i - 1), embed_id(i), style="invis", minlen=".1"
                    )

        pos_embeds = list(embed.get(CircuitComponent.POS_EMBED, set()))

        if pos_embeds:
            with G.subgraph() as subgraph:
                subgraph.attr(rank="same")
                for pos_embed in pos_embeds:
                    seq_index = pos_embed[2]

                    subgraph.node(str(pos_embed), label=f"POS: {seq_index}")

                    G.edge(
                        embed_id(seq_index),
                        str(pos_embed),
                        style="invis",
                        # , minlen=".1"
                    )

        unembeds = list(embed.get(CircuitComponent.UNEMBED_AT_TOKEN, set()))

        if not unembeds:
            raise ValueError("We're trying to do an unembed")

        with G.subgraph(name="unembed") as subgraph:
            subgraph.attr(rank="same")
            for unembed in unembeds:
                seq_index = unembed[1]

                subgraph.node(
                    str(unembed), label=f"UE: '{self.str_tokens[seq_index + 1]}'"
                )

        def component_to_name(comp: str) -> str:
            if comp == CircuitComponent.Z_FEATURE:
                return "Z"
            if comp == CircuitComponent.Z_SAE_ERROR:
                return "Z Err"
            if comp == CircuitComponent.Z_SAE_BIAS:
                return "Z Bias"
            if comp == CircuitComponent.BIAS_O:
                return "b_O"
            if comp == CircuitComponent.MLP_FEATURE:
                return "MLP"
            if comp == CircuitComponent.TRANSCODER_ERROR:
                return "M Err"
            if comp == CircuitComponent.TRANSCODER_BIAS:
                return "M Bias"
            if comp == CircuitComponent.ATTN_HEAD:
                return "Head"

            return ""

        for layer in range(self.model.cfg.n_layers):
            head_ids = list(layers[layer].get(CircuitComponent.ATTN_HEAD, set()))

            if head_ids:
                with G.subgraph() as subgraph:
                    subgraph.attr(rank="same")

                    for head_id in head_ids:
                        _, _, head, source, dest, *_ = head_id

                        subgraph.node(
                            str(head_id),
                            label=f"Attn L{layer}H{head}\n'{self.str_tokens[source]}'/{source} → '{self.str_tokens[dest]}'/{dest}",
                        )

            for component in [
                CircuitComponent.Z_FEATURE,
                CircuitComponent.Z_SAE_ERROR,
                CircuitComponent.Z_SAE_BIAS,
                CircuitComponent.BIAS_O,
                CircuitComponent.MLP_FEATURE,
                CircuitComponent.TRANSCODER_ERROR,
                CircuitComponent.TRANSCODER_BIAS,
            ]:
                comp_ids = list(layers[layer].get(component, set()))

                if not comp_ids:
                    continue

                with G.subgraph() as subgraph:
                    subgraph.attr(rank="same")

                    for comp_id in comp_ids:
                        _, _, seq_index, feature = comp_id

                        if component in [
                            CircuitComponent.Z_FEATURE,
                            CircuitComponent.MLP_FEATURE,
                        ]:
                            label = f"{component_to_name(component)} L{layer}\n'{self.str_tokens[seq_index]}'/{seq_index}\n{feature}"
                        else:
                            label = f"{component_to_name(component)} L{layer}\n'{self.str_tokens[seq_index]}'/{seq_index}"

                        subgraph.node(
                            str(comp_id),
                            label=label,
                        )

        def add_edges(node: CircuitDiscoveryNode, G: Digraph):
            if isinstance(node, CircuitDiscoveryHeadNode):
                for head_type in ["q", "k", "v"]:
                    contributors = node.contributors_in_graph(head_type)

                    if head_type == "q":
                        color = "blue"
                    elif head_type == "k":
                        color = "green"
                    elif head_type == "v":
                        color = "red"

                    for contributor in contributors:
                        G.edge(
                            str(contributor.tuple_id), str(node.tuple_id), color=color
                        )

            if isinstance(node, CircuitDiscoveryRegularNode):
                for contributor in node.contributors_in_graph:
                    G.edge(str(contributor.tuple_id), str(node.tuple_id))

        fn = partial(add_edges, G=G)

        self.traverse_graph(fn)

        return G
