import torch
import einops
import circuitsvis as cv
import plotly.express as px

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
from transformer_lens import ActivationCache, HookedTransformer
from ranking_utils import visualize_top_tokens_for_seq
from jaxtyping import Float
from IPython.display import HTML
from plotly_utils import *
from pprint import pprint


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


# class ComponentEdgeTracker:
#     def __init__(self, transformer_model: "TransformerAnalysisModel"):
#         self._reciever_to_contributors: Dict[Tuple, Set[Tuple]] = {}
#         self._contributor_to_recievers: Dict[Tuple, Set[Tuple]] = {}

#         self.transformer_model = transformer_model

#     def add_edge(self, reciever: Tuple, contributor: Tuple):
#         if reciever not in self._reciever_to_contributors:
#             self._reciever_to_contributors[reciever] = set()
#         if contributor not in self._contributor_to_recievers:
#             self._contributor_to_recievers[contributor] = set()

#         self._reciever_to_contributors[reciever].add(contributor)
#         self._contributor_to_recievers[contributor].add(reciever)

#     def remove_edge(self, reciever: Tuple, contributor: Tuple):
#         if reciever in self._reciever_to_contributors:
#             self._reciever_to_contributors[reciever].discard(contributor)
#         if contributor in self._contributor_to_recievers:
#             self._contributor_to_recievers[contributor].discard(reciever)

#     def get_contributors(self, reciever: Tuple) -> List[Tuple]:
#         return list(self._reciever_to_contributors.get(reciever, set()))

#     def clear(self):
#         self._reciever_to_contributors = {}
#         self._contributor_to_recievers = {}


class ComponentEdgeTracker:
    def __init__(self, transformer_model: "TransformerAnalysisModel"):
        self._reciever_to_contributors: Dict[Tuple, Dict[Tuple, float]] = {}
        self._contributor_to_recievers: Dict[Tuple, Dict[Tuple, float]] = {}
        self.transformer_model = transformer_model

    def add_edge(self, reciever: Tuple, contributor: Tuple, weight: float):
        if reciever not in self._reciever_to_contributors:
            self._reciever_to_contributors[reciever] = {}
        if contributor not in self._contributor_to_recievers:
            self._contributor_to_recievers[contributor] = {}

        self._reciever_to_contributors[reciever][contributor] = weight
        self._contributor_to_recievers[contributor][reciever] = weight

    def remove_edge(self, reciever: Tuple, contributor: Tuple):
        if reciever in self._reciever_to_contributors:
            self._reciever_to_contributors[reciever].pop(contributor, None)
        if contributor in self._contributor_to_recievers:
            self._contributor_to_recievers[contributor].pop(reciever, None)

    def get_contributors(self, reciever: Tuple) -> List[Tuple]:
        return list(self._reciever_to_contributors.get(reciever, {}).keys())

    def get_contributors_with_weights(
        self, reciever: Tuple
    ) -> List[Tuple[Tuple, float]]:
        return list(self._reciever_to_contributors.get(reciever, {}).items())

    def get_total_contributions(self, component_type, layer, head):
        incoming_contributions = 0
        outgoing_contributions = 0

        # for reciever, contributors in self._reciever_to_contributors.items():
        #     if reciever[:3] == (component_type, layer, head):
        #         incoming_contributions += sum(contributors.values())

        for contributor, recievers in self._contributor_to_recievers.items():
            if contributor[:3] == (component_type, layer, head):
                outgoing_contributions += sum(recievers.values())

        return incoming_contributions + outgoing_contributions

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
        self, reciever_id: Tuple, contributor_lens: ComponentLens, weight: float
    ) -> "CircuitDiscoveryNode":
        contributor_node = self.get_discovery_node_at_locator(contributor_lens)

        self.edge_tracker.add_edge(reciever_id, contributor_node.tuple_id, weight)

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
        if isinstance(locator, tuple):
            lens = None
            tuple_id: Tuple = locator
        else:
            lens = locator
            tuple_id: Tuple = lens.tuple_id

        component = tuple_id[0]

        if component == CircuitComponent.ATTN_HEAD:
            tuple_id = CircuitDiscoveryHeadNode.process_tuple_id(tuple_id)

        if tuple_id in self.discovery_node_cache:
            return self.discovery_node_cache[tuple_id]

        if component == CircuitComponent.UNEMBED:
            if lens is None:
                raise ValueError("Must provide lens for unembed node creation")

            return self._cache_discovery_node(
                tuple_id, CircuitDiscoveryRegularNode(lens, self.circuit_discovery)
            )
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
    component_lens: ComponentLens

    def __init__(
        self, component_lens: ComponentLens, circuit_discovery: "CircuitDiscovery"
    ):
        self.circuit_discovery = circuit_discovery
        self.component_lens = component_lens

    @property
    def transformer_model(self):
        return self.circuit_discovery.transformer_model

    @property
    def feature(self):
        return self.component_lens.run_data.get("feature", -1)

    @property
    def k(self):
        return self.circuit_discovery.k

    @property
    def tuple_id(self):
        return self.component_lens.tuple_id

    @property
    def component(self):
        return self.tuple_id[0]

    @property
    def layer(self):
        raise NotImplementedError("Override this")

    def __str__(self):
        return f"DiscoveryNode: {str(self.component_lens)}"

    def __repr__(self):
        return str(self)

    @property
    def explored(self):
        return NotImplementedError("Override this")


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
    def seq_index(self):
        return int(self.component_lens.run_data.get("seq_index", -1))

    @property
    def top_k_contributors(self) -> List[ComponentLensWithValue]:
        if self._top_k_contributors is None:
            self._top_k_contributors, *_ = self.component_lens(
                visualize=False, k=self.k
            )

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
    def explored(self):
        return not self.top_k_allowed_contributors or bool(self.contributors_in_graph)

    @property
    def contributors_in_graph(self) -> List["CircuitDiscoveryNode"]:
        return self.transformer_model.get_contributors_in_graph(self.tuple_id)

    @property
    def contributors_in_graph_with_weight(self):
        contributors = self.contributors_in_graph

        id_value = {node.tuple_id: value for (node, value) in self.top_k_contributors}

        return [(node, id_value[node.tuple_id]) for node in contributors]

    @property
    def sorted_contributors_in_graph(self) -> List["CircuitDiscoveryNode"]:
        id_value = {node.tuple_id: value for (node, value) in self.top_k_contributors}

        return sorted(self.contributors_in_graph, key=lambda x: -id_value[x.tuple_id])

    def add_contributor_edge(
        self, component_lens: ComponentLens, weight: float
    ) -> "CircuitDiscoveryNode":
        return self.transformer_model.add_contributor_edge(
            self.tuple_id, component_lens, weight
        )

    def get_top_unused_contributors(self) -> List[ComponentLensWithValue]:
        contributors_in_graph = self.contributors_in_graph

        contributor_ids = [c.tuple_id for c in contributors_in_graph]

        return [
            new_contributor
            for new_contributor in self.top_k_allowed_contributors
            if new_contributor[0].tuple_id not in contributor_ids
        ]

    def lens(self):
        self.component_lens()


class CircuitDiscoveryHeadNode(CircuitDiscoveryNode):
    def __init__(
        self, component_lens: ComponentLens, circuit_discovery: "CircuitDiscovery"
    ):
        super().__init__(component_lens, circuit_discovery)

        self._top_k_allowed_contributors: Dict[str, List[ComponentLensWithValue]] = {}
        self._top_k_contributors: Dict[str, List[ComponentLensWithValue]] = {}

    def lens(self, head_type):
        self._validate_head_type(head_type)

        self.component_lens(head_type=head_type)

    @property
    def layer(self):
        return self.component_lens.run_data["layer"]

    @property
    def head(self):
        return self.component_lens.run_data["head"]

    @property
    def source(self):
        return int(self.component_lens.run_data["source_index"])

    @property
    def explored(self):
        return all(
            not self.top_k_allowed_contributors(head_type)
            or bool(self.contributors_in_graph(head_type))
            for head_type in ("q", "k", "v")
        )

    @property
    def dest(self):
        return int(self.component_lens.run_data["destination_index"])

    def _validate_head_type(self, head_type: str):
        assert head_type in ["q", "k", "v"]

    def top_k_contributors(self, head_type) -> List[ComponentLensWithValue]:
        self._validate_head_type(head_type)

        if head_type not in self._top_k_contributors:
            self._top_k_contributors[head_type], *_ = self.component_lens(
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

    def contributors_in_graph_with_weight(self, head_type: str):
        contributors = self.contributors_in_graph(head_type)

        id_value = {
            node.tuple_id: value for (node, value) in self.top_k_contributors(head_type)
        }

        return [(node, id_value[node.tuple_id]) for node in contributors]

    def sorted_contributors_in_graph(self, head_type) -> List["CircuitDiscoveryNode"]:
        self._validate_head_type(head_type)

        id_value = {
            node.tuple_id: value for (node, value) in self.top_k_contributors(head_type)
        }

        return sorted(
            self.contributors_in_graph(head_type), key=lambda x: -id_value[x.tuple_id]
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
        self, head_type: str, component_lens: ComponentLens, weight: float
    ) -> "CircuitDiscoveryNode":
        self._validate_head_type(head_type)

        return self.transformer_model.add_contributor_edge(
            self.tuple_id_for_head_type(head_type), component_lens, weight
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
        seq_index=None,
        token: Optional[Union[str, int]] = None,
        allowed_components_filter: Callable[[str], bool] = all_allowed,
        k=10,
    ):
        if seq_index is None:
            if token is None:
                seq_index = -2
            else:
                seq_index = -1

        self.lens = CircuitLens(prompt)
        self.seq_index = self.lens.process_seq_index(seq_index)
        self.allowed_components_filter = allowed_components_filter

        self.k = k

        self.transformer_model = TransformerAnalysisModel(self)

        if token:
            if isinstance(token, str):
                token = int(self.model.to_single_token(token))

            self.token = token
            self.root_node = self.transformer_model.get_discovery_node_at_locator(
                ComponentLens.create_unembed_lens(self.lens, self.seq_index, token),
            )
        else:
            self.token = None
            self.root_node = self.transformer_model.get_discovery_node_at_locator(
                ComponentLens.create_unembed_at_token_lens(self.lens, self.seq_index),
            )

    def set_root(self, tuple_id, no_graph_reset=False):
        if not no_graph_reset:
            self.reset_graph()

        self.root_node = self.transformer_model.get_discovery_node_at_locator(
            ComponentLens.init_from_tuple_id(tuple_id=tuple_id, circuit_lens=self.lens)
        )

    def set_root_to_z_feature(self, layer, seq_index, feature):
        self.set_root((CircuitComponent.Z_FEATURE, layer, seq_index, feature))

    def set_root_to_mlp_feature(self, layer, seq_index, feature):
        self.set_root((CircuitComponent.MLP_FEATURE, layer, seq_index, feature))

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

    def all_tuple_ids_in_graph(self):
        all_ids = []

        def visit_fn(node, all_ids):
            all_ids.append(node)

        fn = partial(visit_fn, all_ids=all_ids)
        self.traverse_graph(fn)

        return all_ids

    def get_graph_edge_matrix(self, flatten=False):
        matrix_methods = EdgeMatrixMethods(self.model)

        edge_matrix = matrix_methods.create_empty_edge_matrix()

        def get_contrib_source_indices(contrib: CircuitDiscoveryRegularNode):
            indices = []

            if contrib.component == CircuitComponent.MLP_FEATURE:
                indices.append(matrix_methods.mlp_source_index(contrib.layer))
            elif contrib.component in [
                CircuitComponent.Z_FEATURE,
                CircuitComponent.Z_SAE_ERROR,
            ]:
                for head in contrib.contributors_in_graph:
                    if not isinstance(head, CircuitDiscoveryHeadNode):
                        continue

                    indices.append(
                        matrix_methods.head_source_index(head.layer, head.head)
                    )

            return indices

        def visit_fn(node: CircuitDiscoveryNode, edge_matrix):
            if node.component not in [
                CircuitComponent.ATTN_HEAD,
                CircuitComponent.MLP_FEATURE,
            ]:
                return

            if isinstance(node, CircuitDiscoveryRegularNode):
                for contrib in node.contributors_in_graph:
                    if not isinstance(contrib, CircuitDiscoveryRegularNode):
                        continue

                    source_indices = get_contrib_source_indices(contrib)

                    edge_matrix[
                        source_indices, matrix_methods.mlp_target_index(node.layer)
                    ] = 1
            elif isinstance(node, CircuitDiscoveryHeadNode):
                for head_type in ["q", "k", "v"]:
                    for contrib in node.contributors_in_graph(head_type):
                        if not isinstance(contrib, CircuitDiscoveryRegularNode):
                            continue

                        source_indices = get_contrib_source_indices(contrib)

                        edge_matrix[
                            source_indices,
                            matrix_methods.head_target_index(
                                node.layer, node.head, head_type
                            ),
                        ] = 1

        fn = partial(visit_fn, edge_matrix=edge_matrix)
        self.traverse_graph(fn)

        if flatten:
            return einops.rearrange(edge_matrix, "s t -> (s t)")
        else:
            return edge_matrix

    def get_token_pos_referenced_by_graph(self, no_bos=False) -> List[int]:
        pos_set = set()

        def visit(node, pos_set):
            if not (
                isinstance(node, CircuitDiscoveryRegularNode)
                and node.component
                in [CircuitComponent.EMBED, CircuitComponent.POS_EMBED]
            ):
                return

            pos_set.add(node.seq_index)

        fn = partial(visit, pos_set=pos_set)
        self.traverse_graph(fn)

        if no_bos:
            pos_set.discard(0)

        final = sorted(list(pos_set))

        return final

    def get_context_referenced_prompt(
        self,
        pos=None,
        token_lr=("<<", ">>"),
        context_lr=("[[", "]]"),
        merge_nearby_context=False,
    ):
        tl, tr = token_lr
        cl, cr = context_lr

        if pos is None:
            pos = self.seq_index

        context_token_pos = self.get_token_pos_referenced_by_graph(no_bos=True)

        str_tokens: List[str] = self.model.to_str_tokens(self.tokens)  # type: ignore

        if pos in context_token_pos:
            context_token_pos.remove(pos)

        prompt = "".join(str_tokens[1 : pos + 1])

        if not context_token_pos:
            prompt_with_emphasis = (
                "".join(str_tokens[1:pos]) + f"{tl}{str_tokens[pos]}{tr}"
            )
            return prompt, prompt_with_emphasis

        if not merge_nearby_context:
            prompt_with_context = ""

            for i, tok in enumerate(str_tokens[:pos]):
                if i == 0:
                    continue

                if i in context_token_pos:
                    prompt_with_context += f"{cl}{tok}{cr}"
                else:
                    prompt_with_context += tok

            prompt_with_context += f"{tl}{str_tokens[pos]}{tr}"

            return prompt, prompt_with_context

        token_ranges = []

        i = 0

        while i < len(self.tokens):
            if i in context_token_pos:
                range_start = i

                k = i + 1

                while k < len(self.tokens):
                    if k in context_token_pos:
                        k += 1
                    else:
                        break

                token_ranges.append((range_start, k))
                i = k
            else:
                i += 1

        prompt_with_context = "".join(str_tokens[1 : token_ranges[0][0]])

        for i, token_range in enumerate(token_ranges):
            if i == len(token_ranges) - 1:
                next_start = pos
            else:
                next_start = token_ranges[i + 1][0]

            s, e = token_range

            prompt_with_context += f"{cl}{''.join(str_tokens[s:e])}{cr}"
            prompt_with_context += "".join(str_tokens[e:next_start])

        prompt_with_context += f"{tl}{str_tokens[pos]}{tr}"

        return prompt, prompt_with_context

    def all_nodes_in_graph(self):
        all_nodes = []

        def visit_fn(node, all_nodes):
            all_nodes.append(node)

        fn = partial(visit_fn, all_nodes=all_nodes)
        self.traverse_graph(fn)

        return all_nodes

    def add_greedy_pass_against_all_existing_nodes(
        self,
        contributors_per_node=1,
        skip_z_features=False,
        layer_threshold=-1,
        minimal=False,
    ):
        for node in self.all_nodes_in_graph():
            if skip_z_features and node.component == CircuitComponent.Z_FEATURE:
                continue

            if node.layer < layer_threshold and node.component not in [
                CircuitComponent.UNEMBED,
                CircuitComponent.UNEMBED_AT_TOKEN,
            ]:
                continue

            self.add_greedy_pass(
                contributors_per_node=contributors_per_node, root=node, minimal=minimal
            )

    def get_top_next_contributors(
        self, k=1, reciever_threshold=None, contributor_threshold=None
    ):
        all_top_contributors = []

        if reciever_threshold is None:
            reciever_threshold = 0

        if contributor_threshold is None:
            contributor_threshold = 0

        def visit_fn(node, all_top_contributors, k):
            if node.layer < reciever_threshold and node.component not in [
                CircuitComponent.UNEMBED,
                CircuitComponent.UNEMBED_AT_TOKEN,
            ]:
                return

            if isinstance(node, CircuitDiscoveryRegularNode):
                contributors = [
                    c
                    for c in node.get_top_unused_contributors()
                    if c[0].run_data.get("layer", 0) >= contributor_threshold
                ][:k]

                for contributor in contributors:
                    all_top_contributors.append((contributor[1], contributor[0], node))

            elif isinstance(node, CircuitDiscoveryHeadNode):
                for head_type in ["q", "k", "v"]:
                    contributors = [
                        c
                        for c in node.get_top_unused_contributors(head_type)
                        if c[0].run_data.get("layer", 0) >= contributor_threshold
                    ][:k]

                    for contributor in contributors:
                        all_top_contributors.append(
                            (contributor[1], contributor[0], node, head_type)
                        )

        fn = partial(visit_fn, all_top_contributors=all_top_contributors, k=k)

        self.traverse_graph(fn)

        top_contribs = sorted(all_top_contributors, key=lambda x: x[0], reverse=True)

        return top_contribs

    def greedily_add_top_contributors(
        self,
        k=1,
        print_new_contributs=False,
        reciever_threshold=None,
        contributor_threshold=None,
    ):
        top_contribs = self.get_top_next_contributors(
            k,
            reciever_threshold=reciever_threshold,
            contributor_threshold=contributor_threshold,
        )[:k]

        if print_new_contributs:
            pprint(top_contribs)

        for contrib in top_contribs:
            reciever = contrib[2]
            contribution_value = contrib[0]

            if isinstance(reciever, CircuitDiscoveryRegularNode):
                node = reciever.add_contributor_edge(contrib[1], contribution_value)
            elif isinstance(reciever, CircuitDiscoveryHeadNode):
                node = reciever.add_contributor_edge(
                    contrib[3], contrib[1], contribution_value
                )

            if not node.explored:
                self.add_greedy_pass(root=node, minimal=True)

    def add_greedy_pass(self, contributors_per_node=1, root=None, minimal=False):
        """
        if `minimal` then we don't recursively explore nodes that are already in the graph
        """

        visited_ids = []

        if root is None:
            root = self.root_node

        queue: List[CircuitDiscoveryNode] = [root]

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
                    child_discovery_node = node.add_contributor_edge(child[0], child[1])

                    total_contrib += child[1]

                    if minimal and child_discovery_node.explored:
                        continue

                    queue.append(child_discovery_node)

            elif isinstance(node, CircuitDiscoveryHeadNode):
                for head_type in ["q", "k", "v"]:
                    top_contribs = node.get_top_unused_contributors(head_type)
                    if len(top_contribs) == 0:
                        continue

                    total_contrib = 0

                    for child in top_contribs[:contributors_per_node]:
                        child_discovery_node = node.add_contributor_edge(
                            head_type, child[0], child[1]
                        )

                        total_contrib += child[1]

                        if minimal and child_discovery_node.explored:
                            continue

                        queue.append(child_discovery_node)

                    # print(
                    #     node.tuple_id_for_head_type(head_type),
                    #     f"Contrib: {total_contrib:.3g}%",
                    # )

    def attn_heads_tensor(self, visualize=False):
        attn_heads = torch.zeros(12, 12)

        def track_included_heads_and_mlps(node: CircuitDiscoveryNode, attn_heads):
            if isinstance(node, CircuitDiscoveryHeadNode):
                attn_heads[node.layer, node.head] = 1

        fn = partial(track_included_heads_and_mlps, attn_heads=attn_heads)

        self.traverse_graph(fn)

        if visualize:
            imshow(attn_heads, labels={"x": "Head", "y": "Layer"})

        return attn_heads

    def mlps_tensor(self, visualize=False):
        mlps = torch.zeros(12)

        def track_included_heads_and_mlps(node: CircuitDiscoveryNode, mlps):
            if isinstance(node, CircuitDiscoveryRegularNode):
                if node.component == CircuitComponent.MLP_FEATURE:
                    mlps[node.layer] = 1

        fn = partial(track_included_heads_and_mlps, mlps=mlps)

        self.traverse_graph(fn)

        if visualize:
            imshow(mlps, labels={"x": "Layer"})

        return mlps

    def get_heads_and_mlps_in_graph(self) -> Tuple[List[List[int]], List[int]]:
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

    def are_heads_above_layer(self, layer: int):
        attn_heads, _ = self.get_heads_and_mlps_in_graph()

        for layer_heads in attn_heads[layer:]:
            if layer_heads:
                return True

        return False

    def get_heads_and_mlps_in_graph_at_seq(
        self,
    ) -> Tuple[List[List[List[int]]], List[List[int]]]:
        attn_heads_set = set()
        mlps = set()

        def track_included_heads_and_mlps(
            node: CircuitDiscoveryNode,
            attn_heads_set,
            mlps,
        ):
            if isinstance(node, CircuitDiscoveryHeadNode):
                attn_heads_set.add((node.layer, node.head, node.dest))
            elif isinstance(node, CircuitDiscoveryRegularNode):
                if node.component == CircuitComponent.MLP_FEATURE:
                    mlps.add((node.layer, node.seq_index))

        fn = partial(
            track_included_heads_and_mlps, attn_heads_set=attn_heads_set, mlps=mlps
        )

        self.traverse_graph(fn)

        attn_heads = [
            [[] for _ in range(self.model.cfg.n_heads)]
            for _ in range(self.model.cfg.n_layers)
        ]
        mlps_at_seq = [[] for _ in range(self.model.cfg.n_layers)]

        for layer, seq_index in mlps:
            mlps_at_seq[layer].append(seq_index)

        for layer, head, seq_index in attn_heads_set:
            attn_heads[layer][head].append(seq_index)

        return attn_heads, mlps_at_seq

    def print_attn_heads_and_mlps_in_graph(self):
        attn_heads, mlps = self.get_heads_and_mlps_in_graph()

        for layer in reversed(range(self.model.cfg.n_layers)):
            layer_heads = attn_heads[layer]
            layer_heads.sort()

            if layer in mlps:
                print(f"MLP{layer}")
            if layer_heads:
                print(f"L{layer}H:", layer_heads)

    def visualize_attn_heads_in_graph(self, max_width=800, value_weighted=False):
        _, cache = self.model.run_with_cache(self.tokens)

        attn_heads, _ = self.get_heads_and_mlps_in_graph()

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

        min_pos, max_pos = self.get_min_max_pos_in_graph()

        patterns = patterns[:, min_pos : max_pos + 1, min_pos : max_pos + 1]

        # Circuitsvis Plot (note we get the code version so we can concatenate with the title)
        plot = cv.attention.attention_heads(
            attention=patterns,
            tokens=self.str_tokens[min_pos : max_pos + 1],
            attention_head_names=labels,
        ).show_code()

        # Display the title
        title_html = f"<h2>Attention Heads in Graph</h2><br/>"

        # Return the visualisation as raw code
        return HTML(
            f"<div style='max-width: {str(max_width)}px;'>{title_html + plot}</div>"
        )

    def get_logits_for_graph(self, z_hook, mlp_hook, **kwargs):
        include_all_mlps = kwargs.get("include_all_mlps", False)
        include_all_heads = kwargs.get("include_all_heads", False)

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

    def get_logits_for_graph_against_base_ablation(self, **kwargs):
        head_ablation_style = kwargs.get("head_ablation_style", "bos")

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

        attn_heads, mlps = self.get_heads_and_mlps_in_graph()

        z_hook = partial(ablate_z_out, attn_heads=attn_heads)
        mlp_hook = partial(ablate_mlp_out, mlps=mlps)

        return self.get_logits_for_graph(mlp_hook=mlp_hook, z_hook=z_hook, **kwargs)

    def get_logits_for_graph_against_mean_ablation(
        self, mean_cache: ActivationCache, **kwargs
    ):

        def ablate_z_out(acts, hook: HookPoint, attn_heads):
            layer_heads = attn_heads[hook.layer()]

            mean_acts = einops.repeat(
                mean_cache[hook.name], "... -> B ...", B=acts.size(0)
            )

            for head in range(self.model.cfg.n_heads):
                if head not in layer_heads:
                    acts[:, :, head, :] = mean_acts[:, :, head, :]

            return acts

        def ablate_mlp_out(acts, hook: HookPoint, mlps):
            mean_acts = einops.repeat(
                mean_cache[hook.name], "... -> B ...", B=acts.size(0)
            )

            if hook.layer() not in mlps:
                acts = mean_acts

            return acts

        attn_heads, mlps = self.get_heads_and_mlps_in_graph()

        z_hook = partial(ablate_z_out, attn_heads=attn_heads)
        mlp_hook = partial(ablate_mlp_out, mlps=mlps)

        return self.get_logits_for_graph(mlp_hook=mlp_hook, z_hook=z_hook, **kwargs)

    def get_logits_for_graph_against_edge_based_mean_ablation(
        self, mean_cache: ActivationCache, **kwargs
    ):
        def ablate_z_out(acts, hook: HookPoint, attn_heads):
            layer_heads = attn_heads[hook.layer()]

            mean_acts = einops.repeat(
                mean_cache[hook.name], "... -> B ...", B=acts.size(0)
            )

            # for i, seq_index_for_head in enumerate(attn_heads):
            for head in range(self.model.cfg.n_heads):
                seq_indices = layer_heads[head]

                for seq_index in range(acts.size(1)):
                    if seq_index not in seq_indices:
                        acts[:, seq_index, head, :] = mean_acts[:, seq_index, head, :]

            return acts

        def ablate_mlp_out(acts, hook: HookPoint, mlps):
            mean_acts = einops.repeat(
                mean_cache[hook.name], "... -> B ...", B=acts.size(0)
            )

            seq_indices = mlps[hook.layer()]

            for seq_index in range(acts.size(1)):
                if seq_index not in seq_indices:
                    acts[:, seq_index, :] = mean_acts[:, seq_index, :]

            return acts

        attn_heads, mlps = self.get_heads_and_mlps_in_graph_at_seq()

        z_hook = partial(ablate_z_out, attn_heads=attn_heads)
        mlp_hook = partial(ablate_mlp_out, mlps=mlps)

        return self.get_logits_for_graph(mlp_hook=mlp_hook, z_hook=z_hook, **kwargs)

    def get_logit_difference_on_mean_ablation(
        self,
        correct_str_token: str,
        counter_str_token: str,
        mean_cache,
        mean_logits,
        visualize=True,
        edge_based_graph_evaluation=False,
        **kwargs,
    ):
        base_logits = self.model(self.tokens, return_type="logits")
        base_logits = base_logits[0, self.seq_index]

        if edge_based_graph_evaluation:
            graph_logits = self.get_logits_for_graph_against_edge_based_mean_ablation(
                mean_cache=mean_cache, **kwargs
            )
        else:
            graph_logits = self.get_logits_for_graph_against_mean_ablation(
                mean_cache=mean_cache, **kwargs
            )

        graph_logits = graph_logits[self.seq_index]
        mean_logits = mean_logits[self.seq_index]
        # print("shapez", base_logits.shape, graph_logits.shape, mean_logits.shape)

        tokens = self.model.to_tokens(
            [correct_str_token, counter_str_token], prepend_bos=False
        ).squeeze()
        correct_token = tokens[0]
        counter_token = tokens[1]

        mean_logit_diff = mean_logits[correct_token] - mean_logits[counter_token]
        base_logit_diff = base_logits[correct_token] - base_logits[counter_token]
        graph_logit_diff = graph_logits[correct_token] - graph_logits[counter_token]

        normalized = (graph_logit_diff - mean_logit_diff) / (
            base_logit_diff - mean_logit_diff
        )

        normalized = normalized.item()
        mean_logit_diff = mean_logit_diff.item()
        base_logit_diff = base_logit_diff.item()
        graph_logit_diff = graph_logit_diff.item()

        if visualize:
            print(
                f"Normalized: {normalized * 100:.3g}% | Base: {base_logit_diff:.3g} | Graph: {graph_logit_diff:.3g} | Mean: {mean_logit_diff:.3g}"
            )

        return normalized, base_logit_diff, graph_logit_diff, mean_logit_diff

    def visualize_graph_performance_against_mean_ablation(self, mean_cache, **kwargs):
        base_logits = self.model(self.tokens, return_type="logits")
        graph_logits = self.get_logits_for_graph_against_mean_ablation(
            mean_cache=mean_cache, **kwargs
        )

        k = kwargs.get("k", 5)

        print("Base Performance:")

        visualize_top_tokens_for_seq(
            self.model, self.tokens, base_logits, self.seq_index, k=k
        )

        print()
        print("Graph Performance:")

        visualize_top_tokens_for_seq(
            self.model, self.tokens, graph_logits, self.seq_index, k=k
        )

    def visualize_graph_performance_against_base_ablation(self, **kwargs):
        base_logits = self.model(self.tokens, return_type="logits")

        graph_logits = self.get_logits_for_graph_against_base_ablation(**kwargs)

        print("Base Performance:")

        visualize_top_tokens_for_seq(
            self.model, self.tokens, base_logits, self.seq_index
        )

        print()
        print("Graph Performance:")

        visualize_top_tokens_for_seq(
            self.model, self.tokens, graph_logits, self.seq_index
        )

    def get_features_at_heads_in_graph(self):
        features_at_heads = [
            [set() for _ in range(self.model.cfg.n_heads)]
            for _ in range(self.model.cfg.n_layers)
        ]

        def visit_fn(node: CircuitDiscoveryNode, features_at_heads: List[List[Set]]):
            if not isinstance(node, CircuitDiscoveryHeadNode):
                return

            head = node.head
            layer = node.layer
            feature = node.component_lens.run_data["feature"]

            features_at_heads[layer][head].add(feature)

        fn = partial(visit_fn, features_at_heads=features_at_heads)

        self.traverse_graph(fn)

        return features_at_heads

    def get_features_at_mlps_in_graph(self):
        features_at_mlps = [set() for _ in range(self.model.cfg.n_layers)]

        def visit_fn(node: CircuitDiscoveryNode, features_at_mlps: List[Set]):
            if not isinstance(node, CircuitDiscoveryRegularNode):
                return

            if node.component == CircuitComponent.MLP_FEATURE:
                layer = node.layer
                feature = node.component_lens.run_data["feature"]

                features_at_mlps[layer].add(feature)

        fn = partial(visit_fn, features_at_mlps=features_at_mlps)

        self.traverse_graph(fn)

        return features_at_mlps

    def component_lens_at_loc(self, loc: List):
        node: CircuitDiscoveryNode = self.root_node
        head_type = "q"

        while loc:
            next_loc = loc.pop(0)

            if isinstance(node, CircuitDiscoveryRegularNode):
                sorted_contribs = node.top_k_contributors
            elif isinstance(node, CircuitDiscoveryHeadNode):
                sorted_contribs = node.top_k_contributors(head_type)

            if next_loc < 0 or next_loc > len(sorted_contribs) - 1:
                raise ValueError("Invalid location")

            node = self.transformer_model.get_discovery_node_at_locator(
                sorted_contribs[next_loc][0]
            )

            if isinstance(node, CircuitDiscoveryHeadNode):
                if not loc:
                    raise ValueError("Need to provide 'q', 'k', or 'v'")

                head_type = loc.pop(0)
                if head_type not in ["q", "k", "v"]:
                    raise ValueError("Invalid head input")

        if isinstance(node, CircuitDiscoveryRegularNode):
            node.lens()
        elif isinstance(node, CircuitDiscoveryHeadNode):
            node.lens(head_type)

    def component_lens_at_loc_on_graph(self, loc: List):
        node: CircuitDiscoveryNode = self.root_node
        head_type = "q"

        while loc:
            next_loc = loc.pop(0)

            if isinstance(node, CircuitDiscoveryRegularNode):
                sorted_contribs = node.sorted_contributors_in_graph
            elif isinstance(node, CircuitDiscoveryHeadNode):
                sorted_contribs = node.sorted_contributors_in_graph(head_type)

            if next_loc < 0 or next_loc > len(sorted_contribs) - 1:
                raise ValueError("Invalid location")

            node = sorted_contribs[next_loc]

            if isinstance(node, CircuitDiscoveryHeadNode):
                if not loc:
                    raise ValueError("Need to provide 'q', 'k', or 'v'")

                head_type = loc.pop(0)
                if head_type not in ["q", "k", "v"]:
                    raise ValueError("Invalid head input")

        if isinstance(node, CircuitDiscoveryRegularNode):
            node.lens()
        elif isinstance(node, CircuitDiscoveryHeadNode):
            node.lens(head_type)

    def get_min_max_pos_in_graph(self):
        min_max = {"min": 10**4, "max": 0}

        def visit(node: CircuitDiscoveryNode, min_max):
            if isinstance(node, CircuitDiscoveryRegularNode):
                if node.seq_index == 0:
                    return

                if node.seq_index < min_max["min"]:
                    min_max["min"] = node.seq_index

                if node.seq_index > min_max["max"]:
                    min_max["max"] = node.seq_index

            elif isinstance(node, CircuitDiscoveryHeadNode):
                for pos in [node.source, node.dest]:
                    if pos == 0:
                        continue

                    if pos < min_max["min"]:
                        min_max["min"] = pos

                    if pos > min_max["max"]:
                        min_max["max"] = pos

        fn = partial(visit, min_max=min_max)
        self.traverse_graph(fn)

        return (min_max["min"], min_max["max"])

    def visualize_graph(self, begin_layer=-1, cutoff_tokens_at_min_max=True):
        G = Digraph()
        G.graph_attr.update(rankdir="BT", newrank="true")
        G.node_attr.update(
            shape="box", style="rounded", fontsize="10pt", margin="0.01,0.01"
        )
        G.attr(nodesep="0.1")

        layers = [dict() for _ in range(self.model.cfg.n_layers)]
        embed = dict()

        def edge_color(opacity: float, color: str = "black"):
            opacity = min(1.0, max(0.0, opacity))
            start = 0.1

            opacity = start + (1 - start) * opacity

            R = 1 - opacity
            B = 1 - opacity
            G = 1 - opacity

            if color == "red":
                R = 1
            elif color == "green":
                G = 1
            elif color == "blue":
                B = 1

            return "#{:02x}{:02x}{:02x}".format(
                int(R * 255), int(G * 255), int(B * 255)
            )

        def node_id(tuple_id):
            if tuple_id != CircuitComponent.ATTN_HEAD:
                return str(tuple_id)

            comp, layer, head, source, dest, *_ = tuple_id

            return str((comp, layer, head, source, dest))

        def add_node_to_record(node: CircuitDiscoveryNode, layers, embed):
            component = node.tuple_id[0]

            if component in [
                CircuitComponent.UNEMBED,
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

        min_pos, max_pos = self.get_min_max_pos_in_graph()
        max_pos += 1

        if begin_layer < 0:
            with G.subgraph(name="words") as subgraph:  # type: ignore
                subgraph.attr(rank="same")
                for i, str_token in enumerate(self.str_tokens):
                    if cutoff_tokens_at_min_max and (i < min_pos or i > max_pos):
                        continue

                    subgraph.node(
                        embed_id(i),
                        label=str_token + f"\n{i}",
                    )

                    min_cutoff = 0
                    if cutoff_tokens_at_min_max:
                        min_cutoff = min_pos

                    if i > min_cutoff:
                        subgraph.edge(
                            embed_id(i - 1), embed_id(i), style="invis", minlen=".1"
                        )

            pos_embeds = list(embed.get(CircuitComponent.POS_EMBED, set()))

            if pos_embeds:
                with G.subgraph() as subgraph:  # type: ignore
                    subgraph.attr(rank="same")
                    for pos_embed in pos_embeds:
                        seq_index = pos_embed[2]

                        subgraph.node(node_id(pos_embed), label=f"POS: {seq_index}")

                        G.edge(
                            embed_id(seq_index),
                            node_id(pos_embed),
                            style="invis",
                            # , minlen=".1"
                        )
        unembeds_at_token = list(embed.get(CircuitComponent.UNEMBED_AT_TOKEN, set()))

        if unembeds_at_token:
            with G.subgraph(name="unembed") as subgraph:  # type: ignore
                subgraph.attr(rank="same")
                for unembed in unembeds_at_token:
                    seq_index = unembed[1]

                    subgraph.node(
                        node_id(unembed),
                        label=f"UE: '{self.str_tokens[seq_index + 1]}'",
                    )

        unembeds = list(embed.get(CircuitComponent.UNEMBED, set()))

        if unembeds:
            with G.subgraph(name="unembed") as subgraph:  # type: ignore
                subgraph.attr(rank="same")
                for unembed in unembeds:
                    _, seq_index, token_i = unembed

                    subgraph.node(
                        node_id(unembed),
                        label=f"UE: '{self.model.to_single_str_token(token_i)}'",
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

        for layer in range(begin_layer, self.model.cfg.n_layers):
            head_ids = list(layers[layer].get(CircuitComponent.ATTN_HEAD, set()))

            if head_ids:
                with G.subgraph() as subgraph:  # type: ignore
                    subgraph.attr(rank="same")

                    for head_id in head_ids:
                        _, _, head, source, dest, *_ = head_id

                        subgraph.node(
                            node_id(head_id),
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

                with G.subgraph() as subgraph:  # type: ignore
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
                            node_id(comp_id),
                            label=label,
                        )

        def add_edges(node: CircuitDiscoveryNode, G: Digraph):
            if isinstance(node, CircuitDiscoveryHeadNode):
                if node.layer < begin_layer:
                    return

                for head_type in ["q", "k", "v"]:
                    contributors = node.contributors_in_graph_with_weight(head_type)

                    if head_type == "q":
                        color = "blue"
                    elif head_type == "k":
                        color = "green"
                    elif head_type == "v":
                        color = "red"

                    for contributor, weight in contributors:
                        if contributor.layer < begin_layer:
                            continue

                        G.edge(
                            node_id(contributor.tuple_id),
                            node_id(node.tuple_id),
                            color=edge_color(opacity=weight, color=color),
                            label=f"{weight*100:.3g}%",
                            fontsize="8pt",
                        )

            if isinstance(node, CircuitDiscoveryRegularNode):
                if node.layer < begin_layer and node.component not in [
                    CircuitComponent.UNEMBED_AT_TOKEN,
                    CircuitComponent.UNEMBED,
                ]:
                    return

                for contributor, weight in node.contributors_in_graph_with_weight:
                    if contributor.layer < begin_layer:
                        continue
                    G.edge(
                        node_id(contributor.tuple_id),
                        node_id(node.tuple_id),
                        color=edge_color(opacity=weight),
                        label=f"{weight*100:.3g}%",
                        fontsize="8pt",
                    )

        fn = partial(add_edges, G=G)

        self.traverse_graph(fn)

        return G

    def add_directed_edges_with_weights(self, root_node, contributors_per_node=1):
        queue = [root_node]
        visited_ids = set()

        while queue:
            node = queue.pop(0)
            if node.tuple_id in visited_ids:
                continue

            visited_ids.add(node.tuple_id)

            if isinstance(node, CircuitDiscoveryRegularNode):
                top_contribs = node.get_top_unused_contributors()
                for contributor, value in top_contribs[:contributors_per_node]:
                    print(
                        f"Node: {node.tuple_id}, Contributor: {contributor}, Value: {value}"
                    )
                    contributor_node = node.add_contributor_edge(contributor, value)
                    self.transformer_model.edge_tracker.add_edge(
                        node.tuple_id, contributor_node.tuple_id, weight=value
                    )
                    queue.append(contributor_node)

            elif isinstance(node, CircuitDiscoveryHeadNode):
                for head_type in ["q", "k", "v"]:
                    top_contribs = node.get_top_unused_contributors(head_type)
                    for contributor, value in top_contribs[:contributors_per_node]:
                        print(
                            f"Node: {node.tuple_id}, Head Type: {head_type}, Contributor: {contributor}, Value: {value}"
                        )
                        contributor_node = node.add_contributor_edge(
                            head_type, contributor, value
                        )
                        self.transformer_model.edge_tracker.add_edge(
                            node.tuple_id, contributor_node.tuple_id, weight=value
                        )
                        queue.append(contributor_node)

    def recursive_explore_all_nodes(self, root_node, contributors_per_node=1):
        """
        Recursively explore all nodes and add edges for each node.
        """
        visited_ids = set()

        def explore_node(node):
            if node.tuple_id in visited_ids:
                return

            visited_ids.add(node.tuple_id)
            self.add_directed_edges_with_weights(node, contributors_per_node)

            if isinstance(node, CircuitDiscoveryRegularNode):
                for contributor_node in node.contributors_in_graph:
                    explore_node(contributor_node)

            elif isinstance(node, CircuitDiscoveryHeadNode):
                for head_type in ["q", "k", "v"]:
                    for contributor_node in node.contributors_in_graph(head_type):
                        explore_node(contributor_node)

        explore_node(root_node)

    def build_directed_graph(self, contributors_per_node=1):
        """
        Build the entire directed graph by exploring all nodes from the root.
        """
        self.reset_graph()
        self.recursive_explore_all_nodes(self.root_node, contributors_per_node)


class EdgeMatrixMethods:
    def __init__(self, model: HookedTransformer):
        self.model = model

        self.num_head_sources = self.model.cfg.n_heads * (self.model.cfg.n_layers)
        self.num_mlp_sources = self.model.cfg.n_layers - 1

        self.num_head_targets = (
            3 * self.model.cfg.n_heads * (self.model.cfg.n_layers - 1)
        )
        self.num_mlp_targets = self.model.cfg.n_layers

    def create_empty_edge_matrix(self):
        return torch.zeros(
            self.num_head_sources + self.num_mlp_sources,
            self.num_head_targets + self.num_mlp_targets,
        )

    def head_source_index(self, layer: int, head: int):
        return (layer * self.model.cfg.n_heads) + head

    def mlp_source_index(self, layer: int):
        return self.num_head_sources + layer

    def head_target_index(self, layer: int, head: int, head_type: str):
        if head_type == "q":
            type_i = 0
        elif head_type == "k":
            type_i = 1
        elif head_type == "v":
            type_i = 2

        # Head 0 isn't a target
        return ((layer - 1) * self.model.cfg.n_heads * 3) + (head * 3) + type_i

    def mlp_target_index(self, layer: int):
        return self.num_head_targets + layer


class EdgeMatrixAnalyzer:
    def __init__(self, methods: EdgeMatrixMethods, graph_matrices, seq_pos):
        self.methods = methods
        self.model = methods.model

        self.graph_matrices = graph_matrices
        self.graph_matrix_total = graph_matrices.sum(dim=0)

        self.seq_pos = seq_pos

    def get_source_labels(self):
        labels = []

        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                labels.append(f"L{layer}H{head}")

        for layer in range(self.model.cfg.n_layers - 1):
            labels.append(f"L{layer}MLP")

        return labels

    def get_target_labels(self):
        labels = []

        for layer in range(1, self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                labels.append(f"L{layer}H{head}Q")
                labels.append(f"L{layer}H{head}K")
                labels.append(f"L{layer}H{head}V")

        for layer in range(self.model.cfg.n_layers):
            labels.append(f"L{layer}MLP")

        return labels

    def imshow_totals(self):
        imshow(
            self.graph_matrix_total,
            x=self.get_target_labels(),
            y=self.get_source_labels(),
            zmax=20,
            zmin=-20,
            labels={"x": "Target", "y": "Source"},
        )

    def get_matrices_given_source_dest(self, source_dest_list):
        gm = self.graph_matrices
        sp = self.seq_pos

        for source, dest in source_dest_list:
            indices = (
                gm[
                    :,
                    self.methods.head_source_index(*source),
                    self.methods.head_target_index(*dest),
                ]
                .nonzero()
                .squeeze()
            )

            gm = gm[indices]
            sp = sp[indices]

        return gm, sp

    def get_top_head_connections(self, layer_thresh=1, k=100, source_dest_list=[]):
        source_labels = self.get_source_labels()
        target_labels = self.get_target_labels()

        gm, _ = self.get_matrices_given_source_dest(source_dest_list)

        # gm = self.graph_matrices

        # for source, dest in source_dest_list:
        #     indices = (
        #         gm[
        #             :,
        #             self.methods.head_source_index(*source),
        #             self.methods.head_target_index(*dest),
        #         ]
        #         .nonzero()
        #         .squeeze()
        #     )

        #     gm = gm[indices.int()]

        gm = gm.sum(dim=0)

        total = gm[: self.methods.num_head_sources, : self.methods.num_head_targets]

        total[: layer_thresh * self.model.cfg.n_heads, :] = 0

        top_vals, top_indices = torch.topk(total.flatten(), k=k)
        top_indices = torch.stack(torch.unravel_index(top_indices, total.shape)).T

        top_list = []
        for v, source_target in zip(top_vals, top_indices):
            print(source_target)
            source, target = source_target
            source, target = source, target

            top_list.append((source_labels[source], target_labels[target], v))

        return top_list
