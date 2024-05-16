from typing import Union, Optional, List, Dict, Tuple, Set
from circuit_lens import CircuitLens, ComponentLens, CircuitComponent
from abc import ABC, abstractmethod


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
    def __init__(self, attn_head_layer: "AttnHeadAnalysisLayer"):
        super().__init__(attn_head_layer.circuit_discovery)

        self.attn_head_layer = attn_head_layer


class AttnHeadAnalysisLayer(AnalysisObject):
    def __init__(self, transformer_layer: "TransformerAnalysisLayer"):
        super().__init__(transformer_layer.circuit_discovery)

        self.transformer_layer = transformer_layer

        self.attn_heads_at_seq = [
            AttnHeadAnalysisNode(self) for _ in range(self.n_tokens)
        ]


class TransformerAnalysisLayer(AnalysisObject):
    def __init__(self, transformer_model: "TransformerAnalysisModel"):
        super().__init__(transformer_model.circuit_discovery)

        self.transformer_model = transformer_model

        self.mlps_at_seq = [MlpAnalysisNode(self) for _ in range(self.n_tokens)]
        self.z_features_at_seq = [
            ZFeatureAnalysisNode(self) for _ in range(self.n_tokens)
        ]

        self.attn_head_layers = [
            AttnHeadAnalysisLayer(self) for _ in range(self.model.cfg.n_heads)
        ]

    @property
    def circuit_discovery(self):
        return self.transformer_model.circuit_discovery


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


class TransformerAnalysisModel(AnalysisObject):
    discovery_node_cache: Dict[Tuple, "CircuitDiscoveryNode"]

    embed_nodes: Dict[int, "CircuitDiscoveryNode"]
    pos_embed_nodes: Dict[int, "CircuitDiscoveryNode"]

    unembed_nodes: Dict[int, "CircuitDiscoveryNode"]

    def __init__(self, circuit_discovery: "CircuitDiscovery"):
        super().__init__(circuit_discovery)

        self.layers = [
            TransformerAnalysisLayer(self) for _ in range(self.model.cfg.n_layers)
        ]

        self.edge_tracker = ComponentEdgeTracker(self)

    def add_contributor_edge(
        self, reciever: "CircuitDiscoveryNode", contributor_lens: ComponentLens
    ):
        contributor_node = self.get_discovery_node_at_locator(contributor_lens)

        self.edge_tracker.add_edge(reciever.tuple_id, contributor_node.tuple_id)

    def get_contributors_in_graph(
        self, reciever: "CircuitDiscoveryNode"
    ) -> List["CircuitDiscoveryNode"]:
        return [
            self.get_discovery_node_at_locator(tuple_id)
            for tuple_id in self.edge_tracker.get_contributors(reciever.tuple_id)
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

        if tuple_id in self.discovery_node_cache:
            return self.discovery_node_cache[tuple_id]

        component = tuple_id[0]

        if component == CircuitComponent.UNEMBED:
            raise NotImplementedError("Havne't implemented generic unembed yet!")
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


class CircuitDiscoveryNode(ABC):
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


class CircuitDiscoveryRegularNode(CircuitDiscoveryNode):
    def __init__(
        self, component_lens: ComponentLens, circuit_discovery: "CircuitDiscovery"
    ):
        super().__init__(component_lens, circuit_discovery)

        # self.explored_children: List["CircuitDiscoveryNode"] = []

    _top_k_contributors: Optional[List[ComponentLens]] = None

    @property
    def top_k_contributors(self) -> List[ComponentLens]:
        if self._top_k_contributors is None:
            self._top_k_contributors = self.component_lens(visualize=False, k=self.k)

        return self._top_k_contributors

    @property
    def contributors_in_graph(self) -> List["CircuitDiscoveryNode"]:
        return self.transformer_model.get_contributors_in_graph(self)

    def add_contributor_edge(self, component_lens: ComponentLens):
        """
        This method essentially adds an edge from the current node to one of the top k
        contributors.  In our TransformerModel, this node can now recieve a contribution
        of signal from the contributor component.
        """
        self.transformer_model.add_contributor_edge(self, component_lens)


class CircuitDiscoveryHeadNode(CircuitDiscoveryNode):
    def __init__(
        self, component_lens: ComponentLens, circuit_discovery: "CircuitDiscovery"
    ):
        super().__init__(component_lens, circuit_discovery)

    _top_k_contributors: Dict[str, List[ComponentLens]]

    def top_k_contributors(self, head_type) -> List[ComponentLens]:
        assert head_type in ["q", "k", "v"]

        if head_type not in self._top_k_contributors:
            self._top_k_contributors[head_type] = self.component_lens(
                head_type=head_type, visualize=False, k=self.k
            )

        return self._top_k_contributors[head_type]


class CircuitDiscovery:
    def __init__(self, prompt, seq_index: int, k=10):
        self.lens = CircuitLens(prompt)
        self.seq_index = self.lens.process_seq_index(seq_index)

        self.k = k

        self.transformer_model = TransformerAnalysisModel(self)

        self.root_node = self.transformer_model.get_discovery_node_at_locator(
            ComponentLens.create_root_unembed_lens(self.lens, self.seq_index),
        )

    @property
    def model(self):
        return self.lens.model

    @property
    def n_tokens(self):
        return self.lens.n_tokens

    @property
    def z_saes(self):
        return self.lens.z_saes

    @property
    def mlp_transcoders(self):
        return self.lens.mlp_transcoders

    def run(self):
        pass
