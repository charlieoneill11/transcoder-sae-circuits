from typing import Union, Optional, List, Dict
from circuit_lens import CircuitLens, ComponentLens
from dataclasses import dataclass, field


class CircuitDiscoveryRegularNode:
    def __init__(self, component_lens: ComponentLens, children_rank: int, k=10):
        self.component_lens = component_lens
        self.children_rank = children_rank

        self.k = k

        self.explored_children: List["CircuitDiscoveryNode"] = []

    _children_components: Optional[List[ComponentLens]] = None

    @property
    def children_components(self) -> List[ComponentLens]:
        if self._children_components is None:
            self._children_components = self.component_lens(visualize=False, k=self.k)

        return self._children_components


class CircuitDiscoveryHeadNode:
    def __init__(self, component_lens: ComponentLens, children_rank: int, k=10):
        self.component_lens = component_lens
        self.children_rank = children_rank

        self.k = k

        self.explored_q_children: List["CircuitDiscoveryNode"] = []
        self.explored_k_children: List["CircuitDiscoveryNode"] = []
        self.explored_v_children: List["CircuitDiscoveryNode"] = []

    _children_components: Dict[str, List[ComponentLens]]

    def children_components(self, head_type) -> List[ComponentLens]:
        assert head_type in ["q", "k", "v"]

        if head_type not in self._children_components:
            self._children_components[head_type] = self.component_lens(
                head_type=head_type, visualize=False, k=self.k
            )

        return self._children_components[head_type]


CircuitDiscoveryNode = Union[CircuitDiscoveryRegularNode, CircuitDiscoveryHeadNode]


class CircuitDiscovery:
    def __init__(self, prompt, seq_index: int, k=10):
        self.lens = CircuitLens(prompt)
        self.seq_index = self.lens.process_seq_index(seq_index)

        self.k = k

        self.root_node = CircuitDiscoveryRegularNode(
            ComponentLens.create_root_unembed_lens(self.lens, self.seq_index),
            -1,
            self.k,
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
