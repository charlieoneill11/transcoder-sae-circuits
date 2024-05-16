from dataclasses import dataclass, field
from typing import Union, Optional, List, Dict

@dataclass
class Node:
    component_lens: ComponentLens
    children: List["Node"] = field(default_factory=list)

@dataclass
class Path:
    nodes: List[Node] = field(default_factory=list)


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

    _children_components: Dict[str, List[ComponentLens]] = {}

    def children_components(self, head_type) -> List[ComponentLens]:
        assert head_type in ["q", "k", "v"]
        if head_type not in self._children_components:
            self._children_components[head_type] = self.component_lens(
                head_type=head_type, visualize=False, k=self.k
            )
        return self._children_components[head_type]


class CircuitDiscovery:
    def __init__(self, prompt, seq_index: int, k=10):
        self.lens = CircuitLens(prompt)
        self.seq_index = self.lens.process_seq_index(seq_index)
        self.k = k

        root_nodes = self.lens.get_unembed_lens_for_prompt_token(self.seq_index, visualize=False)
        filtered_root_nodes = filter_children_nodes(root_nodes)

        self.root_node = CircuitDiscoveryRegularNode(
            filtered_root_nodes[0],
            -1,
            self.k,
        )
        print(f"Root node: {self.root_node.component_lens.run_data}")
        self.computational_graph: List[Path] = []

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

    def explore_node(self, node: CircuitDiscoveryNode, path: Path):
        print(f"Exploring node: {node.component_lens.run_data}")
        if isinstance(node, CircuitDiscoveryRegularNode):
            children_components = node.children_components
            print(f"Children components: {children_components}")
            if not children_components:
                print(f"No children for node: {node.component_lens.run_data}")
                return

            filtered_children = filter_children_nodes(children_components)
            print(f"Filtered children: {filtered_children}")

            for child in filtered_children:
                if child.run_data['run_type'] == "z_feature_head_seq":
                    head_node = CircuitDiscoveryHeadNode(child, node.children_rank, self.k)
                    self.explore_head_node(head_node, path)
                else:
                    new_node = CircuitDiscoveryRegularNode(child, node.children_rank, self.k)
                    node.explored_children.append(new_node)
                    path.nodes.append(Node(component_lens=child))
                    self.explore_node(new_node, path)

        elif isinstance(node, CircuitDiscoveryHeadNode):
            self.explore_head_node(node, path)

    def explore_head_node(self, head_node: CircuitDiscoveryHeadNode, path: Path):
        print(f"Exploring head node: {head_node.component_lens.run_data}")
        for head_type in ['q', 'k', 'v']:
            head_children = head_node.children_components(head_type=head_type)
            print(f"Head children ({head_type}): {head_children}")
            if not head_children:
                print(f"No {head_type} children for head node: {head_node.component_lens.run_data}")
                continue

            filtered_head_children = filter_children_nodes(head_children)
            print(f"Filtered {head_type} children: {filtered_head_children}")

            for child in filtered_head_children:
                new_node = CircuitDiscoveryRegularNode(child, head_node.children_rank, self.k)
                if head_type == 'q':
                    head_node.explored_q_children.append(new_node)
                elif head_type == 'k':
                    head_node.explored_k_children.append(new_node)
                elif head_type == 'v':
                    head_node.explored_v_children.append(new_node)
                path.nodes.append(Node(component_lens=child))
                self.explore_node(new_node, path)

    def run(self):
        initial_path = Path(nodes=[Node(component_lens=self.root_node.component_lens)])
        self.explore_node(self.root_node, initial_path)
        self.computational_graph.append(initial_path)
        return self.computational_graph
    
def print_path(path: Path):
    for i, node in enumerate(path.nodes):
        component_lens = node.component_lens
        run_data = component_lens.run_data
        component_str = f"{component_lens.component} | Layer: {run_data.get('layer', 'N/A')} | Seq Index: {run_data.get('seq_index', 'N/A')} | Feature: {run_data.get('feature', 'N/A')}"
        print(f"Step {i+1}: {component_str}")

def print_computational_graph(computational_graph):
    for i, path in enumerate(computational_graph):
        print(f"\nPath {i+1}:")
        print_path(path)
