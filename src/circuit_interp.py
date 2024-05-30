import torch

from dataclasses import dataclass
from circuit_discovery import (
    CircuitDiscovery,
    CircuitDiscoveryHeadNode,
    CircuitDiscoveryRegularNode,
    CircuitDiscoveryNode,
)
from circuit_lens import get_model_encoders, CircuitComponent
from jinja2 import Template
from text_utils import dedent
from max_act_analysis import MaxActAnalysis
from typing import Dict, Union, Tuple, List
from openai_utils import gen_openai_completion
from circuit_interp_prompts import attn_head_max_act_prompt
from aug_interp_prompts import extract_explanation


def tm(s: str) -> Template:
    return Template(dedent(s))


@dataclass
class LabeledFeature:
    node: CircuitDiscoveryNode
    explanation: str


class CircuitInterp:
    def __init__(
        self,
        circuit_discovery: CircuitDiscovery,
        discovery_strategy,
        num_max_act_seqs_in_prompt: int = 10,
        num_open_web_text_seq: int = 5000,
    ):
        self.circuit_discovery = circuit_discovery
        self.discovery_strategy = discovery_strategy

        self.num_max_act_seqs_in_prompt = num_max_act_seqs_in_prompt
        self.num_open_web_text_seq = num_open_web_text_seq

        self.child_max_acts_cache: Dict[Tuple, Dict[str, MaxActAnalysis]] = {}

    @property
    def component_filter(self):
        return self.circuit_discovery.allowed_components_filter

    @property
    def model(self):
        return self.circuit_discovery.model

    @property
    def final_token(self) -> str:
        return self.model.tokenizer.decode(self.circuit_discovery.token)

    def interpret_heads_in_circuit(self, layer_threshold=-1, visualize=False):
        queue: List[LabeledFeature] = []
        visited_ids = []

        queue.append(
            LabeledFeature(
                node=self.circuit_discovery.root_node,
                explanation=f"This neuron causes the model to strongly predict '{self.final_token}' as the next token.",
            )
        )

        explanation_dict = {}

        while queue:
            next_task = queue.pop(0)

            node = next_task.node

            while node is not None and isinstance(node, CircuitDiscoveryRegularNode):
                node = node.sorted_contributors_in_graph[0]

                if node.component not in [
                    CircuitComponent.Z_FEATURE,
                    CircuitComponent.MLP_FEATURE,
                    CircuitComponent.ATTN_HEAD,
                ]:
                    node = None
                    break

            if not isinstance(node, CircuitDiscoveryHeadNode):
                continue

            if node.tuple_id in visited_ids or node.layer <= layer_threshold:
                continue

            (
                head_label,
                head_explanation,
                query_explanation,
                key_explanation,
                value_explanation,
                _,
            ) = self.get_attn_head_interp(
                node, next_task.explanation, visualize=visualize
            )

            explanation_dict[node.tuple_id] = (head_label, head_explanation)

            queue.extend(
                [
                    LabeledFeature(
                        node=node.sorted_contributors_in_graph("q")[0],
                        explanation=query_explanation,
                    ),
                    LabeledFeature(
                        node=node.sorted_contributors_in_graph("k")[0],
                        explanation=key_explanation,
                    ),
                    LabeledFeature(
                        node=node.sorted_contributors_in_graph("v")[0],
                        explanation=value_explanation,
                    ),
                ]
            )

        return explanation_dict

    def get_attn_head_interp(
        self,
        node: CircuitDiscoveryHeadNode,
        output_description: str,
        source_lr=("[[", "]]"),
        dest_lr=("{{", "}}"),
        token_lr=("<<", ">>"),
        context_lr=("[[", "]]"),
        visualize: bool = False,
    ):
        if node.tuple_id not in self.child_max_acts_cache:
            child_nodes: Dict[str, CircuitDiscoveryNode] = {}

            for head_type in ["q", "k", "v"]:
                top_node = node.sorted_contributors_in_graph(head_type)[0]

                if (
                    isinstance(top_node, CircuitDiscoveryRegularNode)
                    and top_node.component == CircuitComponent.Z_FEATURE
                ):
                    child_nodes[head_type] = top_node.sorted_contributors_in_graph[0]
                else:
                    child_nodes[head_type] = top_node

            for child_node in child_nodes.values():
                if child_node.component not in [
                    CircuitComponent.MLP_FEATURE,
                    CircuitComponent.ATTN_HEAD,
                ]:
                    raise NotImplementedError(
                        "Only MLP and Z features are supported right now"
                    )

            def feature_type(node):
                if node.component == CircuitComponent.MLP_FEATURE:
                    return "mlp"
                else:
                    return "attn"

            child_max_acts = {}

            for head_type in ["q", "k", "v"]:
                child_node = child_nodes[head_type]

                child_max_acts[head_type] = MaxActAnalysis(
                    feature_type=feature_type(child_node),
                    layer=child_node.layer,
                    feature=child_node.feature,
                    num_sequences=self.num_open_web_text_seq,
                )

            self.child_max_acts_cache[node.tuple_id] = child_max_acts

        child_max_acts = self.child_max_acts_cache[node.tuple_id]

        layer = node.layer
        head = node.head
        sl, sr = source_lr
        dl, dr = dest_lr
        tl, tr = token_lr
        cl, cr = context_lr

        source_token = self.circuit_discovery.str_tokens[node.source]
        dest_token = self.circuit_discovery.str_tokens[node.dest]

        sequence = "".join(self.circuit_discovery.str_tokens) + self.final_token
        source_dest_annotated_seq = ""

        for i, tok in enumerate(self.circuit_discovery.str_tokens):
            if i == node.source:
                source_dest_annotated_seq += f"{sl}{tok}{sr}"
            elif i == node.dest:
                source_dest_annotated_seq += f"{dl}{tok}{dr}"
            else:
                source_dest_annotated_seq += tok

        source_dest_annotated_seq += self.final_token

        head_label = f"Attention Head L{layer}H{head}"

        prompt = tm(attn_head_max_act_prompt).render(
            {
                "head_label": head_label,
                "head_output_description": output_description,
                "tr": tr,
                "tl": tl,
                "cl": cl,
                "cr": cr,
                "sr": sr,
                "sl": sl,
                "dr": dr,
                "dl": dl,
                "source_token": source_token,
                "dest_token": dest_token,
                "source_dest_annotated_seq": source_dest_annotated_seq,
                "sequence": sequence,
                "query_examples": child_max_acts[
                    "q"
                ].get_context_referenced_prompts_for_range(
                    0,
                    self.num_max_act_seqs_in_prompt,
                ),
                "top_query_tokens": child_max_acts["q"].get_top_k_tokens(),
                "key_examples": child_max_acts[
                    "k"
                ].get_context_referenced_prompts_for_range(
                    0,
                    self.num_max_act_seqs_in_prompt,
                ),
                "top_key_tokens": child_max_acts["k"].get_top_k_tokens(),
                "value_examples": child_max_acts[
                    "v"
                ].get_context_referenced_prompts_for_range(
                    0,
                    self.num_max_act_seqs_in_prompt,
                ),
                "top_value_tokens": child_max_acts["v"].get_top_k_tokens(),
            }
        )

        res = gen_openai_completion(prompt, visualize_stream=visualize)

        query_explanation = extract_explanation(res, delim="QUERY_NEURON_EXPLANATION")
        key_explanation = extract_explanation(res, delim="KEY_NEURON_EXPLANATION")
        value_explanation = extract_explanation(res, delim="VALUE_NEURON_EXPLANATION")

        head_explanation = extract_explanation(
            res,
            delim="ATTENTION_HEAD_COMPUTATION_DESCRIPTION",
            # delim="ATTENTION_HEAD_COMPUTATION_HYPOTHESIS",
        )

        head_label = extract_explanation(res, delim="ATTENTION_HEAD_LABEL")

        return (
            head_label,
            head_explanation,
            query_explanation,
            key_explanation,
            value_explanation,
            prompt,
        )
