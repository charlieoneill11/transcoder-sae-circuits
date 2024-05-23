from circuit_lens import CircuitComponent
from circuit_discovery import CircuitDiscovery


def create_filter(remove_comps=[], add_comps=[], no_sae_error=False):
    if no_sae_error:
        remove_comps.append(CircuitComponent.Z_SAE_ERROR)

    base_comps = [
        CircuitComponent.Z_FEATURE,
        CircuitComponent.MLP_FEATURE,
        CircuitComponent.ATTN_HEAD,
        CircuitComponent.UNEMBED,
        # CircuitComponent.UNEMBED_AT_TOKEN,
        CircuitComponent.EMBED,
        CircuitComponent.POS_EMBED,
        # CircuitComponent.BIAS_O,
        CircuitComponent.Z_SAE_ERROR,
        # CircuitComponent.Z_SAE_BIAS,
        # CircuitComponent.TRANSCODER_ERROR,
        # CircuitComponent.TRANSCODER_BIAS,
    ]

    base_comps.extend(add_comps)
    base_comps = [comp for comp in base_comps if comp not in remove_comps]

    def filter(component: str):
        return component in base_comps

    return filter


def create_simple_greedy_strategy(
    passes=5,
    node_contributors=1,
    minimal=True,
    do_sub_pass=False,
    sub_passes=1,
    sub_pass_layer_threshold=-1,
    sub_pass_minimal=True,
):
    def strategy(cd: CircuitDiscovery):
        for _ in range(passes):
            cd.add_greedy_pass(contributors_per_node=node_contributors, minimal=minimal)

            if do_sub_pass:
                for _ in range(sub_passes):
                    cd.add_greedy_pass_against_all_existing_nodes(
                        contributors_per_node=node_contributors,
                        skip_z_features=True,
                        layer_threshold=sub_pass_layer_threshold,
                        minimal=sub_pass_minimal,
                    )

    return strategy


def create_top_contributor_strategy(
    num_greedy_passes=20,
    num_contributor_per_pass=1,
    layer_threshold=-1,
):
    def strategy(cd: CircuitDiscovery):
        for _ in range(num_greedy_passes):
            cd.greedily_add_top_contributors(
                k=num_contributor_per_pass, reciever_threshold=layer_threshold
            )

    return strategy
