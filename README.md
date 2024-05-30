# Transcoder SAE Circuits

Using transcoders (for MLPs) and SAEs (for attention heads) to automatically discover circuits in large language models.

## Setup

If working on an external machine (ie a Vast.AI instance), run `source remote_setup.sh` to setup a virtualenv for this project and install all pip requirements.

## Important Files/Classes

[`src/circuit_lens.py`](src/circuit_lens.py):

- `CircuitLens` deconstructs the residual stream into contributions from Transcoder + SAE Features, in addition to bias terms, and token + pos embeddings.
- `ComponentLens` allows us to perform the equivalent of "Logit Lens" on any Feature, Unembedding Vector, or Z SAE Error term to see which previous components substantially impact subsequent components. (These runs are coordinated and organized by a given `CircuitLens` instance)

[`src/circuit_discovery.py`](src/circuit_discovery.py):

- `CircuitDiscovery` models the computation graph, where nodes are Transcoder/SAE Features, Unembeddings, Embeddings, and (optionally) Z SAE Error terms. `CircuitDiscovery` allows experimenters to employ different "strategies" for adding edges to the computation graph, effectively finding circuits that compute different outputs.
- `CircuitDiscovery` has a variety of methods for visualizing the computation graph, like `visualize_graph`, `print_attn_heads_and_mlps_in_graph`, and `visualize_attn_heads_in_graph`.
- `CircuitDiscovery` internally maintains a `CircuitLens` instance, and provides the helper method `component_lens_at_loc` to visualize `ComponentLens` runs for different components in the computation graph.

[`src/task_evaluation.py`](src/task_evaluation.py):

- Takes in a dataset of prompts from a given task (like IOI), runs `CircuitDiscovery` on these prompts, finds the frequency with which different attn heads/mlps occur in the computation graph, and computes metrics like faithfulness on this data.

[`src/max_act_analysis.py`](src/max_act_analysis.py):

- Takes in a SAE/Transcoder feature, and finds the (max) activating examples of this feature firing across OpenWebText. Includes methods to use this information to get the feature auto-interp description.

[`src/circuit_interp.py`](src/circuit_interp.py):

- `CircuitInterp` takes in an instance of `CircuitDiscovery` and a circuit discovery strategy, and attempts to label each component (MLP/Attn Head) in the computation graph based on the auto-interp descriptions of the features in the computation graph.
