# %%
%load_ext autoreload
%autoreload 2

# %%
from example_prompts import SUCCESSOR_EXAMPLE_PROMPT, IOI_EXAMPLE_PROMPT
from exo_genesis import CircuitDiscovery, only_feature


import time


# %%

# %%
start = time.time()

cd = CircuitDiscovery(SUCCESSOR_EXAMPLE_PROMPT, -2, allowed_components_filter=only_feature)

cd.add_greedy_first_pass()

print("\n Elapsed", time.time() - start)

# %%
cd.visualize_graph()

# %%
