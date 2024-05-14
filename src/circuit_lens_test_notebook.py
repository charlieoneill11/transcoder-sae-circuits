# %%
%load_ext autoreload
%autoreload 2

# %%
from circuit_lens import CircuitLens
from memory import get_gpu_memory
from plotly_utils import *
from pprint import pprint
import torch

from torch import tensor

# %%
circuit_lens = CircuitLens("Mary and Jeff went to the store, and Mary gave an apple to Jeff")

# %%
unembed_children = circuit_lens.get_unembed_lens_for_prompt_token(-2)

# %%
l9 = unembed_children[0]()

# %%
# %%
l9_v = l9[0]('v')
