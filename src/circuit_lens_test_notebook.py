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
circuit_lens = CircuitLens("14. Colorado 15. Missouri 16. Illinois 17")


# %%
unembed_children = circuit_lens.get_unembed_lens_for_prompt_token(-2)

# %%
l9 = unembed_children[3]()

# %%
# %%
l9_v = l9[0]('v')

# %%
