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
# circuit_lens = CircuitLens("14. Colorado 15. Missouri 16. Illinois 17")

circuit_lens = CircuitLens("Mary and Jeff went to the store, and Mary gave an apple to Jeff")


# %%
unembed_children = circuit_lens.get_unembed_lens_for_prompt_token(-2)

# %%
l9 = unembed_children[0]()

# %%
# %%
l9_q = l9[0]('q')

# %%
l8 = l9_q[0]()

# %%
l8_q = l8[0]('q')



# %%
l8_q = l8[0]('q')
# %%
l8_k = l8[0]('k')

# %%
l8_k[4]()

# %%


l7 = l8_q[3]()

# %%
l7_k = l7[0]('k')
# %%
l7_k[2]()
