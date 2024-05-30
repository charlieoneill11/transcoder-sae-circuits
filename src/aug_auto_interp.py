# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
import yaml
import einops
from openai import OpenAI

from openai_utils import gen_openai_completion
from transformer_lens import HookedTransformer
from z_sae import ZSAE


# %%
out = gen_openai_completion("Why is the sky blue? (2 sentence answer)")

# %%
model = HookedTransformer.from_pretrained('gpt2')

# %%
layer = 8
zsae = ZSAE.load_zsae_for_layer(layer)

# %%
f = zsae.W_dec[16513]
# f = zsae.W_dec[15647]
f = einops.rearrange(f, "(n d) -> n d", n=12)

# %%
ll = einops.einsum(f, model.W_O[layer], "n d, n d o -> o")

# %%
r = einops.einsum(ll, model.W_U, "d, d c -> c")
r = r.softmax()

# %%
values, indices = r.topk(10)


# %%
for i, v in zip(indices, values):
    print(model.tokenizer.decode(i), v.item())

# %%
r.shape

# %%
values[0]

# %%
