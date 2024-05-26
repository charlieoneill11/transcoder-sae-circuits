# %%
import sys
import yaml
from openai import OpenAI

from openai_utils import gen_openai_completion

# %%
out = gen_openai_completion("Why is the sky blue? (2 sentence answer)")
