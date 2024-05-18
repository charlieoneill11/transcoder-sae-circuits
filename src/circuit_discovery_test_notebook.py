# %%
%load_ext autoreload
%autoreload 2

# %%
from circuit_discovery import CircuitDiscovery
from example_prompts import IOI_EXAMPLE_PROMPT
from circuit_lens import CircuitComponent

# %%

def g(**kwargs):
    print("gg", kwargs)

def fn(**kwargs):

    print(kwargs, 'a' in kwargs)

    g(**kwargs)

fn(a=1, b=2)

# %%
a = {}
a.get('a', 1)

class B:
    a = 1

b = B()

# %%
setattr(b, 'a', 2)




# %%

disc = CircuitDiscovery(IOI_EXAMPLE_PROMPT)


# %%

disc.lens


# %%
