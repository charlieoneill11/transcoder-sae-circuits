# %%
%load_ext autoreload
%autoreload 2

# %%
from ioi_dataset import NAMES, SINGLE_TOKEN_NAMES, ABBA_TEMPLATES, gen_prompt_uniform, BABA_TEMPLATES, NOUNS_DICT, gen_templated_prompts
from transformer_lens import HookedTransformer, utils, ActivationCache

# %%
model = HookedTransformer.from_pretrained("gpt2-small")

# %%
better_names = []

for name in NAMES:
    if len(model.to_str_tokens(name)) == 2:
        better_names.append(name)

# %%
better_names

# %%

all(len(model.to_str_tokens(name)) == 2 for name in better_names)

# for name in better_names:
#     print(len(model.to_str_tokens(name)) == 2:
#         better_names.append(name)

# %%
ABBA_TEMPLATES[0]

# %%
p = gen_prompt_uniform(
    [BABA_TEMPLATES[0], ABBA_TEMPLATES[0]], SINGLE_TOKEN_NAMES, NOUNS_DICT, 100, True
)

# %%
toks = model.to_tokens([' Danny', ' Joseph'], prepend_bos=False).squeeze()

model.tokens_to_residual_directions(toks).shape, toks.shape

# %%
model.tokenizer.decode(toks[0])





# %%
p[1]

# %%

for prompt in p:
    print(len(model.to_str_tokens(prompt['text'])))


# %%

all(len(model.to_str_tokens(prompt['text'])) == 17 for prompt in p)

# %%
model.to_str_tokens(p[0]['text'])[-2]



# %%
prompts = gen_templated_prompts(template_idex=3)

# %%
prompts[0]


# %%
comp = len(model.to_str_tokens(prompts[0])) 

all(len(model.to_str_tokens(prompt)) == comp for prompt in prompts)

# %%
prompts[0]

# %%
tokens = model.to_tokens(prompts)

# %%
tokens[:5, -4:]

# %%
_, cache = model.run_with_cache(tokens)

# %%
for key, value in cache.items():
    print(key, value.shape)

# %%
mean_cache = {}

for key, value in cache.items():
    mean_cache[key] = value.mean(dim=0)

mean_cache = ActivationCache(mean_cache, model)

# %%
mean_cache['z', 0].shape

