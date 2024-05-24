# %%
%load_ext autoreload
%autoreload 2

# %%
import einops

from autointerpretability import *


# %%
model = HookedTransformer.from_pretrained('gpt2-small')

dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=10_000)
tokenized_owt = tokenize_and_concatenate(dataset, model.tokenizer, max_length=128, streaming=True)
tokenized_owt = tokenized_owt.shuffle(42)
tokenized_owt = tokenized_owt.take(12800 * 2)
owt_tokens = np.stack([x['tokens'] for x in tokenized_owt])
owt_tokens_torch = torch.tensor(owt_tokens)

device = 'cpu'
tl_model, z_saes, transcoders = get_model_encoders(device=device)

# %%
features = [16513, 7861]
sae = z_saes[8]
feature_scores = get_feature_scores(model, sae, owt_tokens_torch, features, batch_size=64)

# %%
feature_scores = torch.tensor(feature_scores)
abba_feature = feature_scores[:, 0, :]

# %%
indices = abba_feature.flatten().argsort(descending=True)

seq_pos = torch.stack(
    torch.unravel_index(indices, abba_feature.shape)
).T

# %%
seq_pos[0]

# %%
abba_feature[3245, 88]

# %%
model.tokenizer.decode(owt_tokens_torch[3245])

# %%
model.tokenizer.decode(owt_tokens_torch[3245][88])

# %%

feature_idx = 0 # corresponding to 16513
example_html, examples_clean_text = display_top_k_activating_examples(model, feature_scores[:, 0, :].numpy(), owt_tokens_torch, k=10, show_score=True)

# %%
toks = owt_tokens_torch[[3245, 3, 4]]

# %%

_, cache = model.run_with_cache(toks)

# %%
pos = 88

z_flat = einops.rearrange(cache['z', 8][:, pos], "b n d -> b (n d)")
acts = sae.encode(z_flat)

# print(acts[:, 16513].item())

# %%
utils.get_act_name('z', 8)

# %%
acts[:, 16513]


# %%
z_flat = einops.rearrange(cache['z', 8], "b pos n d -> (b pos) (n d)")
z_flat_2 = cache['z', 8].reshape(-1, sae.W_enc.shape[0])

# %%

# %%
z_flat.shape

# %%
acts = sae.encode(z_flat)
acts.shape


# %%

acts[:, [16513, 7861]].shape

# %%

curr_scores = acts[:, [16513, 7861]]

# %%
first_re = einops.rearrange(curr_scores, "(b pos) n -> b n pos", b=3)
second_re = curr_scores.reshape(-1, 2, 128)

# %%

(first_re - second_re).abs().max()

# %%
first_re.shape, second_re.shape

# %%
def get_feature_scores(
    model,
    encoder,
    tokens_arr,
    feature_indices,
    batch_size=64,
    act_name=None,
):

    # Determine the type of encoder and set defaults
    # if isinstance(encoder, ZSAE):
    #     print("ZSAE")
    act_name = act_name or "attn.hook_z"
    layer = encoder.cfg["layer"]
    name_filter = f"blocks.{layer}.attn.hook_z"
    # elif isinstance(encoder, SparseTranscoder):
    #     print("SparseTranscoder")
    #     act_name = act_name or encoder.cfg.hook_point
    #     layer = encoder.cfg.hook_point_layer
    #     name_filter = act_name
    # else:
    #     raise ValueError("Unsupported encoder type")

    scores = []

    for i in tqdm(range(0, tokens_arr.shape[0], batch_size)):
        with torch.no_grad():
            curr_tokens = tokens_arr[i : i+batch_size]

            _, cache = model.run_with_cache(
                curr_tokens,
                stop_at_layer=layer + 1,
                names_filter=name_filter,
            )
            acts = cache[name_filter]
            # acts_flat = acts.reshape(-1, encoder.W_enc.shape[0])
            acts_flat = einops.rearrange(acts, "b pos n d -> (b pos) (n d)")


            hidden_acts = encoder.encode(acts_flat)
            curr_scores = hidden_acts[:, feature_indices]
            del hidden_acts


        scores.append(
            einops.rearrange(curr_scores, "(b pos) n -> b n pos", b=curr_tokens.size(0))
        )

    return torch.concat(scores, dim=0)

# %%
scores = get_feature_scores(model, sae, owt_tokens_torch[:1000], [16513, 7861], batch_size=64)

# %%

# %%
scores.shape

abba_feature = scores[:, 0, :]

indices = abba_feature.flatten().argsort(descending=True)

seq_pos = torch.stack(
    torch.unravel_index(indices, abba_feature.shape)
).T

# %%
seq, pos = seq_pos[0]

toks = owt_tokens_torch[seq]

_, cache = model.run_with_cache(toks)

z_flat = einops.rearrange(cache['z', 8][:, pos], "b n d -> b (n d)")
acts = sae.encode(z_flat)

acts[:, 16513]


# %%
feature_idx = 0 # corresponding to 16513
example_html, examples_clean_text = display_top_k_activating_examples(model, scores[:, 1, :].cpu().numpy(), owt_tokens_torch, k=10, show_score=True)

# %%
