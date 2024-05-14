# %%

%load_ext autoreload
%autoreload 2


# %%

from circuit_lens import CircuitLens

# %%

lens = CircuitLens("Mary and Jeff went to the store, and Mary gave an apple to Jeff")

# %%
from test_circuit_lens import *

test_compare_max_attn_features(lens)
test_compare_sae_out(lens)
test_compare_max_mlp_features(lens)
test_compare_attn_out(lens)
test_compare_mlp_out(lens)
test_compare_transcoder_out(lens)
test_compare_resid_post(lens)
test_compare_mlp_feature_activations(lens)
test_correct_comp_contributions_for_mlp_feature(lens)
test_correct_z_feature_head_seq_decomposition(lens)
test_compare_v(lens)

# %%
from test_circuit_lens import *
test_correct_value_contribs_for_z_feature(lens, layers=[6])

# %%
torch.tensor([1, 4,3, 34,23]).argmax()

# %%
lens.model.W_V[0].shape



# %%
layer = 0

mlp_in = lens.cache['normalized', layer, 'ln2'][:, lens.process_seq_index(-3)]
mlp_transcoder = lens.mlp_transcoders[layer]
mlp_acts = mlp_transcoder(mlp_in)[1][0]

# %%
feature_i = mlp_acts.argmax()
mlp_acts[mlp_acts.argmax()]

# %%
(einops.einsum(
    mlp_in[0] - mlp_transcoder.b_dec,
    mlp_transcoder.W_enc[:, feature_i],
    "d_model, d_model -> ",
) + mlp_transcoder.b_enc[feature_i])

# %%
mlp_transcoder.W_dec.shape

# %%
