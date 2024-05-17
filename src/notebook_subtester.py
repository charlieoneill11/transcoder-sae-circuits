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
test_correct_value_contribs_for_z_feature(lens)  # , layers=[11])
test_compare_q(lens)
test_compare_k(lens)
test_correct_query_contribs(lens)
test_correct_key_contribs(lens)

# %%