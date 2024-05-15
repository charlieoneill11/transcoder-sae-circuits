import pytest
import torch
import einops

from circuit_lens import CircuitLens

ATOL = 1e-4


@pytest.fixture(scope="module")
def clens() -> CircuitLens:
    return CircuitLens(
        "Mary and Jeff went to the store, and Mary gave an apple to Jeff"
    )


def test_compare_max_attn_features(
    clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3
):
    for layer in layers:
        seq_index = clens.process_seq_index(seq_index)
        active_features = clens.get_active_features(seq_index, cache=False)

        z_sae = clens.z_saes[layer]

        layer_z = einops.rearrange(
            clens.cache["z", layer][0, seq_index],
            "n_heads d_head -> (n_heads d_head)",
        )
        _, _, z_acts, _, _ = clens.z_saes[layer](layer_z)

        z_winner_count = z_acts.nonzero().numel()

        z_values, z_max_features = z_acts.topk(k=z_winner_count)

        z_contributions = z_sae.W_dec[z_max_features.squeeze(0)] * z_values.squeeze(
            0
        ).unsqueeze(-1)

        z_contributions = einops.rearrange(
            z_contributions,
            "winners (n_head d_head) -> winners n_head d_head",
            n_head=clens.model.cfg.n_heads,
        )
        z_residual_vectors = einops.einsum(
            z_contributions,
            clens.model.W_O[layer],
            "winners n_head d_head, n_head d_head d_model -> winners d_model",
        )

        max_features_from_active_features = active_features.get_attn_feature_vectors(
            layer
        )

        all_close = torch.allclose(
            z_residual_vectors, max_features_from_active_features, atol=ATOL
        )

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                (z_residual_vectors - max_features_from_active_features).norm().item(),
            )

        assert all_close


def test_compare_sae_out(clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3):
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        layer_z = einops.rearrange(
            clens.cache["z", layer][0, seq_index],
            "n_heads d_head -> (n_heads d_head)",
        )

        test_sae_out = clens.get_active_features(
            seq_index, cache=False
        ).get_sae_out_reconstruction(layer)

        _, z_recon, _, _, _ = clens.z_saes[layer](layer_z)

        z_recon = einops.rearrange(
            z_recon,
            "(n_head d_head) -> n_head d_head",
            n_head=clens.model.cfg.n_heads,
        )
        z_recon = einops.einsum(
            z_recon,
            clens.model.W_O[layer],
            "n_head d_head, n_head d_head d_model -> d_model",
        )

        all_close = torch.allclose(z_recon, test_sae_out, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                (z_recon - test_sae_out).norm().item(),
            )

        assert all_close


def test_compare_max_mlp_features(
    clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3
):
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        mlp_transcoder = clens.mlp_transcoders[layer]
        mlp_input = clens.cache["normalized", layer, "ln2"][:, seq_index]

        _, mlp_acts, *_ = mlp_transcoder(mlp_input)

        mlp_acts = mlp_acts[0]

        mlp_winner_count = mlp_acts.nonzero().numel()

        mlp_values, mlp_max_features = mlp_acts.topk(k=mlp_winner_count)

        mlp_residual_vectors = mlp_transcoder.W_dec[
            mlp_max_features.squeeze(0)
        ] * mlp_values.squeeze(0).unsqueeze(-1)

        test_vectors = clens.get_active_features(
            seq_index, cache=False
        ).get_mlp_feature_vectors(layer)

        all_close = torch.allclose(mlp_residual_vectors, test_vectors, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                (mlp_residual_vectors - test_vectors).norm().item(),
            )

        assert all_close


def test_compare_attn_out(clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3):
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        active_features = clens.get_active_features(seq_index, cache=False)
        feature_recon = active_features.get_reconstructed_attn_out(layer)

        attn_out = clens.cache["attn_out", layer][0, seq_index]

        all_close = torch.allclose(feature_recon, attn_out, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                (feature_recon - attn_out).norm().item(),
            )

        assert all_close


def test_compare_mlp_out(clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3):
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        active_features = clens.get_active_features(seq_index, cache=False)
        feature_recon = active_features.get_reconstructed_mlp_out(layer)

        mlp_out = clens.cache["mlp_out", layer][0, seq_index]

        all_close = torch.allclose(feature_recon, mlp_out, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                (feature_recon - mlp_out).norm().item(),
            )

        assert all_close


def test_compare_transcoder_out(clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3):
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        test_transcoder_out = clens.get_active_features(
            seq_index, cache=False
        ).get_transcoder_reconstruction(layer)

        mlp_input = clens.cache["normalized", layer, "ln2"]
        out = clens.mlp_transcoders[layer](mlp_input)[0][0, seq_index]
        all_close = torch.allclose(out, test_transcoder_out, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                (out - test_transcoder_out).norm().item(),
            )

        assert all_close


def test_compare_resid_post(clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3):
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        actual_resid_post = clens.cache["resid_post", layer][0, seq_index]
        test_resid_post = clens.get_active_features(
            seq_index, cache=False
        ).get_reconstructed_resid_post(layer)

        all_close = torch.allclose(actual_resid_post, test_resid_post, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                (actual_resid_post - test_resid_post).norm().item(),
            )

        assert all_close


def test_compare_mlp_feature_activations(
    clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3
):
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        mlp_input = clens.cache["normalized", layer, "ln2"][:, seq_index]
        mlp_transcoder = clens.mlp_transcoders[layer]

        mlp_acts = mlp_transcoder(mlp_input)[1][0]

        feature_i = mlp_acts.argmax()
        feature_val = mlp_acts[feature_i]

        resid_mid = clens.get_active_features(
            seq_index, cache=False
        ).get_reconstructed_resid_mid(layer)

        device, dtype = resid_mid.device, resid_mid.dtype

        mlp_scale = clens.cache["scale", layer, "ln2"][0, seq_index]

        scaled_ev = (
            torch.ones_like(resid_mid, dtype=dtype, device=device) * resid_mid.mean()
        ) / mlp_scale

        scaled_resid = resid_mid / mlp_scale

        reconstructed_normalized = scaled_resid - scaled_ev

        value = (
            einops.einsum(
                reconstructed_normalized - mlp_transcoder.b_dec,
                mlp_transcoder.W_enc[:, feature_i],
                "d_model, d_model -> ",
            )
            + mlp_transcoder.b_enc[feature_i]
        )

        all_close = torch.allclose(value, feature_val, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                (value - feature_val).norm().item(),
            )

        assert all_close


def test_correct_comp_contributions_for_mlp_feature(
    clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3
):
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        mlp_input = clens.cache["normalized", layer, "ln2"][:, seq_index]
        mlp_transcoder = clens.mlp_transcoders[layer]

        mlp_acts = mlp_transcoder(mlp_input)[1][0]

        feature_i = mlp_acts.argmax()
        feature_val = mlp_acts[feature_i]

        vectors = clens.get_active_features(seq_index).get_vectors_before_comp(
            "mlp", layer
        )

        mlp_scale = clens.cache["scale", layer, "ln2"][0, seq_index]

        scaled_vectors = vectors / mlp_scale

        scaled_contributions = einops.einsum(
            scaled_vectors,
            mlp_transcoder.W_enc[:, feature_i],
            "comp d_model, d_model -> comp",
        )

        contrib_sum = scaled_contributions.sum()

        resid_mid = clens.get_active_features(
            seq_index, cache=False
        ).get_reconstructed_resid_mid(layer)

        device, dtype = resid_mid.device, resid_mid.dtype

        scaled_ev = (
            torch.ones_like(resid_mid, dtype=dtype, device=device) * resid_mid.mean()
        ) / mlp_scale

        test_contrib_sum = (
            feature_val
            - mlp_transcoder.b_enc[feature_i]
            + einops.einsum(
                scaled_ev + mlp_transcoder.b_dec,
                mlp_transcoder.W_enc[:, feature_i],
                "d_model, d_model -> ",
            )
        )

        all_close = torch.allclose(contrib_sum, test_contrib_sum, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                contrib_sum.item(),
                test_contrib_sum.item(),
            )

        assert all_close


def test_correct_z_feature_head_seq_decomposition(
    clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3
):
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        z_sae = clens.z_saes[layer]

        layer_z = einops.rearrange(
            clens.cache["z", layer][0, seq_index],
            "n_heads d_head -> (n_heads d_head)",
        )

        z_acts = clens.z_saes[layer](layer_z)[2]

        feature_i = z_acts.argmax()
        feature_val = z_acts[feature_i]

        feature_acts = clens.get_head_seq_activations_for_z_feature(
            layer, seq_index, feature_i
        )

        total_contrib = feature_acts.sum()

        target_value = (
            feature_val
            - z_sae.b_enc[feature_i]
            + einops.einsum(
                z_sae.b_dec,
                z_sae.W_enc[:, feature_i],
                "d_model, d_model -> ",
            )
        )

        all_close = torch.allclose(target_value, total_contrib, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                target_value.item(),
                total_contrib.item(),
            )

        assert all_close


def get_layer_norm_decomp(clens, resid, ln_type, layer, seq_index):
    device, dtype = resid.device, resid.dtype

    mlp_scale = clens.cache["scale", layer, ln_type][0, seq_index]

    ev = torch.ones_like(resid, dtype=dtype, device=device) * resid.mean()

    return (resid - ev) / mlp_scale, ev, mlp_scale, ev / mlp_scale


def test_compare_v(clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3):
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        resid_pre = clens.get_active_features(
            seq_index, cache=False
        ).get_reconstructed_resid_pre(layer)

        attn_input = get_layer_norm_decomp(clens, resid_pre, "ln1", layer, seq_index)[0]

        recon_v = (
            einops.einsum(
                attn_input,
                clens.model.W_V[layer],
                "d_model, n_head d_model d_head -> n_head d_head",
            )
            + clens.model.b_V[layer]
        )

        true_v = clens.cache["v", layer][0, seq_index]

        all_close = torch.allclose(recon_v, true_v, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                (recon_v - true_v).norm().item(),
            )

        assert all_close


def test_correct_value_contribs_for_z_feature(
    clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3
):
    """
    Assumes `test_correct_z_feature_head_seq_decomposition` passes
    """
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        layer_z = einops.rearrange(
            clens.cache["z", layer][0, seq_index],
            "n_heads d_head -> (n_heads d_head)",
        )
        z_sae = clens.z_saes[layer]
        z_acts = z_sae(layer_z)[2]

        feature_i = z_acts.argmax()

        feature_acts = clens.get_head_seq_activations_for_z_feature(
            layer, seq_index, feature_i
        )

        amax = feature_acts.argmax()

        source_i = int(amax % clens.n_tokens)
        head_i = amax // clens.n_tokens

        pattern_val = clens.cache["pattern", layer][0, head_i, seq_index, source_i]

        feature_acts_reshaped = einops.rearrange(
            feature_acts,
            "(n_head seq) -> n_head seq",
            n_head=clens.model.cfg.n_heads,
        )

        feature_value = feature_acts_reshaped[head_i, source_i]

        vectors = clens.get_active_features(
            source_i, cache=False
        ).get_vectors_before_comp("attn", layer)

        resid_pre = vectors.sum(dim=0)

        _, _, mlp_scale, scaled_ev = get_layer_norm_decomp(
            clens, resid_pre, "ln1", layer, source_i
        )

        better_w_enc = einops.rearrange(
            z_sae.W_enc, "(n_head d_head) feature -> n_head d_head feature", n_head=12
        )[head_i, :, feature_i]

        target_value = feature_value + einops.einsum(
            +einops.einsum(
                scaled_ev,
                clens.model.W_V[layer, head_i],
                "d_model, d_model d_head -> d_head",
            )
            - clens.model.b_V[layer, head_i],
            better_w_enc,
            "d_head, d_head -> ",
        )

        all_vector_contribs = (
            clens.get_v_lens_activations(layer, head_i, source_i, feature_i).sum()
            * pattern_val
            / mlp_scale
        )

        all_close = torch.allclose(target_value, all_vector_contribs, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                target_value.item(),
                all_vector_contribs.item(),
            )

        assert all_close


def test_compare_q(clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3):
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        resid_pre = clens.get_active_features(
            seq_index, cache=False
        ).get_reconstructed_resid_pre(layer)

        attn_input = get_layer_norm_decomp(clens, resid_pre, "ln1", layer, seq_index)[0]

        recon_q = (
            einops.einsum(
                attn_input,
                clens.model.W_Q[layer],
                "d_model, n_head d_model d_head -> n_head d_head",
            )
            + clens.model.b_Q[layer]
        )

        true_q = clens.cache["q", layer][0, seq_index]

        all_close = torch.allclose(recon_q, true_q, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                (recon_q - true_q).norm().item(),
            )

        assert all_close


def test_correct_query_contribs(
    clens: CircuitLens,
    layers=[0, 3, 5, 11],
    source_index=-5,
    destination_index=-3,
):
    source_index = clens.process_seq_index(source_index)
    destination_index = clens.process_seq_index(destination_index)

    for layer in layers:
        for head in range(clens.model.cfg.n_heads):
            vectors = clens.get_active_features(
                destination_index, cache=False
            ).get_vectors_before_comp("attn", layer)

            resid_pre = vectors.sum(dim=0)

            _, _, mlp_scale, scaled_ev = get_layer_norm_decomp(
                clens, resid_pre, "ln1", layer, destination_index
            )

            effective_k = clens.cache["k", layer][0, source_index, head]

            vector_values = (
                clens.get_q_lens_activations(
                    layer, head, source_index, destination_index
                )
                / mlp_scale
            )

            vector_sum = vector_values.sum(dim=0)

            bias_contrib = einops.einsum(
                clens.model.b_Q[layer, head],
                effective_k,
                "d_head, d_head -> ",
            )

            ev_contrib = einops.einsum(
                scaled_ev,
                clens.model.W_Q[layer, head],
                effective_k,
                "d_model, d_model d_head, d_head -> ",
            )

            scores = (
                clens.cache["attn_scores", layer][
                    0, head, destination_index, source_index
                ]
                * clens.model.blocks[layer].attn.attn_scale
            )

            targets = scores + ev_contrib - bias_contrib

            all_close = torch.allclose(vector_sum, targets, atol=ATOL)

            if not all_close:
                print(
                    f"Layer: {layer} Seq: {destination_index}",
                    (vector_sum - targets).flatten().norm().item(),
                )

            assert all_close


def test_compare_k(clens: CircuitLens, layers=[0, 3, 5, 11], seq_index=-3):
    seq_index = clens.process_seq_index(seq_index)

    for layer in layers:
        resid_pre = clens.get_active_features(
            seq_index, cache=False
        ).get_reconstructed_resid_pre(layer)

        attn_input = get_layer_norm_decomp(clens, resid_pre, "ln1", layer, seq_index)[0]

        recon_k = (
            einops.einsum(
                attn_input,
                clens.model.W_K[layer],
                "d_model, n_head d_model d_head -> n_head d_head",
            )
            + clens.model.b_K[layer]
        )

        true_k = clens.cache["k", layer][0, seq_index]

        all_close = torch.allclose(recon_k, true_k, atol=ATOL)

        if not all_close:
            print(
                f"Layer: {layer} Seq: {seq_index}",
                (recon_k - true_k).norm().item(),
            )

        assert all_close


def test_correct_key_contribs(
    clens: CircuitLens, layers=[0, 3, 5, 11], source_index=-5, destination_index=-3
):
    source_index = clens.process_seq_index(source_index)
    destination_index = clens.process_seq_index(destination_index)

    for layer in layers:
        for head in range(clens.model.cfg.n_heads):
            vectors = clens.get_active_features(
                source_index, cache=False
            ).get_vectors_before_comp("attn", layer)

            resid_pre = vectors.sum(dim=0)

            _, _, mlp_scale, scaled_ev = get_layer_norm_decomp(
                clens, resid_pre, "ln1", layer, source_index
            )

            effective_q = clens.cache["q", layer][0, destination_index, head]

            vector_values = (
                clens.get_k_lens_activations(
                    layer, head, source_index, destination_index
                )
                / mlp_scale
            )

            vector_sum = vector_values.sum(dim=0)

            bias_contrib = einops.einsum(
                clens.model.b_K[layer, head],
                effective_q,
                "d_head, d_head -> ",
            )

            ev_contrib = einops.einsum(
                scaled_ev,
                clens.model.W_K[layer, head],
                effective_q,
                "d_model, d_model d_head, d_head -> ",
            )

            scores = (
                clens.cache["attn_scores", layer][
                    0, head, destination_index, source_index
                ]
                * clens.model.blocks[layer].attn.attn_scale
            )

            targets = scores + ev_contrib - bias_contrib

            all_close = torch.allclose(vector_sum, targets, atol=ATOL)

            if not all_close:
                print(
                    f"Layer: {layer} Seq: {destination_index}",
                    (vector_sum - targets).flatten().norm().item(),
                )

            assert all_close
