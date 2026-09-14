"""Tests for tinylab.presets: expand("gpt", depth) produces a valid modelcore config tree, and the
derivation rules (window sizes, value-embed parity, lambda schedule) are pinned."""
import pytest
from modelcore import ModelManager

from tinylab import presets


def test_expand_gpt_is_valid_modelcore_config():
    config = presets.expand("gpt", depth=4, max_seq_len=256, vocab_size=1024)
    report = ModelManager().validate_config(config)
    assert report.ok, report


def test_expand_unknown_preset_raises():
    with pytest.raises(ValueError, match="bogus"):
        presets.expand("bogus", depth=4)


def test_mup_dims_rounds_up_to_head_dim_multiple():
    n_embd, n_head = presets.mup_dims(depth=4, aspect_ratio=64, head_dim=128)
    assert n_embd % 128 == 0
    assert n_head == n_embd // 128


def test_compute_window_sizes_last_layer_always_full_context():
    windows = presets.compute_window_sizes("SSSL", n_layer=6, sequence_len=2048)
    assert len(windows) == 6
    assert windows[-1] == 2048  # forced full context regardless of pattern
    assert windows[0] == -(-2048 // 4 // 128) * 128  # 'S' -> quarter context, tiled to 128


def test_has_value_embed_alternates_with_last_layer_included():
    n_layer = 6
    assert presets.has_value_embed(n_layer - 1, n_layer) is True
    assert presets.has_value_embed(n_layer - 2, n_layer) is False


def test_gpt_lambda_schedule_decays_with_depth():
    resid0, x00 = presets.gpt_lambda_schedule(0, 6)
    resid_last, x0_last = presets.gpt_lambda_schedule(5, 6)
    assert resid0 > resid_last
    assert x00 > x0_last


def test_resolve_model_config_from_raw_tree_round_trips():
    config = presets.expand("gpt", depth=4, max_seq_len=256, vocab_size=1024)
    tree = ModelManager().config_to_dict(config)
    resolved = presets.resolve_model_config(tree, depth=4, aspect_ratio=64, head_dim=128, max_seq_len=256, vocab_size=1024)
    assert resolved.n_embd == config.n_embd
    assert resolved.n_layer == config.n_layer


def test_resolve_reference_config_re_expands_at_ref_depth():
    config = presets.expand("gpt", depth=6, max_seq_len=256, vocab_size=1024)
    ref = presets.resolve_reference_config(config, ref_depth=12)
    assert ref.n_layer == 12
