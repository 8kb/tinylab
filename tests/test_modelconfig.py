"""Tests for tinylab.modelconfig: loading and validating a materialized ModelConfig tree dumped
by nanochat's scripts/model_info.py --dump-config. No preset/depth-dial derivation happens here
any more -- see AGENTS.md."""
import json
import os

import pytest
from modelcore import ModelManager

from tinylab import modelconfig

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "gpt_tiny.json")


def test_load_model_config_hydrates_a_valid_tree():
    config = modelconfig.load_model_config(_FIXTURE, sequence_len=32, vocab_size=32768)
    report = ModelManager().validate_config(config)
    assert report.ok, report
    assert config.n_layer == 2


def test_load_model_config_rejects_sequence_len_mismatch():
    with pytest.raises(AssertionError, match="sequence_len"):
        modelconfig.load_model_config(_FIXTURE, sequence_len=64, vocab_size=32768)


def test_load_model_config_rejects_vocab_size_mismatch():
    with pytest.raises(AssertionError, match="vocab_size"):
        modelconfig.load_model_config(_FIXTURE, sequence_len=32, vocab_size=50257)


def test_load_model_config_round_trips_through_to_dict(tmp_path):
    config = modelconfig.load_model_config(_FIXTURE, sequence_len=32, vocab_size=32768)
    tree = ModelManager().config_to_dict(config)
    dumped = tmp_path / "round_trip.json"
    dumped.write_text(json.dumps(tree))
    reloaded = modelconfig.load_model_config(str(dumped), sequence_len=32, vocab_size=32768)
    assert reloaded.n_embd == config.n_embd
    assert reloaded.n_layer == config.n_layer


def _v1_copy(tmp_path):
    """The fixture the way v1 wrote it: no template/norm/mlp/defaults, full attention spelled
    window=sequence_len -- what an old checkpoint's config (or an old job's model_config) looks like."""
    d = json.load(open(_FIXTURE))
    d["format"] = "modelcore.v1"
    for key in ("template", "_comment"):
        d.pop(key, None)
    del d["shared"]["norm"]
    del d["shared"]["rope"]["over_compute"]
    del d["output"]["softcap"]
    for block in d["body"]["blocks"]:
        block.pop("_comment", None)
        del block["mlp"]
        block["window"] = 32  # == sequence_len: v1's spelling of full attention
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(d))
    return str(path)


def _without_comments(node):
    """The same dict with every `_`-prefixed key dropped, at any depth."""
    if isinstance(node, dict):
        return {k: _without_comments(v) for k, v in node.items() if not k.startswith("_")}
    if isinstance(node, list):
        return [_without_comments(v) for v in node]
    return node


def test_a_v1_model_config_file_still_loads_and_builds_the_same_model(tmp_path):
    """modelcore upgrades it on load, writing out everything v1 left implicit -- so an old job's
    model_config keeps working, and builds exactly the tree the v2 fixture describes."""
    v1 = modelconfig.load_model_config(_v1_copy(tmp_path), sequence_len=32, vocab_size=32768)
    v2 = modelconfig.load_model_config(_FIXTURE, sequence_len=32, vocab_size=32768)
    assert ModelManager().validate_config(v1).ok
    assert v1.to_dict() == _without_comments(v2.to_dict())
    assert [b.params["window"] for b in v1.body.params["blocks"]] == [-1, -1]  # full attention is -1 now


def test_comments_in_a_model_config_survive_loading():
    config = modelconfig.load_model_config(_FIXTURE, sequence_len=32, vocab_size=32768)
    assert "sequence_len 32" in config.comments["_comment"]
    assert "no defaults" in config.body.params["blocks"][0].comments["_comment"]
    assert ModelManager().validate_config(config).ok  # ...and never trip validation or a constructor


def test_a_mistyped_top_level_key_is_an_error_not_silently_dropped(tmp_path):
    d = json.load(open(_FIXTURE))
    d["adaptrs"] = []
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="unknown top-level key"):
        modelconfig.load_model_config(str(path), sequence_len=32, vocab_size=32768)


def test_the_mismatch_messages_name_the_real_job_key():
    """They used to say "model.config", a key that doesn't exist -- so a user grepping their job
    file for it found nothing."""
    with pytest.raises(AssertionError) as e:
        modelconfig.load_model_config(_FIXTURE, sequence_len=64, vocab_size=32768)
    assert '"model_config"' in str(e.value) and "model.config" not in str(e.value)
