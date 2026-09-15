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
