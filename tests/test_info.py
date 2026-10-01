"""`python -m tinylab info`: the read-only inspector. Config files and checkpoint tags (only the
meta is read -- a fake checkpoint here has an empty model file), the JSON shape, and that the plan
it reports is exactly modelcore's derive_training_plan on the same numbers."""
import json
import os

import pytest

from modelcore import ModelConfig, ModelManager
from modelcore import derive_training_plan

from conftest import TINY_GPT_CONFIG
from tinylab import info


def _run(argv, capsys):
    rc = info.main(argv)
    return rc, capsys.readouterr().out


def _tiny_stats():
    with open(TINY_GPT_CONFIG) as f:
        return ModelManager().stats(ModelConfig.from_dict(json.load(f)))


def test_config_file_json_shape(capsys):
    rc, out = _run([TINY_GPT_CONFIG, "--json"], capsys)
    row = json.loads(out)
    assert rc == 0
    assert {"shape", "params", "flops", "kv_cache", "training_plan"} <= set(row)
    assert "trained" not in row
    stats = _tiny_stats()
    assert row["params"]["total"] == stats.num_params
    assert row["params"]["scaling"] == stats.num_scaling_params
    assert row["flops"]["per_token"] == stats.flops_per_token
    assert row["kv_cache"]["bytes_per_token"] == stats.kv_bytes_per_token()


def test_plan_equals_derive_training_plan_directly(capsys):
    _rc, out = _run([TINY_GPT_CONFIG, "--target-flops", "1e12", "--target-param-data-ratio", "8", "--d-ref-scaling-params", "123456", "--json"], capsys)
    plan = json.loads(out)["training_plan"]
    stats = _tiny_stats()
    expected = derive_training_plan(
        num_scaling_params=stats.num_scaling_params, d_ref_scaling_params=123456, num_flops_per_token=stats.flops_per_token,
        target_param_data_ratio=8.0, target_flops=1e12, num_iterations=-1, total_batch_size=-1, weight_decay=0.28)
    for key in ("target_tokens", "total_batch_size", "auto_batch_size", "num_iterations", "horizon_source",
                "total_tokens", "total_flops", "batch_lr_scale", "weight_decay_scaled"):
        assert plan[key] == getattr(expected, key), key
    assert plan["horizon_source"] == "target_flops"


def test_default_plan_is_ratio_12_with_itself_as_d_ref(capsys):
    _rc, out = _run([TINY_GPT_CONFIG, "--json"], capsys)
    plan = json.loads(out)["training_plan"]
    stats = _tiny_stats()
    assert plan["horizon_source"] == "target_param_data_ratio"
    assert plan["target_tokens"] == int(12 * stats.num_scaling_params)
    assert plan["d_ref_source"] == "this model itself"


def test_gpu_hours_are_independent_of_num_gpus(capsys):
    _rc, one = _run([TINY_GPT_CONFIG, "--gpu", "NVIDIA H100", "--json"], capsys)
    _rc, four = _run([TINY_GPT_CONFIG, "--gpu", "NVIDIA H100", "--num-gpus", "4", "--json"], capsys)
    a, b = json.loads(one)["training_plan"], json.loads(four)["training_plan"]
    assert a["gpu_hours"] == pytest.approx(b["gpu_hours"])
    assert b["wall_clock_hours"] == pytest.approx(a["wall_clock_hours"] / 4)


@pytest.fixture
def fake_checkpoint(base_dir):
    """A checkpoint dir with only what `info` reads (and find_last_step needs): an empty model
    file and a meta."""
    tag_dir = os.path.join(base_dir, "checkpoints", "fake")
    os.makedirs(tag_dir)
    open(os.path.join(tag_dir, "model_000010.pt"), "w").close()
    with open(TINY_GPT_CONFIG) as f:
        model_config = json.load(f)
    meta = {"step": 10, "val_bpb": 1.25, "total_batch_size": 2048, "total_training_time": 90.0,
            "tokenizer_fingerprint": "deadbeef", "model_config": model_config}
    with open(os.path.join(tag_dir, "meta_000010.json"), "w") as f:
        json.dump(meta, f)
    return "fake"


def test_checkpoint_tag_reports_what_training_produced(fake_checkpoint, capsys):
    rc, out = _run([fake_checkpoint, "--json"], capsys)
    row = json.loads(out)
    assert rc == 0
    assert row["trained"]["step"] == 10
    assert row["trained"]["tokens_trained"] == 10 * 2048
    assert row["trained"]["val_bpb"] == 1.25
    assert row["trained"]["train_time_sec"] == 90.0
    assert row["trained"]["adapters"] == []
    assert "training_plan" not in row  # a trained model's horizon is already in its meta


def test_checkpoint_tag_with_a_plan_flag_adds_the_plan(fake_checkpoint, capsys):
    _rc, out = _run([fake_checkpoint, "--target-param-data-ratio", "10", "--json"], capsys)
    assert json.loads(out)["training_plan"]["horizon_source"] == "target_param_data_ratio"


def test_fingerprint_mismatch_is_flagged(fake_checkpoint, capsys):
    """The fixture's recorded fingerprint is not the bundled tokenizer's."""
    _rc, out = _run([fake_checkpoint, "--json"], capsys)
    assert json.loads(out)["trained"]["tokenizer_fingerprint_status"] == "MISMATCH"


def test_list_targets(capsys):
    rc, out = _run([TINY_GPT_CONFIG, "--list-targets", "--json"], capsys)
    targets = json.loads(out)
    assert rc == 0 and targets
    assert any(t["target"].endswith("mixer.c_q") for t in targets)
    assert not any(t["has_adapter"] for t in targets)


def test_unknown_source_is_a_clean_error(base_dir, capsys):
    assert info.main(["no-such-tag"]) == 1
    assert "neither a config file nor a checkpoint tag" in capsys.readouterr().err


def test_main_dispatches_info(capsys):
    from tinylab.__main__ import main
    assert main(["info", TINY_GPT_CONFIG, "--json"]) == 0
    assert "params" in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("fixture,expected", [
    ("gpt_tiny.json", ["attention"]),
    ("hybrid_conv_tiny.json", None),  # filled in below from the fixture's own blocks
])
def test_mixer_types_come_from_the_tree_not_the_preset_label(fixture, expected):
    path = os.path.join(os.path.dirname(TINY_GPT_CONFIG), fixture)
    with open(path) as f:
        raw = json.load(f)
    config = ModelConfig.from_dict(raw)
    types = info.mixer_types(config)
    if expected is None:
        assert len(types) > 1 and "attention" in types  # a hybrid reports every mixer it contains
    else:
        assert types == expected
    assert info.stats_block(ModelManager(), config)["mixers"] == types
