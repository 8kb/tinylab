"""Tests for tinylab.job: defaults deep-merge, step ordering, --only, unknown-key rejection,
--dry-run, and that a "chat" block is never dispatched as a step."""
import json
import os

import pytest

from tinylab import job


def _write_job(tmp_path, doc):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(doc))
    return str(path)


def _write_config(tmp_path, name="config.json"):
    """A file "model_config" just needs to exist -- job.resolve_steps only checks existence and
    absolutizes the path; content validation happens later, in modelconfig.load_model_config."""
    path = tmp_path / name
    path.write_text("{}")
    return str(path)


def test_defaults_deep_merge_and_step_own_keys_win(tmp_path):
    config_path = _write_config(tmp_path)
    doc = {
        "defaults": {"sequence_len": 256, "model_config": config_path},
        "steps": [
            {"name": "a", "op": "prepare", "kind": "base"},
            {"name": "b", "op": "prepare", "kind": "base", "sequence_len": 512},
        ],
    }
    j = job.load(_write_job(tmp_path, doc))
    steps = job.resolve_steps(j)
    assert steps[0]["sequence_len"] == 256
    assert steps[0]["model_config"] == config_path
    assert steps[1]["sequence_len"] == 512  # step's own key wins over defaults


def test_step_order_is_preserved(tmp_path):
    doc = {"steps": [
        {"name": "z", "op": "prepare", "kind": "base", "sequence_len": 8},
        {"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8},
    ]}
    j = job.load(_write_job(tmp_path, doc))
    steps = job.resolve_steps(j)
    assert [s["name"] for s in steps] == ["z", "a"]


def test_only_filters_to_one_step(tmp_path):
    doc = {"steps": [
        {"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8},
        {"name": "b", "op": "prepare", "kind": "base", "sequence_len": 8},
    ]}
    j = job.load(_write_job(tmp_path, doc))
    steps = job.resolve_steps(j, only="b")
    assert [s["name"] for s in steps] == ["b"]


def test_only_unknown_name_raises(tmp_path):
    doc = {"steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]}
    j = job.load(_write_job(tmp_path, doc))
    with pytest.raises(job.JobError):
        job.resolve_steps(j, only="nope")


def test_duplicate_step_names_raise(tmp_path):
    doc = {"steps": [
        {"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8},
        {"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8},
    ]}
    j = job.load(_write_job(tmp_path, doc))
    with pytest.raises(job.JobError):
        job.resolve_steps(j)


def test_unknown_op_raises_with_suggestion(tmp_path):
    doc = {"steps": [{"name": "a", "op": "preprae", "sequence_len": 8}]}
    j = job.load(_write_job(tmp_path, doc))
    with pytest.raises(job.JobError, match="prepare"):
        job.resolve_steps(j)


def test_unknown_step_key_raises(tmp_path):
    doc = {"steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8, "shrads": 2}]}
    j = job.load(_write_job(tmp_path, doc))
    with pytest.raises(job.JobError, match="shards"):
        job.resolve_steps(j)


def test_kind_specific_key_on_wrong_kind_raises(tmp_path):
    """"mmlu_epochs" is an SFT-only prepare key -- on a kind="base" step it must be rejected, not
    silently accepted and ignored."""
    doc = {"steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8, "mmlu_epochs": 3}]}
    j = job.load(_write_job(tmp_path, doc))
    with pytest.raises(job.JobError, match="mmlu_epochs"):
        job.resolve_steps(j)


def test_model_config_key_typo_raises_with_suggestion(tmp_path):
    doc = {"steps": [{"name": "a", "op": "train", "kind": "base", "sequence_len": 8,
                       "modle_config": "x.json"}]}
    j = job.load(_write_job(tmp_path, doc))
    with pytest.raises(job.JobError, match="model_config"):
        job.resolve_steps(j)


def test_model_config_path_resolves_relative_to_job_dir(tmp_path):
    (tmp_path / "sub").mkdir()
    config_path = _write_config(tmp_path / "sub")
    doc = {"steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8,
                       "model_config": "sub/config.json"}]}
    job_path = _write_job(tmp_path, doc)
    j = job.load(job_path)
    steps = job.resolve_steps(j, job_dir=str(tmp_path))
    assert steps[0]["model_config"] == config_path


def test_missing_model_config_path_raises(tmp_path):
    doc = {"steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8,
                       "model_config": "does_not_exist.json"}]}
    j = job.load(_write_job(tmp_path, doc))
    with pytest.raises(job.JobError, match="does_not_exist.json"):
        job.resolve_steps(j, job_dir=str(tmp_path))


def test_underscore_keys_are_ignored(tmp_path):
    doc = {
        "_comment": "a freeform note",
        "steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8, "_note": "ok"}],
    }
    j = job.load(_write_job(tmp_path, doc))
    steps = job.resolve_steps(j)  # must not raise
    assert steps[0]["_note"] == "ok"


def test_missing_steps_key_raises(tmp_path):
    doc = {"defaults": {}}
    with pytest.raises(job.JobError):
        job.load(_write_job(tmp_path, doc))


def test_chat_block_is_never_a_step(tmp_path):
    doc = {
        "steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}],
        "chat": {"model_tag": "sft"},
    }
    j = job.load(_write_job(tmp_path, doc))
    steps = job.resolve_steps(j)
    assert all(s["name"] != "chat" for s in steps)
    assert len(steps) == 1


def test_chat_block_cannot_be_an_op_value(tmp_path):
    doc = {"steps": [{"name": "chat", "op": "chat"}]}
    j = job.load(_write_job(tmp_path, doc))
    with pytest.raises(job.JobError):
        job.resolve_steps(j)


def test_resolve_chat_merges_defaults(tmp_path):
    doc = {
        "defaults": {"device": "cpu"},
        "steps": [],
        "chat": {"model_tag": "sft"},
    }
    j = job.load(_write_job(tmp_path, doc))
    chat_cfg = job.resolve_chat(j)
    assert chat_cfg == {"device": "cpu", "model_tag": "sft"}


def test_resolve_chat_empty_when_no_chat_block(tmp_path):
    doc = {"steps": []}
    j = job.load(_write_job(tmp_path, doc))
    assert job.resolve_chat(j) == {}


def test_dry_run_touches_nothing_and_returns_resolved_configs(tmp_path, monkeypatch, capsys):
    base_dir = tmp_path / "base"  # deliberately not created yet
    monkeypatch.setenv("TINYLAB_BASE_DIR", str(base_dir))
    doc = {
        "defaults": {"sequence_len": 8},
        "steps": [{"name": "a", "op": "prepare", "kind": "base", "shards": 1}],
    }
    path = _write_job(tmp_path, doc)
    results = job.run_file(path, dry_run=True)
    assert results == [{"name": "a", "op": "prepare", "kind": "base", "shards": 1, "sequence_len": 8}]
    # A real (non-dry-run) op would create TINYLAB_BASE_DIR the moment anything calls
    # get_base_dir(); --dry-run must return before that ever happens.
    assert not base_dir.exists()


def test_device_propagates_to_every_op_not_just_chat(tmp_path, monkeypatch):
    """Regression test: job.run_file used to build Context() with no device_type at all, so only
    tinylab.chat (which builds its own Context separately) honored a job's "device" default --
    every op silently ignored it. Confirm run_file now threads it through too."""
    doc = {
        "defaults": {"device": "cpu", "sequence_len": 8},
        "steps": [{"name": "a", "op": "prepare", "kind": "base", "shards": 1}],
    }
    path = _write_job(tmp_path, doc)

    seen = {}
    def fake_run(cfg, ctx):
        seen["device_type"] = ctx.device_type
        return {"op": "prepare"}
    monkeypatch.setattr("tinylab.ops.prepare.run", fake_run)

    job.run_file(path)
    assert seen["device_type"] == "cpu"
