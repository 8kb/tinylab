"""job.run_file's log files: one general log per job run, one per executed step, both under
<base_dir>/<log_dir>/ -- see job.py's own docstring and docs/architecture.md's On-disk layout."""
import json
import os

import pytest

from tinylab import job
from tinylab.runtime import print0


def _write_job(tmp_path, doc, name="job.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return str(path)


def test_log_dir_is_required_before_any_step_runs(tmp_path, base_dir, monkeypatch):
    calls = []
    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: calls.append(cfg["name"]) or {"op": "prepare"})
    doc = {"steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]}
    with pytest.raises(job.JobError, match="log_dir"):
        job.run_file(_write_job(tmp_path, doc))
    assert calls == []  # refused before touching any step


def test_no_steps_means_no_log_dir_required(tmp_path, base_dir):
    doc = {"steps": []}
    assert job.run_file(_write_job(tmp_path, doc)) == []  # no error, nothing to log either


def test_log_dir_is_validated_the_same_way_a_checkpoint_tag_is(tmp_path, base_dir, monkeypatch):
    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: {"op": "prepare"})
    doc = {"defaults": {"log_dir": "bad dir"},  # a space is not a valid name segment
           "steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]}
    with pytest.raises(job.JobError, match="log_dir"):
        job.run_file(_write_job(tmp_path, doc))


def test_log_files_are_named_and_placed_as_documented(tmp_path, base_dir, monkeypatch):
    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: {"op": "prepare"})
    doc = {"defaults": {"log_dir": "mylogs"}, "steps": [
        {"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8},
        {"name": "b", "op": "prepare", "kind": "base", "sequence_len": 8},
    ]}
    job_path = _write_job(tmp_path, doc, name="myjob.json")
    job.run_file(job_path)

    log_dir = os.path.join(base_dir, "mylogs")
    assert os.path.exists(os.path.join(log_dir, "myjob.log"))
    assert os.path.exists(os.path.join(log_dir, "myjob-a.log"))
    assert os.path.exists(os.path.join(log_dir, "myjob-b.log"))


def test_per_step_log_holds_only_that_steps_own_output_prefix_stripped_and_timestamped(tmp_path, base_dir, monkeypatch):
    def fake_run(cfg, ctx):
        print0(f"[{cfg['name']}] hello from {cfg['name']}")
        return {"op": "prepare"}
    monkeypatch.setattr("tinylab.ops.prepare.run", fake_run)

    doc = {"defaults": {"log_dir": "mylogs"}, "steps": [
        {"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8},
        {"name": "b", "op": "prepare", "kind": "base", "sequence_len": 8},
    ]}
    job_path = _write_job(tmp_path, doc, name="myjob.json")
    job.run_file(job_path)

    log_dir = os.path.join(base_dir, "mylogs")
    a_text = open(os.path.join(log_dir, "myjob-a.log")).read()
    b_text = open(os.path.join(log_dir, "myjob-b.log")).read()
    general_text = open(os.path.join(log_dir, "myjob.log")).read()

    # The step's own message: prefix stripped (filename already says "a"), timestamped.
    assert "hello from a" in a_text
    assert "[a] hello" not in a_text
    assert a_text.count("\n") >= 1
    assert a_text.lstrip().startswith("[")  # a "[YYYY-MM-DD HH:MM:SS] " timestamp prefix
    # Never leaks into the other step's file or the general log.
    assert "hello from a" not in b_text
    assert "hello from a" not in general_text
    assert "hello from b" in b_text


def test_general_log_holds_run_lifecycle_not_step_internals(tmp_path, base_dir, monkeypatch):
    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: {"op": "prepare", "kind": "base"})
    doc = {"defaults": {"log_dir": "mylogs"}, "steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]}
    job_path = _write_job(tmp_path, doc, name="myjob.json")
    job.run_file(job_path)

    general_text = open(os.path.join(base_dir, "mylogs", "myjob.log")).read()
    assert "job myjob started" in general_text
    assert "=== [a] op=prepare ===" in general_text
    assert '"op": "prepare"' in general_text  # the step's result line
    assert "job state (final):" in general_text
    assert "=== job finished ===" in general_text


def test_a_failing_step_logs_to_the_general_log_and_keeps_the_job_state_file(tmp_path, base_dir, monkeypatch):
    monkeypatch.setattr(job, "_pid_alive", lambda pid: False)
    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    doc = {"defaults": {"log_dir": "mylogs"}, "steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]}
    job_path = _write_job(tmp_path, doc, name="myjob.json")

    with pytest.raises(RuntimeError, match="boom"):
        job.run_file(job_path)

    general_text = open(os.path.join(base_dir, "mylogs", "myjob.log")).read()
    assert "job failed" in general_text
    assert "boom" in general_text
    assert "job finished" not in general_text  # never reached
    assert os.listdir(os.path.join(base_dir, "job_state"))  # left in place, as intended on failure
