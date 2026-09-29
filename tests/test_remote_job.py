"""job.run_file with a "remote": fail-fast write check, experiment folder layout and sync, results."""
import json
import os

import pytest

from tinylab import job
from tinylab import remote as R


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(R.Uploader, "IDLE_SECONDS", 0.05)


def _job(tmp_path, url, *, steps=None, extra_defaults=None, name="myjob.json"):
    (tmp_path / "cfg.json").write_text("{}")
    doc = {"defaults": {"experiment": "01-demo", "remote": url, "model_config": "cfg.json", **(extra_defaults or {})},
           "steps": steps or [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8, "push": False}]}
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return str(path)


def test_run_syncs_logs_snapshots_and_results_to_the_experiment_folder(tmp_path, base_dir, monkeypatch):
    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: {"op": "prepare", "n": 1})
    remote = R.MemoryRemote.named("jobrun")
    remote.files.clear()
    job.run_file(_job(tmp_path, "memory://jobrun"))
    files = set(remote.files)
    assert {"experiments/01-demo/logs/myjob.log", "experiments/01-demo/logs/myjob-a.log", "experiments/01-demo/jobs/myjob.json",
            "experiments/01-demo/configs/cfg.json", "experiments/01-demo/results/myjob.results.jsonl"} <= files
    record = json.loads(remote.files["experiments/01-demo/results/myjob.results.jsonl"].splitlines()[0])
    assert record["step"] == "a" and record["n"] == 1
    assert "job finished" in remote.files["experiments/01-demo/logs/myjob.log"].decode()  # the last sync happens after the log closes


def test_no_remote_means_no_extra_files(tmp_path, base_dir, monkeypatch):
    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: {"op": "prepare"})
    doc = {"defaults": {"experiment": "e"}, "steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]}
    path = tmp_path / "j.json"
    path.write_text(json.dumps(doc))
    job.run_file(str(path))
    assert sorted(os.listdir(os.path.join(base_dir, "experiments", "e"))) == ["logs"]


def test_a_pushing_job_fails_fast_when_the_remote_is_not_writable(tmp_path, base_dir, monkeypatch):
    calls = []
    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: calls.append(1) or {"op": "prepare"})
    R.MemoryRemote.named("ro").writable = False
    steps = [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]  # push defaults to true
    with pytest.raises(job.JobError, match="cannot write"):
        job.run_file(_job(tmp_path, "memory://ro", steps=steps))
    assert calls == []  # refused before any step (i.e. before any GPU time)


def test_a_pull_only_job_runs_read_only_and_pushes_nothing(tmp_path, base_dir, monkeypatch):
    remote = R.MemoryRemote.named("ro2")
    remote.writable = False
    seen = {}
    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: seen.update(remote=ctx.remote, uploader=ctx.uploader) or {"op": "prepare"})
    job.run_file(_job(tmp_path, "memory://ro2"))  # push: false
    assert seen["remote"] is remote and seen["uploader"] is None


def test_upload_failures_surface_after_the_run_but_do_not_hide_a_step_error(tmp_path, base_dir, monkeypatch):
    remote = R.MemoryRemote.named("flaky")
    remote.files.clear()

    def prepare(cfg, ctx):
        ctx.uploader.submit("e", lambda r: (_ for _ in ()).throw(RuntimeError("net down")), label="e")
        return {"op": "prepare"}

    monkeypatch.setattr("tinylab.ops.prepare.run", prepare)
    steps = [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]
    with pytest.raises(R.UploadError, match="net down"):
        job.run_file(_job(tmp_path, "memory://flaky", steps=steps))

    def boom(cfg, ctx):
        ctx.uploader.submit("e", lambda r: (_ for _ in ()).throw(RuntimeError("net down")), label="e")
        raise ValueError("step exploded")

    monkeypatch.setattr("tinylab.ops.prepare.run", boom)
    monkeypatch.setattr(job, "_pid_alive", lambda pid: False)
    with pytest.raises(ValueError, match="step exploded"):
        job.run_file(_job(tmp_path, "memory://flaky", steps=steps, name="j2.json"))


def test_changed_job_file_gets_a_timestamped_snapshot_not_an_overwrite(tmp_path, base_dir, monkeypatch):
    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: {"op": "prepare"})
    remote = R.MemoryRemote.named("snap")
    remote.files.clear()
    path = _job(tmp_path, "memory://snap")
    job.run_file(path)
    doc = json.loads(open(path).read())
    doc["defaults"]["sequence_len"] = 8
    open(path, "w").write(json.dumps(doc))
    job.run_file(path)
    snaps = sorted(p for p in remote.files if p.startswith("experiments/01-demo/jobs/"))
    assert len(snaps) == 2 and "experiments/01-demo/jobs/myjob.json" in snaps
