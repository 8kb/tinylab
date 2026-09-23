"""Tests for tinylab.job: defaults deep-merge, step ordering, --only, unknown-key rejection,
--dry-run, and that a "chat" block is never dispatched as a step."""
import json
import os
import subprocess
import sys

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


def test_resume_skips_completed_steps_after_a_crash(tmp_path, base_dir, monkeypatch):
    """A 3-step job where the second step raises: run once (crashes, the job state file records
    step 'a' as done); a second attempt without resume=True must refuse to start; a third attempt
    with resume=True must skip 'a' (never call its run again), retry 'b', then run 'c', and clean
    up the state file on success.

    _pid_alive is stubbed to False throughout -- this test simulates a crash by having a step
    raise and then calling job.run_file again from this *same* still-running test process, so the
    state file's recorded pid (this process's own) would otherwise look genuinely alive. That
    liveness check is exercised for real, separately, in test_resume_refuses_when_recorded_pid_is_
    still_alive/test_resume_proceeds_when_recorded_pid_is_dead below."""
    monkeypatch.setattr(job, "_pid_alive", lambda pid: False)
    doc = {"defaults": {"log_dir": "logs"}, "steps": [
        {"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8},
        {"name": "b", "op": "prepare", "kind": "base", "sequence_len": 8},
        {"name": "c", "op": "prepare", "kind": "base", "sequence_len": 8},
    ]}
    path = _write_job(tmp_path, doc)

    calls = []

    def failing_run(cfg, ctx):
        calls.append(cfg["name"])
        if cfg["name"] == "b":
            raise RuntimeError("simulated crash")
        return {"op": "prepare"}

    monkeypatch.setattr("tinylab.ops.prepare.run", failing_run)

    with pytest.raises(RuntimeError, match="simulated crash"):
        job.run_file(path)
    assert calls == ["a", "b"]

    # A ".old" generation is expected here (two writes have happened: the initial empty one, then
    # "a" done) -- normal 2-generation rotation, not stray state.
    state_dir = os.path.join(base_dir, "job_state")
    state_path = os.path.join(state_dir, [f for f in os.listdir(state_dir) if f.endswith(".json")][0])
    assert job._read_state(state_path)["steps"] == {"a": {"status": "done"}}

    calls.clear()
    with pytest.raises(job.JobError, match="already exists"):
        job.run_file(path)
    assert calls == []  # nothing ran -- refused before touching any step

    def succeeding_run(cfg, ctx):
        calls.append(cfg["name"])
        return {"op": "prepare"}

    monkeypatch.setattr("tinylab.ops.prepare.run", succeeding_run)
    results = job.run_file(path, resume=True)
    assert calls == ["b", "c"]  # "a" skipped -- already completed
    assert len(results) == 2  # only steps actually run this invocation
    assert not os.listdir(state_dir)  # state file (and any .old/.tmp) removed on clean completion


def test_resume_with_no_state_file_is_a_normal_fresh_run(tmp_path, base_dir, monkeypatch):
    doc = {"defaults": {"log_dir": "logs"}, "steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]}
    path = _write_job(tmp_path, doc)
    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: {"op": "prepare"})
    results = job.run_file(path, resume=True)
    assert len(results) == 1
    assert not os.listdir(os.path.join(base_dir, "job_state"))


def test_only_bypasses_the_state_file_mechanism_entirely(tmp_path, base_dir, monkeypatch):
    """--only is a deliberate single-step override -- it must never touch the job state file, so
    it keeps working even while a previous full-pipeline run's state file still exists."""
    doc = {"defaults": {"log_dir": "logs"}, "steps": [
        {"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8},
        {"name": "b", "op": "prepare", "kind": "base", "sequence_len": 8},
    ]}
    path = _write_job(tmp_path, doc)
    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: {"op": "prepare"})

    state_dir = os.path.join(base_dir, "job_state")
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "stale.json"), "w") as f:
        f.write("{}")

    results = job.run_file(path, only="b")
    assert len(results) == 1
    assert os.listdir(state_dir) == ["stale.json"]  # untouched, not removed


def test_write_state_and_read_state_round_trip(tmp_path):
    path = str(tmp_path / "job.json")
    written = job._write_state(path, {"steps": {"a": {"status": "done"}}})
    assert written["counter"] == 1
    assert job._read_state(path) == written

    written2 = job._write_state(path, dict(written, steps={"a": {"status": "done"}, "b": {"status": "done"}}))
    assert written2["counter"] == 2  # monotonic across writes
    assert job._read_state(path) == written2
    assert os.path.exists(path + ".old")  # previous generation kept


def test_read_state_falls_back_to_old_generation_when_current_is_corrupt(tmp_path, capsys):
    path = str(tmp_path / "job.json")
    job._write_state(path, {"steps": {"a": {"status": "done"}}})
    job._write_state(path, {"steps": {"a": {"status": "done"}, "b": {"status": "done"}}})
    # Simulate a crash mid-write: the current generation is truncated/corrupt, but .old (written
    # by the first call, fully durable) must still be there and still be trusted.
    with open(path, "w") as f:
        f.write("{not valid json")
    recovered = job._read_state(path)
    assert recovered is not None
    assert recovered["steps"] == {"a": {"status": "done"}}  # the .old generation, not the corrupt one
    assert "previous generation" in capsys.readouterr().out  # warned, not silent


def test_read_state_falls_back_to_old_generation_when_current_is_simply_absent(tmp_path, capsys):
    """The other way a crash can leave only ".old" around: it died between demoting the old
    current file to ".old" and promoting the temp file to current, so "current" doesn't exist at
    all (not even truncated) -- must be treated identically to "current is corrupt", not as
    "nothing here"."""
    path = str(tmp_path / "job.json")
    job._write_state(path, {"steps": {"a": {"status": "done"}}})
    job._write_state(path, {"steps": {"a": {"status": "done"}, "b": {"status": "done"}}})
    os.remove(path)
    assert job._state_exists(path)  # still detected as "a previous attempt happened"
    recovered = job._read_state(path)
    assert recovered is not None
    assert recovered["steps"] == {"a": {"status": "done"}}
    assert "previous generation" in capsys.readouterr().out


def test_read_state_returns_none_when_both_generations_are_unusable(tmp_path, capsys):
    path = str(tmp_path / "job.json")
    job._write_state(path, {"steps": {"a": {"status": "done"}}})
    with open(path, "w") as f:
        f.write("{not valid json")
    with open(path + ".old", "w") as f:
        f.write("{also not valid")
    assert job._read_state(path) is None
    assert "fresh start" in capsys.readouterr().out  # warned, not silent


def test_pid_alive_true_for_the_current_process_false_for_a_dead_one():
    assert job._pid_alive(os.getpid()) is True
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert job._pid_alive(proc.pid) is False
    assert job._pid_alive(None) is False
    assert job._pid_alive("not-a-pid") is False


def test_resume_refuses_when_recorded_pid_is_still_alive(tmp_path, base_dir):
    """A still-alive recorded pid is refused even with resume=True -- it means a genuinely
    concurrent run, not a crashed one, and racing two processes on the same checkpoint files
    would be real corruption."""
    doc = {"defaults": {"log_dir": "logs"}, "steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]}
    path = _write_job(tmp_path, doc)
    state_path = job._state_path(path)
    job._write_state(state_path, {"job_path": path, "pid": os.getpid(), "steps": {}})

    with pytest.raises(job.JobError, match="still appears to be running"):
        job.run_file(path, resume=True)
    with pytest.raises(job.JobError, match="still appears to be running"):
        job.run_file(path)  # same refusal without --resume too


def test_resume_proceeds_when_recorded_pid_is_dead(tmp_path, base_dir, monkeypatch):
    doc = {"defaults": {"log_dir": "logs"}, "steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]}
    path = _write_job(tmp_path, doc)
    state_path = job._state_path(path)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    job._write_state(state_path, {"job_path": path, "pid": proc.pid, "steps": {"a": {"status": "done"}}})

    monkeypatch.setattr("tinylab.ops.prepare.run", lambda cfg, ctx: {"op": "prepare"})
    results = job.run_file(path, resume=True)
    assert results == []  # "a" was already recorded done -- correctly skipped, not re-run


def test_train_step_records_progress_and_resume_passes_it_as_a_hint(tmp_path, base_dir, monkeypatch):
    """A fake "train"-shaped op that calls ctx.record_checkpoint mid-run, then crashes: the job
    state file must record it as "in_progress" with the reported checkpoint_step, and a resumed
    run must see that exact value via ctx.resume_checkpoint_step -- the same channel
    tinylab.ops.train's own resume detection reads from. _pid_alive stubbed False for the same
    reason as test_resume_skips_completed_steps_after_a_crash above."""
    monkeypatch.setattr(job, "_pid_alive", lambda pid: False)
    doc = {"defaults": {"log_dir": "logs"},
           "steps": [{"name": "pre", "op": "train", "kind": "base", "sequence_len": 8,
                       "model_config": "cfg.json", "total_batch_size": 1, "world_size": 1,
                       "eval_tokens": 1, "num_iterations": 1}]}
    # model_config must exist on disk -- resolve_steps checks it, even though this fake op never
    # reads it.
    (tmp_path / "cfg.json").write_text("{}")
    path = _write_job(tmp_path, doc)

    seen_hints = []

    def fake_train_run(cfg, ctx):
        seen_hints.append(ctx.resume_checkpoint_step(cfg["name"]))
        ctx.record_checkpoint(cfg["name"], 3)
        raise RuntimeError("simulated crash after one checkpoint")

    monkeypatch.setattr("tinylab.ops.train.run", fake_train_run)
    with pytest.raises(RuntimeError, match="simulated crash"):
        job.run_file(path)
    assert seen_hints == [None]  # nothing to resume from yet, first attempt

    state_dir = os.path.join(base_dir, "job_state")
    state_path = os.path.join(state_dir, [f for f in os.listdir(state_dir) if f.endswith(".json")][0])
    assert job._read_state(state_path)["steps"] == {"pre": {"status": "in_progress", "checkpoint_step": 3}}

    with pytest.raises(RuntimeError, match="simulated crash"):
        job.run_file(path, resume=True)
    assert seen_hints == [None, 3]  # second attempt sees the recorded checkpoint step


def test_device_propagates_to_every_op_not_just_chat(tmp_path, monkeypatch):
    """Regression test: job.run_file used to build Context() with no device_type at all, so only
    tinylab.chat (which builds its own Context separately) honored a job's "device" default --
    every op silently ignored it. Confirm run_file now threads it through too."""
    doc = {
        "defaults": {"device": "cpu", "sequence_len": 8, "log_dir": "logs"},
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


# -- checkpoint tags (one namespace; the tag is the whole address) ------------------------------

def _train_job(**step_overrides):
    step = {"name": "a", "op": "train", "kind": "base", "sequence_len": 8}
    step.update(step_overrides)
    return {"steps": [step]}


@pytest.mark.parametrize("key", ["output_tag", "source_tag", "model_tag"])
@pytest.mark.parametrize("bad", ["", "/abs", "a//b", "a/", "/a", "../x", "a/../b", "./a", "a\\b",
                                  "a.b", "a b", "a" * 17, "a/b/c/d/e"])
def test_a_malformed_checkpoint_tag_is_a_job_error_before_anything_runs(tmp_path, key, bad):
    op_step = {"op": "bench", "suite": "core"} if key == "model_tag" else {"op": "train", "kind": "sft"}
    doc = {"steps": [dict({"name": "a", "sequence_len": 8}, **op_step, **{key: bad})]}
    with pytest.raises(job.JobError, match=key):
        job.resolve_steps(job.load(_write_job(tmp_path, doc)), job_dir=str(tmp_path))


@pytest.mark.parametrize("good", ["gpt-d12-base", "kvcache/d13-chat", "exp/2026-09/run3", "d12", "a_b"])
def test_any_reasonable_checkpoint_tag_is_accepted(tmp_path, good):
    doc = _train_job(output_tag=good)
    assert job.resolve_steps(job.load(_write_job(tmp_path, doc)), job_dir=str(tmp_path))[0]["output_tag"] == good


def test_a_bad_step_name_is_rejected_up_front_even_before_it_would_double_as_a_tag(tmp_path):
    """A step's "name" is validated as a single name (see checkpoints.validate_name)
    unconditionally, in resolve_steps's own loop -- before a train step's own "no output_tag, name
    is the tag" fallback (_check_tag_shaped_keys) would otherwise need to check it separately. A
    slash-containing name is invalid either way, but now it's caught as a bad step name, not a bad
    checkpoint tag."""
    doc = _train_job(name="a/b")
    with pytest.raises(job.JobError, match="step name"):
        job.resolve_steps(job.load(_write_job(tmp_path, doc)), job_dir=str(tmp_path))
    # ...an explicit output_tag doesn't rescue an invalid name either -- "name" is checked on its
    # own terms regardless of whether it would have doubled as the tag
    with pytest.raises(job.JobError, match="step name"):
        job.resolve_steps(job.load(_write_job(tmp_path, _train_job(name="a/b", output_tag="ok"))), job_dir=str(tmp_path))


def test_the_source_key_is_gone(tmp_path):
    """The base/sft namespace split is replaced by tags: a job still saying "source" is told so."""
    for step in ({"name": "a", "op": "bench", "suite": "core", "model_tag": "t", "source": "sft"},
                 {"name": "a", "op": "train", "kind": "sft", "sequence_len": 8, "source_tag": "t", "source": "base"}):
        with pytest.raises(job.JobError, match="unknown key 'source'"):
            job.resolve_steps(job.load(_write_job(tmp_path, {"steps": [step]})), job_dir=str(tmp_path))


def test_chat_block_rejects_source_and_bad_tags(tmp_path):
    from tinylab import chat
    j = job.load(_write_job(tmp_path, {"steps": [], "chat": {"model_tag": "t", "source": "sft"}}))
    cfg = job.resolve_chat(j, job_dir=str(tmp_path))
    with pytest.raises(job.JobError, match="unknown key 'source'"):
        job.check_known_keys(cfg, chat.ACCEPTED_KEYS | job.COMMON_KEYS, where='"chat"')
    with pytest.raises(job.JobError, match="model_tag"):
        job.resolve_chat(job.load(_write_job(tmp_path, {"steps": [], "chat": {"model_tag": "../x"}})), job_dir=str(tmp_path))


# -- tokenizer selection ------------------------------------------------------------------------

def test_a_bare_tokenizer_name_is_left_alone_and_a_path_is_made_absolute_against_the_job_dir(tmp_path):
    for value, expected in (("bpe32k", "bpe32k"),
                            ("./toks/mine", os.path.join(str(tmp_path), "toks", "mine")),
                            ("../shared/toks", os.path.normpath(os.path.join(str(tmp_path), "..", "shared", "toks"))),
                            ("/opt/toks", "/opt/toks")):
        doc = {"defaults": {"tokenizer": value}, "steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]}
        steps = job.resolve_steps(job.load(_write_job(tmp_path, doc)), job_dir=str(tmp_path))
        assert steps[0]["tokenizer"] == expected, value


def test_a_tokenizer_paths_need_not_exist_yet(tmp_path):
    """Unlike model_config: a "tokenizer" step may be about to create it."""
    doc = {"defaults": {"tokenizer": "./not/yet"}, "steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]}
    job.resolve_steps(job.load(_write_job(tmp_path, doc)), job_dir=str(tmp_path))


@pytest.mark.parametrize("bad", ["", "  ", 3, None])
def test_a_non_string_tokenizer_is_a_job_error(tmp_path, bad):
    doc = {"defaults": {"tokenizer": bad}, "steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8}]}
    with pytest.raises(job.JobError, match="tokenizer"):
        job.resolve_steps(job.load(_write_job(tmp_path, doc)), job_dir=str(tmp_path))


def test_tokenizer_output_key_belongs_to_the_tokenizer_op_only(tmp_path):
    ok = {"steps": [{"name": "t", "op": "tokenizer", "output": "./toks/new"}]}
    assert job.resolve_steps(job.load(_write_job(tmp_path, ok)), job_dir=str(tmp_path))[0]["output"] == os.path.join(str(tmp_path), "toks", "new")
    bad = {"steps": [{"name": "a", "op": "prepare", "kind": "base", "sequence_len": 8, "output": "x"}]}
    with pytest.raises(job.JobError, match="unknown key 'output'"):
        job.resolve_steps(job.load(_write_job(tmp_path, bad)), job_dir=str(tmp_path))


def test_chat_blocks_tokenizer_path_resolves_against_the_job_dir(tmp_path):
    doc = {"defaults": {"tokenizer": "./toks/mine"}, "steps": [], "chat": {"model_tag": "sft"}}
    cfg = job.resolve_chat(job.load(_write_job(tmp_path, doc)), job_dir=str(tmp_path))
    assert cfg["tokenizer"] == os.path.join(str(tmp_path), "toks", "mine")


def test_the_run_wide_tokenizer_reaches_every_ops_context(tmp_path, base_dir, monkeypatch):
    """Like "device": read off the first resolved step, so it lives in defaults."""
    doc = {"defaults": {"device": "cpu", "sequence_len": 8, "tokenizer": "bpe32k", "log_dir": "logs"},
           "steps": [{"name": "a", "op": "prepare", "kind": "base", "shards": 1}]}
    seen = {}
    def fake_run(cfg, ctx):
        seen["spec"], seen["name"] = ctx.tokenizer_spec, ctx.tokenizer_name
        return {"op": "prepare"}
    monkeypatch.setattr("tinylab.ops.prepare.run", fake_run)
    job.run_file(_write_job(tmp_path, doc))
    assert seen == {"spec": "bpe32k", "name": "bpe32k"}
    doc["defaults"].pop("tokenizer")
    job.run_file(_write_job(tmp_path, doc))
    assert seen == {"spec": None, "name": "default"}
