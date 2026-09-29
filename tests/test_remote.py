"""
The central bucket (tinylab.remote / remote_stores / readme / remote_cli), against MemoryRemote --
an in-memory Remote, no network. The one real training round-trip lives in test_remote_roundtrip.py.
"""
import json
import os
import threading
import time

import numpy as np
import pytest
from datacore import BestFitCropPacker, DataManager

from tinylab import readme
from tinylab import remote as R
from tinylab import remote_cli
from tinylab.remote_stores import PrefetchingDatasetStore, UploadingDatasetStore

PRODUCER = {"repo": "tinylab", "experiment": "scratch", "job": "j", "step": "pre", "version": "v0", "git": None}


@pytest.fixture(autouse=True)
def fast_uploader(monkeypatch):
    monkeypatch.setattr(R.Uploader, "IDLE_SECONDS", 0.05)
    monkeypatch.setattr(R, "_sleep", lambda s: None)


@pytest.fixture
def remote():
    return R.MemoryRemote("t")


def _write_step(directory, step, *, ranks=(0,), size=8, fill=b"x"):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"model_{step:06d}.pt"), "wb") as f:
        f.write(fill * size)
    with open(os.path.join(directory, f"meta_{step:06d}.json"), "w") as f:
        json.dump({"step": step}, f)
    for r in ranks:
        with open(os.path.join(directory, f"optim_{step:06d}_rank{r}.pt"), "wb") as f:
            f.write(fill * size * 2)


def _push(remote, directory, step, **kw):
    args = dict(push_model="last", push_optim="last", ranks=[0], producer=PRODUCER)
    args.update(kw)
    R.push_checkpoint_step(remote, str(directory), "exp/m1", step, **args)


# ---- policy / layout ---------------------------------------------------------------------------

def test_wants_policies():
    assert R.wants("last", 5) and R.wants("all", 5)
    assert not R.wants("none", 5)
    assert R.wants([5, 7], 5) and not R.wants([5, 7], 6)


@pytest.mark.parametrize("bad", ["some", 3, [1, "a"], [True]])
def test_validate_policy_rejects_junk(bad):
    with pytest.raises(ValueError):
        R.validate_policy(bad, key="push_model")


@pytest.mark.parametrize("rel,ok", [
    ("checkpoints/a/model_000001.pt", True), ("job_state/x.json", False), ("locks/a", False),
    ("prepared/d/train_000000.tmp.npy", False), ("eval_bundle.zip", False), ("base_data_climbmix/shard_0.parquet", False),
    ("task_data/HuggingFaceTB--smoltalk/all/train/x.parquet", False), ("task_data/cais--mmlu/all/test/x.parquet", True),
    ("experiments/e/logs/a.log", True), ("a/b.part", False),
])
def test_is_syncable(rel, ok):
    assert R.is_syncable(rel) is ok


# ---- checkpoint push: marker last, retention, immutability, producer -----------------------------

def test_meta_marker_is_uploaded_last_and_readme_after(remote, tmp_path):
    _write_step(tmp_path, 10)
    _push(remote, tmp_path, 10)
    order = [p for _, p in remote.log]
    meta = order.index("checkpoints/exp/m1/meta_000010.json")
    assert meta > order.index("checkpoints/exp/m1/model_000010.pt") and meta > order.index("checkpoints/exp/m1/optim_000010_rank0.pt")
    assert order[-1] == "checkpoints/exp/m1/README.md"


def test_retention_last_keeps_only_the_newest_step(remote, tmp_path):
    for s in (1, 2, 3):
        _write_step(tmp_path, s)
        _push(remote, tmp_path, s)
    assert R.complete_steps(remote.list("checkpoints/exp/m1"), "checkpoints/exp/m1") == [3]
    assert not [p for p in remote.files if p.endswith(("000001.pt", "000002.pt", "000001.json", "000002.json"))]
    # meta (the marker) is deleted before the model, so an old step never looks complete without weights
    deletes = [p for op, p in remote.log if op == "delete"]
    assert deletes.index("checkpoints/exp/m1/meta_000001.json") < deletes.index("checkpoints/exp/m1/model_000001.pt")


def test_retention_all_keeps_everything(remote, tmp_path):
    for s in (1, 2):
        _write_step(tmp_path, s)
        _push(remote, tmp_path, s, push_model="all", push_optim="all")
    assert R.complete_steps(remote.list("checkpoints/exp/m1"), "checkpoints/exp/m1") == [1, 2]
    assert not [1 for op, _ in remote.log if op == "delete"]


def test_retention_list_pushes_only_named_steps(remote, tmp_path):
    for s in (1, 2, 3):
        _write_step(tmp_path, s)
        _push(remote, tmp_path, s, push_model=[2], push_optim="none")
    assert R.complete_steps(remote.list("checkpoints/exp/m1"), "checkpoints/exp/m1") == [2]
    assert not [p for p in remote.files if "optim" in p]


def test_optim_last_prunes_old_optim_but_model_all_keeps_models(remote, tmp_path):
    for s in (1, 2):
        _write_step(tmp_path, s)
        _push(remote, tmp_path, s, push_model="all", push_optim="last")
    names = sorted(os.path.basename(p) for p in remote.files)
    assert "model_000001.pt" in names and "optim_000001_rank0.pt" not in names and "optim_000002_rank0.pt" in names


def test_complete_step_is_immutable(remote, tmp_path):
    _write_step(tmp_path, 5, size=8)
    _push(remote, tmp_path, 5)
    _write_step(tmp_path, 5, size=9)  # same step, different bytes
    with pytest.raises(R.ImmutableError):
        _push(remote, tmp_path, 5)
    assert len(remote.files["checkpoints/exp/m1/model_000005.pt"]) == 8


def test_identical_repush_is_a_noop_and_partial_step_is_overwritten(remote, tmp_path):
    _write_step(tmp_path, 5)
    _push(remote, tmp_path, 5)
    n = len(remote.log)
    _push(remote, tmp_path, 5)  # identical: skipped
    assert [op for op, p in remote.log[n:] if op == "upload"] == []
    # a partial upload (no marker) may be replaced
    del remote.files["checkpoints/exp/m1/meta_000005.json"]
    _write_step(tmp_path, 5, size=9)
    _push(remote, tmp_path, 5)
    assert len(remote.files["checkpoints/exp/m1/model_000005.pt"]) == 9


def test_producer_conflict_refuses_before_uploading(remote, tmp_path):
    _write_step(tmp_path, 1)
    _push(remote, tmp_path, 1)
    n = len(remote.log)
    _write_step(tmp_path, 2)
    with pytest.raises(R.ProducerConflict):
        _push(remote, tmp_path, 2, producer=dict(PRODUCER, step="other"))
    assert len(remote.log) == n
    _push(remote, tmp_path, 2, producer=dict(PRODUCER, step="other"), force=True)


def test_resume_from_another_machine_is_the_same_producer(remote, tmp_path):
    _write_step(tmp_path, 1)
    _push(remote, tmp_path, 1)
    _write_step(tmp_path, 2)
    _push(remote, tmp_path, 2, producer=dict(PRODUCER, version="v9", git="abc", hardware="8xH100"))  # no error


def test_readme_records_steps_and_history(remote, tmp_path):
    _write_step(tmp_path, 1)
    _push(remote, tmp_path, 1, metrics={"val_bpb": 0.5}, motivation="why")
    _write_step(tmp_path, 2)
    _push(remote, tmp_path, 2, metrics={"val_bpb": 0.4})
    meta, body = readme.split(remote.files["checkpoints/exp/m1/README.md"].decode())
    assert meta["kind"] == "checkpoint" and meta["steps"] == [2] and meta["metrics"] == {"val_bpb": 0.4}
    assert meta["retention"] == {"model": "last", "optim": "last"}
    assert "why" in readme.section(remote.files["checkpoints/exp/m1/README.md"].decode(), "Motivation")
    assert "pushed step 1" in body and "pushed step 2" in body and "pruned [1]" in body


# ---- README ------------------------------------------------------------------------------------

def test_readme_update_never_touches_human_sections():
    text = readme.create("checkpoint", "checkpoints/a/b", producer=PRODUCER, inputs={"tokenizer": "tokenizers/t"}, motivation="m")
    text = text.replace("## Notes\n", "## Notes\n\nhand-written note with `code` and --- dashes\n")
    text = text.replace("m\n\n## History", "a hand-edited motivation\n\n## History")
    updated = readme.update(text, meta_updates={"steps": [1, 2]}, history="pushed step 2")
    for name in ("Motivation", "Related", "Notes"):
        assert readme.section(updated, name) == readme.section(text, name)
    meta, _ = readme.split(updated)
    assert meta["steps"] == [1, 2] and meta["producer"] == PRODUCER
    assert readme.section(updated, "History").count("- ") == 2
    assert "[tokenizers/t](../../../tokenizers/t/README.md)" in readme.section(updated, "Related")


def test_readme_frontmatter_roundtrip_and_motivation_detection():
    text = readme.create("dataset", "prepared/x", producer=PRODUCER, metrics={"val": {"n": 1}})
    meta, _ = readme.split(text)
    assert meta["metrics"] == {"val": {"n": 1}} and meta["id"] == "prepared/x"
    assert readme.missing_motivation(text)
    assert not readme.missing_motivation(readme.create("dataset", "prepared/x", producer=PRODUCER, motivation="because"))


# ---- Uploader ----------------------------------------------------------------------------------

def test_uploader_runs_in_order_and_flush_waits(remote):
    seen = []
    up = R.Uploader(remote)
    for i in range(5):
        up.submit("e", lambda r, i=i: (time.sleep(0.01), seen.append(i)), label=str(i))
    up.flush()
    assert seen == [0, 1, 2, 3, 4]


def test_uploader_failure_withholds_the_marker_and_surfaces_on_flush(remote):
    seen = []
    up = R.Uploader(remote)
    up.submit("e", lambda r: (_ for _ in ()).throw(RuntimeError("shard failed")), label="shard")
    up.submit("e", lambda r: seen.append("marker"), marker=True, label="marker")
    up.submit("other", lambda r: seen.append("other-marker"), marker=True, label="om")
    with pytest.raises(R.UploadError, match="shard failed"):
        up.flush()
    assert seen == ["other-marker"]  # the marker of the failed entity never ran; other entities are unaffected
    up.flush()  # errors are reported once


def test_uploader_thread_exits_when_idle(remote):
    up = R.Uploader(remote)
    up.submit("e", lambda r: None)
    up.flush()
    deadline = time.time() + 2
    while time.time() < deadline and any(t.name == "tinylab-uploader" for t in threading.enumerate()):
        time.sleep(0.02)
    assert not any(t.name == "tinylab-uploader" for t in threading.enumerate())


def test_retry_with_backoff_then_success(remote, tmp_path):
    (tmp_path / "f").write_bytes(b"1")
    remote.fail_next = [ConnectionError("x"), ConnectionError("y")]
    R._retry(lambda: remote.upload([(str(tmp_path / "f"), "f")]), what="t")
    assert remote.files["f"] == b"1"


def test_permission_errors_are_not_retried(remote):
    remote.writable = False
    with pytest.raises(PermissionError):
        R._retry(lambda: remote.delete(["x"]), what="t")


# ---- pull ----------------------------------------------------------------------------------------

def test_pull_checkpoint_gets_only_model_and_meta_of_last_complete_step(remote, tmp_path, base_dir):
    src = tmp_path / "src"
    for s in (1, 2):
        _write_step(src, s)
        _push(remote, src, s, push_model="all", push_optim="all")
    remote.files["checkpoints/exp/m1/model_000003.pt"] = b"half-uploaded"  # no meta: invisible
    step = R.pull_checkpoint(remote, "exp/m1")
    assert step == 2
    local = sorted(os.listdir(os.path.join(base_dir, "checkpoints", "exp", "m1")))
    assert local == ["meta_000002.json", "model_000002.pt"]
    assert R.pull_checkpoint(remote, "exp/m1", step=1, optim=True) == 1
    assert "optim_000001_rank0.pt" in os.listdir(os.path.join(base_dir, "checkpoints", "exp", "m1"))
    with pytest.raises(FileNotFoundError):
        R.pull_checkpoint(remote, "exp/m1", step=3)


def test_pull_never_leaves_partial_files(remote, base_dir):
    remote.files["tokenizers/t/tokenizer.pkl"] = b"abc"
    R.pull_tokenizer(remote, "t")
    assert sorted(os.listdir(os.path.join(base_dir, "tokenizers", "t"))) == ["tokenizer.pkl"]


def test_tokenizer_pull_refuses_a_different_local_copy(remote, base_dir):
    remote.files["tokenizers/t/tokenizer.pkl"] = b"remote"
    os.makedirs(os.path.join(base_dir, "tokenizers", "t"))
    with open(os.path.join(base_dir, "tokenizers", "t", "tokenizer.pkl"), "wb") as f:
        f.write(b"local")
    with pytest.raises(R.RemoteError, match="differs"):
        R.pull_tokenizer(remote, "t")
    with open(os.path.join(base_dir, "tokenizers", "t", "tokenizer.pkl"), "rb") as f:
        assert f.read() == b"local"


def test_ensure_local_leaves_an_existing_checkpoint_alone(remote, tmp_path, base_dir):
    from tinylab import checkpoints
    _write_step(os.path.join(base_dir, "checkpoints", "exp", "m1"), 1)
    src = tmp_path / "src"
    _write_step(src, 9)
    _push(remote, src, 9)
    assert checkpoints.ensure_local(remote, "exp/m1") == 1  # local wins, nothing fetched
    assert not os.path.exists(os.path.join(base_dir, "checkpoints", "exp", "m1", "model_000009.pt"))
    assert checkpoints.ensure_local(remote, "exp/m1", step=9) == 9
    assert os.path.exists(os.path.join(base_dir, "checkpoints", "exp", "m1", "model_000009.pt"))


# ---- datasets: upload as written, manifest last; prefetch in order ---------------------------------

class _Text:
    def __init__(self, n):
        self.n = n

    def text_batches(self):
        yield "fake", ["The quick brown fox jumps over the lazy dog near the stone bridge."] * self.n


def _prepare_into(store, tokenizer):
    DataManager().prepare(store, sources={"train": _Text(300), "val": _Text(30)}, tokenizer=tokenizer, sequence_len=16,
                          sequences_per_volume=32, packer=BestFitCropPacker(buffer_size=16))


def test_dataset_upload_puts_manifest_last_and_prefetch_reads_it_back_in_order(remote, tmp_path, base_dir):
    from tinylab.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(base_dir)
    up = R.Uploader(remote)
    store = UploadingDatasetStore(str(tmp_path / "ds"), name="d1", uploader=up, producer=PRODUCER)
    _prepare_into(store, tokenizer)
    up.flush()
    uploads = [p for op, p in remote.log if op == "upload"]
    assert uploads[-1] == "prepared/d1/manifest.json" and len([p for p in uploads if p.endswith(".npy")]) >= 3
    assert "prepared/d1/README.md" in remote.files
    manifest = json.loads(remote.files["prepared/d1/manifest.json"])

    # a fresh disk: manifest + val + token_bytes up front, train shards on demand, a bounded window ahead
    fetched = []
    real = remote._fetch
    remote._fetch = lambda pairs: (fetched.extend(os.path.basename(r) for r, _ in pairs), real(pairs))[1]
    local = str(tmp_path / "fresh" / "d1")
    pstore = PrefetchingDatasetStore(local, name="d1", remote=remote, lookahead=1)
    m = pstore.read_manifest()
    assert m["splits"].keys() == manifest["splits"].keys()
    train_files = [v["file"] for v in m["splits"]["train"]["volumes"]]
    assert not any(f in fetched for f in train_files)  # nothing of train fetched yet
    first = pstore.open_volume(train_files[0])
    assert isinstance(first, np.ndarray)
    deadline = time.time() + 5
    while train_files[1] not in fetched and time.time() < deadline:
        time.sleep(0.01)
    assert train_files[0] in fetched and train_files[1] in fetched  # the lookahead window moved to the next shard
    assert train_files[2] not in fetched  # ...and only that far
    pstore.close()


def test_prefetch_store_without_a_manifest_in_the_bucket_reads_as_not_prepared(remote, tmp_path):
    remote.files["prepared/d/train_000000.npy"] = b"shard"  # shard but no manifest: incomplete
    assert PrefetchingDatasetStore(str(tmp_path / "d"), name="d", remote=remote).read_manifest() is None


def test_prefetcher_blocks_until_downloaded_and_retries_after_failure(remote, tmp_path):
    for n in "abc":
        remote.files[f"d/{n}"] = n.encode()
    gate = threading.Event()
    real = remote._fetch

    def slow(pairs):
        gate.wait(2)
        real(pairs)
    remote._fetch = slow
    pf = R.Prefetcher(remote, "d", str(tmp_path), ["a", "b", "c"], lookahead=1)
    threading.Timer(0.2, gate.set).start()
    t0 = time.time()
    pf.ensure("a")
    assert time.time() - t0 >= 0.15 and (tmp_path / "a").exists()
    pf.close()
    remote.files.pop("d/c")
    remote._fetch = real
    pf = R.Prefetcher(remote, "d", str(tmp_path / "x"), ["a", "b", "c"], lookahead=0)
    with pytest.raises(FileNotFoundError):
        pf.ensure("c")
    remote.files["d/c"] = b"c"
    pf.ensure("c")
    pf.close()


# ---- experiment sync -----------------------------------------------------------------------------

def test_experiment_sync_uploads_changes_only(remote, base_dir):
    log = os.path.join(base_dir, "experiments", "e1", "logs", "j.log")
    os.makedirs(os.path.dirname(log))
    with open(log, "w") as f:
        f.write("a\n")
    sync = R.ExperimentSync(remote, "e1", interval=999)
    assert sync.sync() == 1 and sync.sync() == 0
    with open(log, "a") as f:
        f.write("b\n")
    assert sync.sync() == 1 and remote.files["experiments/e1/logs/j.log"] == b"a\nb\n"
    sync.close()


# ---- CLI -----------------------------------------------------------------------------------------

def _cli(capsys, *argv, url="memory://cli"):
    code = remote_cli.main(["--remote", url, *argv])
    return code, capsys.readouterr().out


def test_cli_ls_check_pull_push_rm(capsys, tmp_path, base_dir, monkeypatch):
    remote = R.MemoryRemote.named("cli")
    remote.files.clear()
    ck = os.path.join(base_dir, "checkpoints", "exp01", "m1", "base")
    _write_step(ck, 4)
    code, out = _cli(capsys, "push", "checkpoints/exp01/m1/base", "--optim")
    assert code == 0 and "pushed" in out
    remote.files["checkpoints/exp01/m1/base/model_000009.pt"] = b"orphan"
    remote.files["prepared/nomanifest/train_000000.npy"] = b"x"

    code, out = _cli(capsys, "ls")
    assert "checkpoint" in out and "steps=[4]" in out and "INCOMPLETE=[9]" in out and "INCOMPLETE (no manifest)" in out

    code, out = _cli(capsys, "check")
    assert "can write" in out and "incomplete step: checkpoints/exp01/m1/base@9" in out
    assert "README needs a Motivation:" in out and "checkpoints/exp01/m1/base" in out
    assert "incomplete dataset (no manifest): prepared/nomanifest" in out

    os.remove(os.path.join(ck, "model_000004.pt"))
    code, out = _cli(capsys, "pull", "checkpoints/exp01/m1/base")
    assert code == 0 and os.path.exists(os.path.join(ck, "model_000004.pt"))

    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    code, out = _cli(capsys, "rm", "checkpoints/exp01/m1/base@4")
    assert code == 1 and "checkpoints/exp01/m1/base/model_000004.pt" in remote.files
    code, out = _cli(capsys, "rm", "checkpoints/exp01/m1/base@4", "--yes")
    assert code == 0 and "checkpoints/exp01/m1/base/model_000004.pt" not in remote.files
    meta, body = readme.split(remote.files["checkpoints/exp01/m1/base/README.md"].decode())
    assert meta["steps"] == [] and "removed step 4" in body


def test_cli_check_flags_a_foreign_tag_prefix(capsys):
    remote = R.MemoryRemote.named("cli2")
    remote.files.clear()
    remote.files["experiments/01-x/README.md"] = readme.create("experiment", "experiments/01-x", producer=PRODUCER, extra={"tag_prefix": "exp01"}).encode()
    remote.files["experiments/01-x/logs/a.log"] = b"x"
    for tag in ("exp01/m1", "zzz/m1", "scratch/t"):
        remote.files[f"checkpoints/{tag}/model_000001.pt"] = b"m"
        remote.files[f"checkpoints/{tag}/meta_000001.json"] = b"{}"
    code, out = _cli(capsys, "check", url="memory://cli2")
    assert "foreign tag prefix: checkpoints/zzz/m1" in out
    assert "foreign tag prefix: checkpoints/exp01/m1" not in out and "checkpoints/scratch/t" not in out.split("foreign")[-1]


def test_cli_push_docs_uploads_the_type_readmes(capsys):
    remote = R.MemoryRemote.named("cli3")
    remote.files.clear()
    code, out = _cli(capsys, "push-docs", url="memory://cli3")
    assert code == 0
    for path in ("README.md", "tokenizers/README.md", "prepared/README.md", "checkpoints/README.md", "experiments/README.md",
                 "task_data/README.md", "eval_bundle/README.md"):
        assert path in remote.files


def test_cli_push_refuses_a_conflicting_existing_entity(capsys, tmp_path, base_dir):
    remote = R.MemoryRemote.named("cli4")
    remote.files.clear()
    src = os.path.join(base_dir, "checkpoints", "a")
    _write_step(src, 1)
    _cli(capsys, "push", "checkpoints/a", url="memory://cli4")
    _write_step(src, 1, size=99)
    code = remote_cli.main(["--remote", "memory://cli4", "push", "checkpoints/a"])
    assert code == 1
    assert remote_cli.main(["--remote", "memory://cli4", "push", "checkpoints/a", "--force"]) == 0


def test_not_found_is_not_retried(monkeypatch):
    class EntryNotFoundError(Exception):
        pass
    calls = []

    def fn():
        calls.append(1)
        raise EntryNotFoundError("gone")
    with pytest.raises(EntryNotFoundError):
        R._retry(fn, what="t")
    assert calls == [1]
