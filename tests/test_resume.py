"""
Step-level resume: ctx.resume=True on a `train` step continues an interrupted run from its own
last checkpoint (model, optimizer, dataloader position) instead of starting over. Real training,
same synthetic in-memory corpus pattern as test_smoke.py -- no mocks of the mechanism itself.

Marked slow: runs real (tiny) training loops. Deselect with `-m "not slow"`.
"""
import json
import os

import pytest
from datacore import BestFitCropPacker, DataManager, FileSystemDatasetStore

from tinylab import job
from tinylab.ops import Context
from tinylab.ops.train import run as train_run
from tinylab.tokenizer import get_tokenizer

pytestmark = pytest.mark.slow

_TINY_GPT_CONFIG = os.path.join(os.path.dirname(__file__), "fixtures", "gpt_tiny.json")

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the old stone bridge.",
    "A small tinylab model learns to predict the next token in a short sentence.",
    "Yesterday the weather was warm and the sky stayed clear until evening.",
    "Researchers trained a tiny transformer on a handful of synthetic sentences.",
]


class _FakeTextSource:
    def __init__(self, sentences, repeats):
        self.sentences = sentences
        self.repeats = repeats

    def text_batches(self):
        yield "fake", self.sentences * self.repeats


def _prepare_fake_dataset(base_dir, tokenizer, sequence_len):
    dataset_dir = os.path.join(base_dir, "prepared", "smoke")
    store = FileSystemDatasetStore(dataset_dir)
    sources = {"train": _FakeTextSource(_SENTENCES, repeats=50), "val": _FakeTextSource(_SENTENCES, repeats=5)}
    packer = BestFitCropPacker(buffer_size=64)
    DataManager().prepare(store, sources=sources, tokenizer=tokenizer, sequence_len=sequence_len,
                           sequences_per_volume=64, packer=packer)
    return dataset_dir


def _base_cfg(sequence_len):
    return {
        "name": "pre", "op": "train", "kind": "base", "dataset": "smoke", "sequence_len": sequence_len,
        "model_config": _TINY_GPT_CONFIG,
        "device_batch_size": 2, "total_batch_size": 64, "world_size": 1,
        "eval_every": 2, "eval_tokens": 64, "save_every": 3,
    }


def test_resume_continues_bit_exactly_from_an_interrupted_step(base_dir):
    """A continuous 0->6 run and an interrupted-then-resumed 0->3, 3->6 run must land on the
    exact same val_bpb at step 6 -- the strongest signal that model weights, optimizer state, and
    dataloader position all round-tripped correctly, not just that *something* got saved."""
    tokenizer = get_tokenizer(base_dir)
    sequence_len = 32
    _prepare_fake_dataset(base_dir, tokenizer, sequence_len)

    continuous = train_run(dict(_base_cfg(sequence_len), num_iterations=6), Context(device_type="cpu", resume=False))

    checkpoint_dir = os.path.join(base_dir, "checkpoints", "pre")
    for name in os.listdir(checkpoint_dir):
        os.remove(os.path.join(checkpoint_dir, name))

    interrupted = train_run(dict(_base_cfg(sequence_len), num_iterations=3), Context(device_type="cpu", resume=False))
    assert interrupted["step"] == 3
    resumed = train_run(dict(_base_cfg(sequence_len), num_iterations=6), Context(device_type="cpu", resume=True))

    assert resumed["step"] == 6
    assert resumed["val_bpb"] == pytest.approx(continuous["val_bpb"])
    assert resumed["total_training_time"] > interrupted["total_training_time"]


def test_resume_default_false_overwrites_from_scratch(base_dir):
    """Without resume=True (today's exact default behavior), re-running the same output_tag at a
    higher num_iterations must NOT continue -- it restarts, unchanged from before this feature."""
    tokenizer = get_tokenizer(base_dir)
    sequence_len = 32
    _prepare_fake_dataset(base_dir, tokenizer, sequence_len)

    train_run(dict(_base_cfg(sequence_len), num_iterations=3), Context(device_type="cpu", resume=False))
    result = train_run(dict(_base_cfg(sequence_len), num_iterations=3), Context(device_type="cpu", resume=False))
    assert result["step"] == 3  # started fresh again, not "already done, skip"


def test_resume_world_size_mismatch_is_a_hard_error(base_dir):
    tokenizer = get_tokenizer(base_dir)
    sequence_len = 32
    _prepare_fake_dataset(base_dir, tokenizer, sequence_len)

    train_run(dict(_base_cfg(sequence_len), num_iterations=3), Context(device_type="cpu", resume=False))
    checkpoint_dir = os.path.join(base_dir, "checkpoints", "pre")
    meta_path = os.path.join(checkpoint_dir, "meta_000003.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["user_config"]["world_size"] = 4
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    with pytest.raises(AssertionError, match="doesn't reshard across world_size"):
        train_run(dict(_base_cfg(sequence_len), num_iterations=6), Context(device_type="cpu", resume=True))


def test_resume_missing_optimizer_shard_is_a_hard_error(base_dir):
    tokenizer = get_tokenizer(base_dir)
    sequence_len = 32
    _prepare_fake_dataset(base_dir, tokenizer, sequence_len)

    train_run(dict(_base_cfg(sequence_len), num_iterations=3), Context(device_type="cpu", resume=False))
    checkpoint_dir = os.path.join(base_dir, "checkpoints", "pre")
    os.remove(os.path.join(checkpoint_dir, "optim_000003_rank0.pt"))

    with pytest.raises(AssertionError, match="refusing to resume with a freshly-initialized optimizer"):
        train_run(dict(_base_cfg(sequence_len), num_iterations=6), Context(device_type="cpu", resume=True))


def test_resume_through_job_ignores_a_checkpoint_that_never_finished_saving(base_dir, tmp_path, monkeypatch):
    """The real payoff of the job-state-file mechanism, not just its plumbing: a directory scan
    alone would pick the highest-numbered checkpoint on disk, even one whose save never fully
    completed (here: model + meta written, optimizer shard not -- simulating a crash mid-save).
    Resuming *through* job.run_file must never even attempt that one -- it only trusts the job
    state file's own record, which is only ever updated after a checkpoint's save fully returns.
    Without this fix, resuming would hit the exact "missing optimizer shard" error tested above,
    on a step that was never actually usable to resume from in the first place -- even though an
    earlier, perfectly good checkpoint (step 3) was sitting right there.

    _pid_alive is stubbed to False: the crash here is simulated by having a checkpoint write raise
    and then calling job.run_file again from this same still-running test process, so the state
    file's recorded pid (this process's own) would otherwise look genuinely alive -- see
    test_job.py's own resume/pid tests for that check exercised for real."""
    monkeypatch.setattr(job, "_pid_alive", lambda pid: False)
    tokenizer = get_tokenizer(base_dir)
    sequence_len = 32
    _prepare_fake_dataset(base_dir, tokenizer, sequence_len)

    job_path = str(tmp_path / "job.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump({"steps": [dict(_base_cfg(sequence_len), num_iterations=9)]}, f)

    from modelcore.store import FileSystemStore
    original_write_optimizer_state = FileSystemStore.write_optimizer_state
    call_count = {"n": 0}

    def flaky_write_optimizer_state(self, state, rank=0):
        call_count["n"] += 1
        if call_count["n"] == 2:  # the step-6 periodic checkpoint (save_every=3: saves at 3, 6, 9)
            raise RuntimeError("simulated crash mid-checkpoint-write")
        return original_write_optimizer_state(self, state, rank=rank)

    monkeypatch.setattr(FileSystemStore, "write_optimizer_state", flaky_write_optimizer_state)
    with pytest.raises(RuntimeError, match="simulated crash mid-checkpoint-write"):
        job.run_file(job_path)

    checkpoint_dir = os.path.join(base_dir, "checkpoints", "pre")
    assert os.path.exists(os.path.join(checkpoint_dir, "model_000006.pt"))            # written...
    assert not os.path.exists(os.path.join(checkpoint_dir, "optim_000006_rank0.pt"))  # ...but incomplete
    assert os.path.exists(os.path.join(checkpoint_dir, "optim_000003_rank0.pt"))      # step 3 is fully intact

    monkeypatch.setattr(FileSystemStore, "write_optimizer_state", original_write_optimizer_state)
    results = job.run_file(job_path, resume=True)
    assert results[0]["step"] == 9  # completed -- resumed from step 3, step 6 never touched
