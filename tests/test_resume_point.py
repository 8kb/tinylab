"""checkpoints.resume_point: which checkpoint step a step continues from. Fast -- it only looks at
the directory and a Context; no training."""
import os

import pytest

from tinylab import checkpoints
from tinylab.ops import Context


def _fake_checkpoint(base_dir, tag, step):
    directory = checkpoints.resolve_checkpoint_dir(tag)
    os.makedirs(directory, exist_ok=True)
    open(os.path.join(directory, f"model_{step:06d}.pt"), "wb").close()


def _tracked(resume, hints=None):
    return Context(device_type="cpu", resume=resume, _resume_checkpoint_steps=hints or {},
                   _on_checkpoint=lambda name, step: None)


def test_no_resume_flag_means_a_fresh_start_even_with_checkpoints_on_disk(base_dir):
    _fake_checkpoint(base_dir, "pre", 3)
    assert checkpoints.resume_point(_tracked(resume=False), "pre", "pre", 1) is None


def test_the_state_files_hint_wins_over_the_directory(base_dir):
    _fake_checkpoint(base_dir, "pre", 3)
    _fake_checkpoint(base_dir, "pre", 5)  # newer on disk, but the hint says only 3 finished saving
    assert checkpoints.resume_point(_tracked(True, {"pre": 3}), "pre", "pre", 1) == 3


def test_a_bare_context_falls_back_to_the_directory_scan(base_dir):
    _fake_checkpoint(base_dir, "pre", 3)
    assert checkpoints.resume_point(Context(device_type="cpu", resume=True), "pre", "pre", 1) == 3


def test_nothing_on_disk_is_an_ordinary_fresh_start(base_dir):
    assert checkpoints.resume_point(_tracked(True), "pre", "pre", 1) is None
    assert checkpoints.resume_point(Context(device_type="cpu", resume=True), "pre", "pre", 1) is None


def test_a_tracked_run_with_no_record_but_a_local_checkpoint_is_an_error_not_a_guess(base_dir):
    """The state file is gone (an earlier run finished) and a checkpoint is on disk: resuming from a
    directory scan could re-enter a finished step, or load a half-written file."""
    _fake_checkpoint(base_dir, "pre", 3)
    with pytest.raises(RuntimeError, match="no record for this step"):
        checkpoints.resume_point(_tracked(True), "pre", "pre", 1)
