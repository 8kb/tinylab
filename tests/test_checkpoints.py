"""Tests for tinylab.checkpoints.find_last_step: picks the highest step from model_NNNNNN.pt
filenames in a checkpoint directory."""
import pytest

from tinylab.checkpoints import find_last_step


def test_finds_the_highest_step(tmp_path):
    for step in (0, 10, 5):
        (tmp_path / f"model_{step:06d}.pt").touch()
    assert find_last_step(str(tmp_path)) == 10


def test_ignores_non_matching_files(tmp_path):
    (tmp_path / "model_000003.pt").touch()
    (tmp_path / "meta_000003.json").touch()
    (tmp_path / "optim_000003_rank0.pt").touch()
    (tmp_path / "notes.txt").touch()
    assert find_last_step(str(tmp_path)) == 3


def test_raises_file_not_found_when_no_checkpoint_exists(tmp_path):
    (tmp_path / "meta_000003.json").touch()  # a stray sibling file, but no model_*.pt
    with pytest.raises(FileNotFoundError):
        find_last_step(str(tmp_path))
