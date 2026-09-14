"""Shared fixtures. `base_dir` isolates TINYLAB_BASE_DIR to a fresh tmp_path for a test -- used by
any test that touches the tokenizer cache, a prepared dataset, or a checkpoint directory, so tests
never read or write the real ~/.cache/tinylab/."""
import pytest


@pytest.fixture
def base_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYLAB_BASE_DIR", str(tmp_path))
    return str(tmp_path)
