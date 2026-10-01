"""Shared fixtures. `base_dir` isolates TINYLAB_BASE_DIR to a fresh tmp_path for a test -- used by
any test that touches the tokenizer cache, a prepared dataset, or a checkpoint directory, so tests
never read or write the real ~/.cache/tinylab/."""
import pytest


@pytest.fixture
def base_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYLAB_BASE_DIR", str(tmp_path))
    return str(tmp_path)


# ---- a tiny trained model, for the tests that need a checkpoint or a prepared dataset -------------
# Same synthetic in-memory corpus pattern as test_smoke.py/test_resume.py: no network, seconds on CPU.
import os

TINY_NANOGPT_CONFIG = os.path.join(os.path.dirname(__file__), "fixtures", "nanogpt_tiny.json")
TINY_SEQUENCE_LEN = 32
_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the old stone bridge.",
    "A small tinylab model learns to predict the next token in a short sentence.",
    "Yesterday the weather was warm and the sky stayed clear until evening.",
    "Researchers trained a tiny transformer on a handful of synthetic sentences.",
]


class _FakeTextSource:
    def __init__(self, repeats):
        self.repeats = repeats

    def text_batches(self):
        yield "fake", _SENTENCES * self.repeats


@pytest.fixture
def tiny_dataset(base_dir):
    """A prepared kind="base" dataset named "smoke" at TINY_SEQUENCE_LEN, under the isolated base dir."""
    from datacore import BestFitCropPacker, DataManager, FileSystemDatasetStore
    from tinylab.tokenizer import get_tokenizer
    store = FileSystemDatasetStore(os.path.join(base_dir, "prepared", "smoke"))
    DataManager().prepare(store, sources={"train": _FakeTextSource(50), "val": _FakeTextSource(5)},
                          tokenizer=get_tokenizer(base_dir), sequence_len=TINY_SEQUENCE_LEN,
                          sequences_per_volume=64, packer=BestFitCropPacker(buffer_size=64))
    return "smoke"


@pytest.fixture
def tiny_checkpoint(tiny_dataset):
    """Trains 2 steps of the tiny GPT on the "smoke" dataset and saves it as tag "pre"; returns the tag."""
    from tinylab.ops import Context
    from tinylab.ops.train import run as train_run
    cfg = {"name": "pre", "op": "train", "kind": "base", "dataset": tiny_dataset, "sequence_len": TINY_SEQUENCE_LEN,
           "model_config": TINY_NANOGPT_CONFIG, "device_batch_size": 2, "total_batch_size": 64, "world_size": 1,
           "num_iterations": 2, "eval_every": 2, "eval_tokens": 64}
    train_run(cfg, Context(device_type="cpu"))
    return "pre"
