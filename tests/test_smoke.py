"""
End-to-end smoke test: prepares a tiny synthetic dataset (no network -- built in-memory, not
downloaded), trains a few steps, and verifies the checkpoint loads back and generates. Exercises
tinylab.ops.train + tinylab.checkpoints + tinylab.engine directly rather than through
tinylab.ops.prepare, since prepare's real corpora (ClimbMix, SmolTalk/MMLU/GSM8K) need network.

Marked slow: it runs a real (tiny) training loop, not instant. Deselect with `-m "not slow"`.
"""
import os

import pytest
import torch
from datacore import BestFitCropPacker, DataManager, FileSystemDatasetStore

from tinylab.checkpoints import load_model
from tinylab.engine import Engine
from tinylab.ops import Context
from tinylab.ops.train import run as train_run
from tinylab.tokenizer import get_tokenizer

pytestmark = pytest.mark.slow

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the old stone bridge.",
    "A small tinylab model learns to predict the next token in a short sentence.",
    "Yesterday the weather was warm and the sky stayed clear until evening.",
    "Researchers trained a tiny transformer on a handful of synthetic sentences.",
    "The cat sat on the mat and watched the rain fall outside the window.",
    "Every good pipeline starts with a small, fast, and fully offline smoke test.",
]


class _FakeTextSource:
    """A minimal datacore TextSource: yields (name, texts) with no file or network involved."""
    def __init__(self, sentences, repeats):
        self.sentences = sentences
        self.repeats = repeats

    def text_batches(self):
        yield "fake", self.sentences * self.repeats


# base_dir fixture lives in conftest.py, shared with test_tokenizer.py


def _prepare_fake_dataset(base_dir, tokenizer, sequence_len):
    dataset_dir = os.path.join(base_dir, "prepared", "smoke")
    store = FileSystemDatasetStore(dataset_dir)
    sources = {
        "train": _FakeTextSource(_SENTENCES, repeats=20),
        "val": _FakeTextSource(_SENTENCES, repeats=5),
    }
    packer = BestFitCropPacker(buffer_size=64)
    DataManager().prepare(store, sources=sources, tokenizer=tokenizer, sequence_len=sequence_len,
                           sequences_per_volume=64, packer=packer)
    return dataset_dir


def test_train_then_load_then_generate(base_dir):
    tokenizer = get_tokenizer(base_dir)
    sequence_len = 32
    _prepare_fake_dataset(base_dir, tokenizer, sequence_len)

    ctx = Context(device_type="cpu")
    cfg = {
        "name": "pre", "op": "train", "kind": "base", "dataset": "smoke", "sequence_len": sequence_len,
        "model": {"preset": "gpt", "depth": 2, "aspect_ratio": 32, "head_dim": 16},
        "device_batch_size": 2, "total_batch_size": 64, "num_iterations": 3,
        "eval_every": 3, "eval_tokens": 64,
    }
    result = train_run(cfg, ctx)

    assert result["output_tag"] == "pre"
    assert result["val_bpb"] is not None and result["val_bpb"] > 0
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", "pre")
    assert os.path.isdir(checkpoint_dir)

    model, loaded_tokenizer, meta = load_model("base", torch.device("cpu"), phase="eval", model_tag="pre")
    assert meta["val_bpb"] == pytest.approx(result["val_bpb"])
    model.eval()

    engine = Engine(model, loaded_tokenizer)
    prompt = loaded_tokenizer.encode("The quick brown fox", prepend="<|bos|>")
    results, masks = engine.generate_batch(prompt, num_samples=1, max_tokens=8, temperature=0.0)
    assert len(results) == 1
    # generate_batch seeds `results` with the prompt itself (engine.py), so ">=" would pass even
    # if nothing were generated at all -- assert strictly more came back, i.e. a real continuation.
    assert len(results[0]) > len(prompt)
