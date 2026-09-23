"""
`kind="sft"`'s two nanochat-ported mechanisms tinylab.ops.train used to be missing entirely --
`init_lr_frac` and the `load_optimizer` momentum warm-start (see nanochat/scripts/chat_sft.py's
own `--init-lr-frac`/`--load-optimizer`, and this repo's docs/job-file.md `## train` table). Real
(tiny) CPU training, same synthetic-corpus pattern as test_resume.py/test_smoke.py -- no mocks of
the mechanism itself.

Marked slow: runs real training loops. Deselect with `-m "not slow"`.
"""
import json
import os

import pytest
from datacore import BestFitCropPacker, DataManager, FileSystemDatasetStore

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
        "eval_every": 3, "eval_tokens": 64, "save_every": 3,
    }


def _sft_cfg(sequence_len, output_tag="chat"):
    return {
        "name": "chat", "op": "train", "kind": "sft", "dataset": "smoke", "sequence_len": sequence_len,
        "source_tag": "pre", "output_tag": output_tag,
        "device_batch_size": 2, "total_batch_size": 64, "world_size": 1,
        "eval_every": 3, "eval_tokens": 64, "save_every": -1,
    }


def _train_pretrain(base_dir):
    """Every test here starts from the same tiny 3-step base checkpoint under tag "pre"."""
    tokenizer = get_tokenizer(base_dir)
    sequence_len = 32
    _prepare_fake_dataset(base_dir, tokenizer, sequence_len)
    train_run(dict(_base_cfg(sequence_len), num_iterations=3), Context(device_type="cpu", resume=False))
    return sequence_len


def test_init_lr_frac_zero_freezes_the_model(base_dir):
    """init_lr_frac=0.0 zeroes every param group's lr -- real forward/backward/optimizer.step()
    calls happen, but produce no actual update. The strongest proof the frac reaches the
    schedule: comparing against a *second*, independent sft run that does zero training steps at
    all (num_iterations=0, eval-only) must land on the same val_bpb."""
    sequence_len = _train_pretrain(base_dir)

    frozen = train_run(dict(_sft_cfg(sequence_len, output_tag="frozen"), num_iterations=3, init_lr_frac=0.0),
                        Context(device_type="cpu", resume=False))
    untouched = train_run(dict(_sft_cfg(sequence_len, output_tag="untouched"), num_iterations=0),
                           Context(device_type="cpu", resume=False))

    assert frozen["val_bpb"] == pytest.approx(untouched["val_bpb"])


def test_load_optimizer_true_vs_false_diverge(base_dir):
    """The warm-start is real: with the same everything else, load_optimizer=True (the momentum
    buffers carry over from the pretrain checkpoint's own optimizer) and load_optimizer=False
    (a cold optimizer) must land on different val_bpb after the same steps."""
    sequence_len = _train_pretrain(base_dir)

    cold = train_run(dict(_sft_cfg(sequence_len, output_tag="cold"), num_iterations=3, load_optimizer=False),
                      Context(device_type="cpu", resume=False))
    warm = train_run(dict(_sft_cfg(sequence_len, output_tag="warm"), num_iterations=3, load_optimizer=True),
                      Context(device_type="cpu", resume=False))

    assert cold["val_bpb"] != pytest.approx(warm["val_bpb"])


def test_load_optimizer_missing_shard_is_a_hard_error(base_dir):
    sequence_len = _train_pretrain(base_dir)
    checkpoint_dir = os.path.join(base_dir, "checkpoints", "pre")
    os.remove(os.path.join(checkpoint_dir, "optim_000003_rank0.pt"))

    with pytest.raises(AssertionError, match="load_optimizer"):
        train_run(dict(_sft_cfg(sequence_len), num_iterations=3), Context(device_type="cpu", resume=False))


def test_load_optimizer_world_size_mismatch_is_a_hard_error(base_dir):
    sequence_len = _train_pretrain(base_dir)
    checkpoint_dir = os.path.join(base_dir, "checkpoints", "pre")
    meta_path = os.path.join(checkpoint_dir, "meta_000003.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["user_config"]["world_size"] = 4
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    with pytest.raises(AssertionError, match="doesn't reshard across world_size"):
        train_run(dict(_sft_cfg(sequence_len), num_iterations=3), Context(device_type="cpu", resume=False))
