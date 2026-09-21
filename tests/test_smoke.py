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

# A materialized 2-layer gpt tree at sequence_len=32, vocab_size=32768 (the bundled default
# tokenizer's own vocab size) -- dumped once via nanochat's `scripts/model_info.py --arch gpt
# --depth 2 --aspect-ratio 32 --head-dim 16 --max-seq-len 32 --vocab-size 32768 --dump-config`.
# tinylab does no preset/depth-dial derivation of its own any more -- see AGENTS.md.
_TINY_GPT_CONFIG = os.path.join(os.path.dirname(__file__), "fixtures", "gpt_tiny.json")

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
        "model_config": _TINY_GPT_CONFIG,
        "device_batch_size": 2, "total_batch_size": 64, "num_iterations": 3, "world_size": 1,
        "eval_every": 3, "eval_tokens": 64,
    }
    result = train_run(cfg, ctx)

    assert result["output_tag"] == "pre"
    assert result["val_bpb"] is not None and result["val_bpb"] > 0
    checkpoint_dir = os.path.join(base_dir, "checkpoints", "pre")
    assert os.path.isdir(checkpoint_dir)

    model, loaded_tokenizer, meta = load_model("pre", torch.device("cpu"), phase="eval")
    assert meta["val_bpb"] == pytest.approx(result["val_bpb"])
    # The checkpoint says which tokenizer it needs, and how it is talked to (a base run keeps the
    # template its model_config named).
    saved = meta["model_config"]
    assert saved["format"] == "modelcore.v2" and saved["template"] == "base"
    assert saved["tokenizer"] == tokenizer.descriptor("default")
    assert saved["tokenizer"]["fingerprint"] == loaded_tokenizer.fingerprint() == meta["tokenizer_fingerprint"]
    model.eval()

    engine = Engine(model, loaded_tokenizer)
    prompt = loaded_tokenizer.encode("The quick brown fox", prepend="<|bos|>")
    results, masks = engine.generate_batch(prompt, num_samples=1, max_tokens=8, temperature=0.0)
    assert len(results) == 1
    # generate_batch seeds `results` with the prompt itself (engine.py), so ">=" would pass even
    # if nothing were generated at all -- assert strictly more came back, i.e. a real continuation.
    assert len(results[0]) > len(prompt)


def test_sft_step_reads_one_tag_writes_another_and_declares_the_chat_template(base_dir):
    """The flat checkpoint namespace end to end: an sft step reads a base tag, writes a *nested* tag,
    records its provenance, and stamps the nanochat template (a base run keeps "base")."""
    tokenizer = get_tokenizer(base_dir)
    sequence_len = 32
    _prepare_fake_dataset(base_dir, tokenizer, sequence_len)
    ctx = Context(device_type="cpu")
    common = {"op": "train", "dataset": "smoke", "sequence_len": sequence_len, "device_batch_size": 2,
              "total_batch_size": 64, "world_size": 1, "eval_every": 2, "eval_tokens": 64}
    train_run(dict(common, name="pre", kind="base", model_config=_TINY_GPT_CONFIG, num_iterations=2), ctx)
    result = train_run(dict(common, name="chat", kind="sft", source_tag="pre", output_tag="kvcache/d13-chat",
                            num_iterations=2), ctx)

    assert result["output_tag"] == "kvcache/d13-chat"
    assert os.path.isdir(os.path.join(base_dir, "checkpoints", "kvcache", "d13-chat"))
    _, _, base_meta = load_model("pre", torch.device("cpu"), phase="eval")
    _, _, sft_meta = load_model("kvcache/d13-chat", torch.device("cpu"), phase="eval")
    assert base_meta["model_config"]["template"] == "base"
    assert sft_meta["model_config"]["template"] == "nanochat"
    assert sft_meta["base_model_tag"] == "pre"
    assert sft_meta["model_config"]["tokenizer"] == tokenizer.descriptor("default")


def test_sft_step_may_not_write_into_its_own_source_tag(base_dir):
    """One namespace means a base and an sft checkpoint can no longer share a tag by living in
    different directories (nanochat reuses "d12" for both) -- refuse before touching anything."""
    cfg = {"name": "chat", "op": "train", "kind": "sft", "sequence_len": 32, "total_batch_size": 64, "eval_tokens": 64,
           "world_size": 1, "source_tag": "d12", "output_tag": "d12"}
    with pytest.raises(AssertionError, match="output_tag == source_tag"):
        train_run(cfg, Context(device_type="cpu"))
    by_name_default = {k: v for k, v in cfg.items() if k != "output_tag"}
    by_name_default["name"] = "d12"  # no output_tag: the step's own name is the tag, and it equals source_tag
    with pytest.raises(AssertionError, match="output_tag == source_tag"):
        train_run(by_name_default, Context(device_type="cpu"))
