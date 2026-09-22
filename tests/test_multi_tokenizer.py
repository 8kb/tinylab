"""
End-to-end proof that a job can genuinely use two different tokenizers: two "tokenizer" steps,
one "prepare" (kind=base) step per tokenizer, and one tiny "train" (kind=base) step per tokenizer
-- each op resolving its OWN step's "tokenizer" key (Context.tokenizer_for) rather than the run's
single default. No network: ClimbMix's shard paths/download are stubbed with a small local parquet
fixture, the same pattern tests/test_tok_train.py uses for the `tokenizer` op alone.

Marked slow: runs two real (tiny) training loops, not instant. Deselect with `-m "not slow"`.
"""
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tinylab.ops import Context
from tinylab.ops.prepare import run as prepare_run
from tinylab.ops.tokenizer import run as tokenizer_run
from tinylab.ops.train import run as train_run
from tinylab.tokenizer import SPECIAL_TOKENS

pytestmark = pytest.mark.slow

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the old stone bridge.",
    "A small tinylab model learns to predict the next token in a short sentence.",
    "Yesterday the weather was warm and the sky stayed clear until evening.",
] * 20

# rustbpe's own floor: vocab_size - len(SPECIAL_TOKENS) must be >= 256.
_MIN_VOCAB_SIZE = 256 + len(SPECIAL_TOKENS)
_VOCAB_A = _MIN_VOCAB_SIZE + 10
_VOCAB_B = _MIN_VOCAB_SIZE + 42  # deliberately a different size, not just a different name


def _write_fake_climbmix_parquet(base_dir):
    """Same fixture tests/test_tok_train.py uses: a tiny local parquet standing in for a real
    ClimbMix shard, so the `tokenizer`/`prepare` ops exercise their real parquet-reading and
    BPE-training/packing mechanism against fixed, offline content."""
    shard_dir = os.path.join(base_dir, "base_data_climbmix")
    os.makedirs(shard_dir, exist_ok=True)
    table = pa.table({"text": _SENTENCES})
    train_path = os.path.join(shard_dir, "shard_00000.parquet")
    val_path = os.path.join(shard_dir, "shard_06542.parquet")
    pq.write_table(table, train_path)
    pq.write_table(table, val_path)
    return train_path, val_path


def _patch_fixture_shards(base_dir, monkeypatch):
    train_path, val_path = _write_fake_climbmix_parquet(base_dir)
    from tinylab import data
    monkeypatch.setattr(data, "climbmix_train_val_paths", lambda num_train_shards: ([train_path], [val_path]))
    monkeypatch.setattr(data, "download_climbmix_shards", lambda n, num_workers=4, log=print: None)


def _write_tiny_gpt_config(path, vocab_size, sequence_len=32, n_embd=32, n_head=2, head_dim=16):
    """A 1-layer gpt tree, shaped like tests/fixtures/gpt_tiny.json but parameterized by
    vocab_size -- each tokenizer in this test needs its own model config, since a train step's
    model_config.vocab_size must match its tokenizer's (checkpoints.build_model asserts this)."""
    config = {
        "format": "modelcore.v2", "sequence_len": sequence_len, "vocab_size": vocab_size,
        "n_embd": n_embd, "pad_vocab_size_to": 8, "template": "base",
        "shared": {
            "rope": {"#type": "rotary", "head_dim": head_dim, "over_compute": 10},
            "norm": {"#type": "rms_norm", "eps": None},
        },
        "input": {"#type": "token_embedding", "smear": True},
        "body": {"#type": "backout", "backout_layer": 0, "backout_lambda_init": 0.2, "blocks": [
            {"#type": "gpt_block", "layer_idx": 0, "n_head": n_head, "n_kv_head": n_head, "head_dim": head_dim,
             "window": -1, "has_value_embed": False, "resid_lambda_init": 1.0, "x0_lambda_init": 0.0,
             "mlp": {"#type": "mlp", "activation": "relu2", "hidden_dim": 64}},
        ]},
        "output": {"#type": "lm_head", "softcap": 15},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f)


def test_two_tokenizer_steps_train_genuinely_different_vocabs(base_dir, monkeypatch):
    _patch_fixture_shards(base_dir, monkeypatch)
    ctx = Context(device_type="cpu")  # no run-default tokenizer_spec -- every step names its own
    tokenizer_run({"name": "tok_a", "op": "tokenizer", "vocab_size": _VOCAB_A, "shards": 1, "output": "vocab_a"}, ctx)
    tokenizer_run({"name": "tok_b", "op": "tokenizer", "vocab_size": _VOCAB_B, "shards": 1, "output": "vocab_b"}, ctx)
    tok_a, tok_b = ctx.tokenizer_for("vocab_a"), ctx.tokenizer_for("vocab_b")
    assert tok_a.get_vocab_size() == _VOCAB_A
    assert tok_b.get_vocab_size() == _VOCAB_B
    assert tok_a.fingerprint() != tok_b.fingerprint()


def test_prepare_step_uses_its_own_tokenizer_not_the_run_default(base_dir, monkeypatch):
    _patch_fixture_shards(base_dir, monkeypatch)
    ctx = Context(device_type="cpu")
    tokenizer_run({"name": "tok_a", "op": "tokenizer", "vocab_size": _VOCAB_A, "shards": 1, "output": "vocab_a"}, ctx)
    tokenizer_run({"name": "tok_b", "op": "tokenizer", "vocab_size": _VOCAB_B, "shards": 1, "output": "vocab_b"}, ctx)

    result_a = prepare_run({"name": "prep_a", "op": "prepare", "kind": "base", "sequence_len": 32,
                            "tokenizer": "vocab_a", "shards": 1}, ctx)
    result_b = prepare_run({"name": "prep_b", "op": "prepare", "kind": "base", "sequence_len": 32,
                            "tokenizer": "vocab_b", "shards": 1}, ctx)

    # Different tokenizers -> different auto dataset names (the name embeds the fingerprint) and
    # different directories on disk, not one dataset silently shared or overwritten by the other.
    assert result_a["dataset"] != result_b["dataset"]
    assert result_a["dataset_dir"] != result_b["dataset_dir"]
    assert os.path.exists(result_a["dataset_dir"])
    assert os.path.exists(result_b["dataset_dir"])

    from datacore import DataManager, FileSystemDatasetStore
    manager = DataManager()
    dataset_a = manager.open(FileSystemDatasetStore(result_a["dataset_dir"]))
    dataset_b = manager.open(FileSystemDatasetStore(result_b["dataset_dir"]))
    assert dataset_a.info.tokenizer_fingerprint == ctx.tokenizer_for("vocab_a").fingerprint()
    assert dataset_b.info.tokenizer_fingerprint == ctx.tokenizer_for("vocab_b").fingerprint()
    assert dataset_a.info.tokenizer_fingerprint != dataset_b.info.tokenizer_fingerprint


def test_train_step_records_its_own_steps_tokenizer_in_the_checkpoint(base_dir, monkeypatch, tmp_path):
    # MuonAdamW.step is unconditionally @torch.compile'd (see modelcore/AGENTS.md), and its
    # compiled-shape cache is torch._dynamo's process-global state, not per-test: two real models
    # of different vocab_size here, on top of every other test in the same pytest process that
    # also trains a real (differently-shaped) model, can exceed the default recompile_limit=8 --
    # an environment quirk, not a correctness issue, but one that would otherwise leak into
    # whichever test happens to run next. Raise the limit for this test's own compiles, then
    # reset() the cache so later tests in the same session start with their own fresh budget.
    import torch
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 32)
    try:
        _patch_fixture_shards(base_dir, monkeypatch)
        ctx = Context(device_type="cpu")
        tokenizer_run({"name": "tok_a", "op": "tokenizer", "vocab_size": _VOCAB_A, "shards": 1, "output": "vocab_a"}, ctx)
        tokenizer_run({"name": "tok_b", "op": "tokenizer", "vocab_size": _VOCAB_B, "shards": 1, "output": "vocab_b"}, ctx)
        prepare_run({"name": "prep_a", "op": "prepare", "kind": "base", "sequence_len": 32, "tokenizer": "vocab_a", "shards": 1}, ctx)
        prepare_run({"name": "prep_b", "op": "prepare", "kind": "base", "sequence_len": 32, "tokenizer": "vocab_b", "shards": 1}, ctx)

        config_a, config_b = str(tmp_path / "cfg_a.json"), str(tmp_path / "cfg_b.json")
        _write_tiny_gpt_config(config_a, _VOCAB_A)
        _write_tiny_gpt_config(config_b, _VOCAB_B)

        train_cfg = dict(kind="base", sequence_len=32, device_batch_size=1, total_batch_size=32,
                         num_iterations=1, eval_every=-1, eval_tokens=32, world_size=1, save_every=0)
        train_run(dict(train_cfg, name="train_a", tokenizer="vocab_a", model_config=config_a), ctx)
        train_run(dict(train_cfg, name="train_b", tokenizer="vocab_b", model_config=config_b), ctx)

        from tinylab import checkpoints
        _model_a, tok_a, meta_a = checkpoints.load_model("train_a", "cpu", phase="eval")
        _model_b, tok_b, meta_b = checkpoints.load_model("train_b", "cpu", phase="eval")
        # Each checkpoint's own recorded tokenizer name matches the step that trained it, not the
        # other step's -- this is exactly what Context.tokenizer_name_for(step's own spec) fixes:
        # the run has no single default tokenizer here, so a checkpoint recording the wrong one
        # would be silently unrecoverable (get_tokenizer(tokenizer=<wrong name>) reads back a
        # real, but different, vocabulary).
        assert meta_a["model_config"]["tokenizer"]["name"] == "vocab_a"
        assert meta_b["model_config"]["tokenizer"]["name"] == "vocab_b"
        assert tok_a.get_vocab_size() == _VOCAB_A
        assert tok_b.get_vocab_size() == _VOCAB_B
    finally:
        torch._dynamo.reset()
