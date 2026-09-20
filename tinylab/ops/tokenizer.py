"""
The `tokenizer` op: trains a fresh BPE vocab and writes it to <base_dir>/tokenizer/, overwriting
whatever was there (the bundled default, or an earlier trained one). Ported from nanochat's
scripts/tok_train.py -- see tinylab.tokenizer for the training mechanism itself (rustbpe +
tiktoken), this module only owns the corpus (ClimbMix, same shards "prepare" kind="base" downloads)
and job-file key handling.

Must run before any earlier step in the job file's "steps" list has already touched ctx.tokenizer
(it lazily loads and caches on first use -- put a "tokenizer" step first if you use one at all;
most job files never need this op, since tinylab ships a committed default vocab).
"""
import os
import time

import torch

from tinylab import data
from tinylab.runtime import get_base_dir, print0
from tinylab.tokenizer import RustBPETokenizer

_COMMON_KEYS = {"max_chars", "doc_cap", "vocab_size", "shards"}


def accepted_keys(cfg: dict) -> set:
    return _COMMON_KEYS


def _text_iterator(train_paths, doc_cap, max_chars):
    """Flattens datacore.ParquetDirectorySource's (path, texts) yield into a doc-cap-truncated,
    max-chars-bounded stream of raw document strings -- the same shape nanochat's own
    parquets_iter_batched-based text_iterator produces, reusing datacore's already-present parquet
    reader instead of a second pyarrow call site."""
    from datacore import ParquetDirectorySource
    nchars = 0
    for _path, texts in ParquetDirectorySource(paths=train_paths).text_batches():
        for text in texts:
            doc = text[:doc_cap] if len(text) > doc_cap else text
            nchars += len(doc)
            yield doc
            if nchars > max_chars:
                return


def run(cfg: dict, ctx) -> dict:
    """Runs one tokenizer step: cfg is a resolved job-file step (see docs/job-file.md), ctx the
    shared Context for this job run (unused here beyond matching every other op's signature --
    this op builds its own tokenizer rather than reading ctx.tokenizer, since it's the one op that
    replaces it). Returns {"op": "tokenizer", "vocab_size", "train_time"}."""
    shards = cfg.get("shards", 8)
    train_paths, _val_paths = data.climbmix_train_val_paths(shards)
    if any(not os.path.exists(p) for p in train_paths):
        data.download_climbmix_shards(shards, log=print0)

    vocab_size = cfg.get("vocab_size", 32768)
    doc_cap = cfg.get("doc_cap", 10_000)
    max_chars = cfg.get("max_chars", 2_000_000_000)
    print0(f"Training tokenizer: vocab_size={vocab_size:,} doc_cap={doc_cap:,} max_chars={max_chars:,}")

    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(_text_iterator(train_paths, doc_cap, max_chars), vocab_size)
    train_time = time.time() - t0
    print0(f"Trained tokenizer in {train_time:.1f}s")

    tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    tokenizer.save(tokenizer_dir)
    token_bytes = torch.tensor(tokenizer.token_byte_lengths(), dtype=torch.int32)
    torch.save(token_bytes, os.path.join(tokenizer_dir, "token_bytes.pt"))
    print0(f"Saved tokenizer to {tokenizer_dir}")

    return {"op": "tokenizer", "vocab_size": tokenizer.get_vocab_size(), "train_time": train_time}
