"""
Tokenizer training: RustBPETokenizer.train_from_iterator/save (real, no network -- a tiny in-memory
corpus) and the `tokenizer` op's own plumbing (real rustbpe training, but against a small local
parquet fixture instead of a real ClimbMix download)."""
import os

import pytest

from tinylab.ops import Context
from tinylab.ops.tokenizer import run as tokenizer_run
from tinylab.tokenizer import SPECIAL_TOKENS, RustBPETokenizer

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the old stone bridge.",
    "A small tinylab model learns to predict the next token in a short sentence.",
    "Yesterday the weather was warm and the sky stayed clear until evening.",
] * 20

# rustbpe's own floor: vocab_size - len(SPECIAL_TOKENS) must be >= 256.
_MIN_VOCAB_SIZE = 256 + len(SPECIAL_TOKENS)


def test_train_from_iterator_produces_a_valid_tokenizer():
    tok = RustBPETokenizer.train_from_iterator(iter(_SENTENCES), vocab_size=_MIN_VOCAB_SIZE + 40)
    assert tok.get_vocab_size() == _MIN_VOCAB_SIZE + 40
    ids = tok.encode("The quick brown fox")
    assert tok.decode(ids) == "The quick brown fox"


def test_train_from_iterator_places_special_tokens_at_the_top_of_the_vocab():
    vocab_size = _MIN_VOCAB_SIZE + 40
    tok = RustBPETokenizer.train_from_iterator(iter(_SENTENCES), vocab_size=vocab_size)
    offset = vocab_size - len(SPECIAL_TOKENS)
    for i, name in enumerate(SPECIAL_TOKENS):
        assert tok.encode_special(name) == offset + i
    assert tok.get_bos_token_id() == offset  # "<|bos|>" is SPECIAL_TOKENS[0]


def test_save_and_from_directory_round_trip(tmp_path):
    tok = RustBPETokenizer.train_from_iterator(iter(_SENTENCES), vocab_size=_MIN_VOCAB_SIZE + 10)
    tok.save(str(tmp_path))
    assert os.path.exists(tmp_path / "tokenizer.pkl")
    reloaded = RustBPETokenizer.from_directory(str(tmp_path))
    assert reloaded.get_vocab_size() == tok.get_vocab_size()
    assert reloaded.fingerprint() == tok.fingerprint()
    assert reloaded.encode("The quick brown fox") == tok.encode("The quick brown fox")


def test_token_byte_lengths_matches_vocab_size_after_training():
    tok = RustBPETokenizer.train_from_iterator(iter(_SENTENCES), vocab_size=_MIN_VOCAB_SIZE + 10)
    lengths = tok.token_byte_lengths()
    assert len(lengths) == tok.get_vocab_size()
    special_ids = {tok.encode_special(name) for name in SPECIAL_TOKENS}
    for token_id in special_ids:
        assert lengths[token_id] == 0


def _write_fake_climbmix_parquet(base_dir):
    """A tiny local parquet file standing in for a real ClimbMix shard -- the `tokenizer` op reads
    it exactly the way it would read a real one (ParquetDirectorySource, "text" column); only the
    shard *paths*/download are stubbed, not the parquet-reading or BPE-training mechanism."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    shard_dir = os.path.join(base_dir, "base_data_climbmix")
    os.makedirs(shard_dir, exist_ok=True)
    table = pa.table({"text": _SENTENCES})
    train_path = os.path.join(shard_dir, "shard_00000.parquet")
    val_path = os.path.join(shard_dir, "shard_06542.parquet")
    pq.write_table(table, train_path)
    pq.write_table(table, val_path)
    return train_path, val_path


def test_tokenizer_op_trains_and_writes_a_real_tokenizer_against_a_local_fixture(base_dir, monkeypatch):
    train_path, val_path = _write_fake_climbmix_parquet(base_dir)

    from tinylab import data
    monkeypatch.setattr(data, "climbmix_train_val_paths", lambda num_train_shards: ([train_path], [val_path]))
    downloaded = []
    monkeypatch.setattr(data, "download_climbmix_shards", lambda n, num_workers=4, log=print: downloaded.append(n))

    cfg = {"name": "tok", "op": "tokenizer", "vocab_size": _MIN_VOCAB_SIZE + 10, "shards": 1}
    result = tokenizer_run(cfg, Context(device_type="cpu"))

    assert result["op"] == "tokenizer"
    assert result["vocab_size"] == _MIN_VOCAB_SIZE + 10
    assert result["chars_per_token"] > 0  # compression stat, measured over the same fixture corpus
    assert not downloaded  # the fixture file already exists -- no download should have been attempted

    tokenizer_dir = os.path.join(base_dir, "tokenizers", "default")
    assert result["output"] == tokenizer_dir
    assert os.path.exists(os.path.join(tokenizer_dir, "tokenizer.pkl"))
    assert os.path.exists(os.path.join(tokenizer_dir, "token_bytes.pt"))

    # A subsequent get_tokenizer() picks up what was just trained, not the bundled default.
    from tinylab.tokenizer import get_tokenizer
    assert get_tokenizer(base_dir).get_vocab_size() == _MIN_VOCAB_SIZE + 10


def test_tokenizer_op_downloads_missing_shards(base_dir, monkeypatch):
    from tinylab import data
    downloaded = []

    def fake_download(n, num_workers=4, log=print):
        downloaded.append(n)
        return _write_fake_climbmix_parquet(base_dir)[0]

    monkeypatch.setattr(data, "download_climbmix_shards", fake_download)
    monkeypatch.setattr(data, "climbmix_train_val_paths",
                         lambda num_train_shards: ([os.path.join(base_dir, "base_data_climbmix", "shard_00000.parquet")],
                                                    [os.path.join(base_dir, "base_data_climbmix", "shard_06542.parquet")]))

    cfg = {"name": "tok", "op": "tokenizer", "vocab_size": _MIN_VOCAB_SIZE + 10, "shards": 1}
    tokenizer_run(cfg, Context(device_type="cpu"))
    assert downloaded == [1]


def _patch_fixture_shards(base_dir, monkeypatch):
    train_path, val_path = _write_fake_climbmix_parquet(base_dir)
    from tinylab import data
    monkeypatch.setattr(data, "climbmix_train_val_paths", lambda num_train_shards: ([train_path], [val_path]))
    monkeypatch.setattr(data, "download_climbmix_shards", lambda n, num_workers=4, log=print: None)


def test_tokenizer_op_writes_to_the_named_output_and_leaves_the_default_alone(base_dir, monkeypatch):
    _patch_fixture_shards(base_dir, monkeypatch)
    from tinylab.tokenizer import get_tokenizer

    bundled_vocab = get_tokenizer(base_dir).get_vocab_size()
    cfg = {"name": "tok", "op": "tokenizer", "vocab_size": _MIN_VOCAB_SIZE + 10, "shards": 1, "output": "small"}
    result = tokenizer_run(cfg, Context(device_type="cpu"))

    assert result["output"] == os.path.join(base_dir, "tokenizers", "small")
    assert get_tokenizer(base_dir, tokenizer="small").get_vocab_size() == _MIN_VOCAB_SIZE + 10
    assert get_tokenizer(base_dir).get_vocab_size() == bundled_vocab  # the default is untouched


def test_tokenizer_op_defaults_its_output_to_the_runs_tokenizer(base_dir, monkeypatch):
    _patch_fixture_shards(base_dir, monkeypatch)
    from tinylab.tokenizer import get_tokenizer

    cfg = {"name": "tok", "op": "tokenizer", "vocab_size": _MIN_VOCAB_SIZE + 10, "shards": 1, "tokenizer": "mine"}
    tokenizer_run(cfg, Context(device_type="cpu", tokenizer_spec="mine"))
    assert get_tokenizer(base_dir, tokenizer="mine").get_vocab_size() == _MIN_VOCAB_SIZE + 10


def test_tokenizer_op_refuses_to_overwrite_one_this_run_already_loaded(base_dir, monkeypatch):
    """The per-name form of "a tokenizer step must come first": a step that already loaded
    "default" would keep using its in-memory copy while the new vocab on disk went unused."""
    _patch_fixture_shards(base_dir, monkeypatch)
    ctx = Context(device_type="cpu")
    ctx.tokenizer  # an earlier step loads the default tokenizer
    cfg = {"name": "tok", "op": "tokenizer", "vocab_size": _MIN_VOCAB_SIZE + 10, "shards": 1}
    with pytest.raises(RuntimeError, match="already loaded"):
        tokenizer_run(cfg, ctx)
    # ...but training a *different* name is fine.
    tokenizer_run(dict(cfg, output="other"), ctx)


def test_context_caches_one_tokenizer_per_directory(base_dir):
    from tinylab.tokenizer import get_tokenizer
    get_tokenizer(base_dir).save(os.path.join(base_dir, "tokenizers", "copy"))
    ctx = Context(device_type="cpu")
    assert ctx.tokenizer is ctx.tokenizer_for(None) is ctx.tokenizer_for("default")
    assert ctx.tokenizer_for("copy") is not ctx.tokenizer
    assert ctx.tokenizer_name == "default" and Context(tokenizer_spec="copy").tokenizer_name == "copy"
