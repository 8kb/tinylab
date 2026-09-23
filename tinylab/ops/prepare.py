"""
The `prepare` op: tokenize + pack a corpus into a datacore dataset. Ported from nanochat's
scripts/data_prep.py -- kind="base" downloads ClimbMix shards (if not already present) and packs
them with BestFitCropPacker; kind="sft" builds the SmolTalk + MMLU + GSM8K conversation mixture and
packs it with BestFitPadPacker. See docs/job-file.md for every cfg key this module reads.
"""
import os
import time

from datacore import BestFitCropPacker, BestFitPadPacker, ExampleMixture, ExampleTokenSource, FileSystemDatasetStore, ParquetDirectorySource

from tinylab import data
from tinylab.runtime import get_base_dir, print0
from tinylab.tokenizer import DEFAULT_MAX_TOKENS_PER_CONVERSATION

# Keys shared by both kinds, plus each kind's own -- see accepted_keys() below, which is kind-
# aware so e.g. "mmlu_epochs" on a kind="base" step is caught as an error rather than silently
# accepted and ignored. "tokenizer" is already in ops.COMMON_KEYS (every op accepts it -- it's the
# step's own tokenizer selection, see Context.tokenizer_for), so it isn't repeated here.
_COMMON_KEYS = {"kind", "dataset", "sequences_per_volume", "buffer_size", "tokenizer_threads"}
_BASE_KEYS = {"shards"}
_SFT_KEYS = {"max_conversations", "mmlu_epochs", "gsm8k_epochs", "max_tokens_per_conversation", "sft_padding_id"}


def accepted_keys(cfg: dict) -> set:
    kind = cfg.get("kind")
    kind_keys = _BASE_KEYS if kind == "base" else _SFT_KEYS if kind == "sft" else (_BASE_KEYS | _SFT_KEYS)
    return _COMMON_KEYS | kind_keys


def prepared_dir(name: str) -> str:
    return os.path.join(get_base_dir(), "prepared", name)


def default_dataset_name(kind: str, sequence_len: int, tokenizer) -> str:
    stem = "climbmix" if kind == "base" else "sft"
    return f"{stem}_t{sequence_len}_{tokenizer.fingerprint()}"


def _prepare_base(cfg, ctx, sequence_len, tokenizer):
    """kind="base": ensures cfg["shards"] ClimbMix train shards (plus the fixed val shard) are on
    disk, downloading whatever's missing, then packs exactly those paths -- never however many
    shard files a larger previous run happened to leave around. Returns (dataset_name, dataset_dir,
    manifest)."""
    manager = ctx.data_manager
    dataset_name = cfg.get("dataset") or default_dataset_name("base", sequence_len, tokenizer)
    dataset_dir = prepared_dir(dataset_name)
    store = FileSystemDatasetStore(dataset_dir)

    shards = cfg.get("shards", 8)
    train_paths, val_paths = data.climbmix_train_val_paths(shards)
    if any(not os.path.exists(p) for p in train_paths + val_paths):
        data.download_climbmix_shards(shards, log=print0)
        missing = [p for p in train_paths + val_paths if not os.path.exists(p)]
        assert not missing, f"still missing after download: {missing}"
    print0(f"Preparing base dataset {dataset_name!r}: {len(train_paths)} train shard(s), {len(val_paths)} val shard(s) -> {dataset_dir}")

    sources = {"train": ParquetDirectorySource(paths=train_paths), "val": ParquetDirectorySource(paths=val_paths)}
    packer = BestFitCropPacker(buffer_size=cfg.get("buffer_size", 1000))
    t0 = time.time()
    manifest = manager.prepare(
        store, sources=sources, tokenizer=tokenizer, sequence_len=sequence_len,
        sequences_per_volume=cfg.get("sequences_per_volume", 16384), packer=packer,
        num_threads=cfg.get("tokenizer_threads", os.cpu_count()),
    )
    print0(f"Prepared in {time.time() - t0:.1f}s")
    return dataset_name, dataset_dir, manifest


def _build_sft_mixtures(cfg, ctx):
    """Builds (train_mixture, val_mixture): SmolTalk + cfg["mmlu_epochs"] passes of MMLU's
    auxiliary-train split + cfg["gsm8k_epochs"] passes of GSM8K's train split for train; a smaller
    fixed val mixture (SmolTalk's own test split, plus capped MMLU/GSM8K test slices). Each is
    truncated to cfg["max_conversations"] if given, for a fast smoke-sized run."""
    from benchcore import GSM8K, MMLU
    cache_dir = get_base_dir()
    mmlu_epochs = cfg.get("mmlu_epochs", 3)
    gsm8k_epochs = cfg.get("gsm8k_epochs", 4)
    train_tasks = [
        data.SmolTalk(split="train"),
        *[MMLU(subset="all", split="auxiliary_train", cache_dir=cache_dir) for _ in range(mmlu_epochs)],
        *[GSM8K(subset="main", split="train", cache_dir=cache_dir) for _ in range(gsm8k_epochs)],
    ]
    # max_conversations (smoke tests) caps both mixtures via ExampleMixture's own stop= kwarg --
    # __len__ clamps stop to each mixture's true length itself, so this needs no separate
    # min(max_conversations, len(val_mixture)) either -- see datacore.records.ExampleSet.
    max_conversations = cfg.get("max_conversations")
    train_mixture = ExampleMixture(train_tasks, stop=max_conversations)
    val_mixture = ExampleMixture([
        data.SmolTalk(split="test"),
        MMLU(subset="all", split="test", cache_dir=cache_dir, stop=5200),
        GSM8K(subset="main", split="test", cache_dir=cache_dir, stop=420),
    ], stop=max_conversations)
    return train_mixture, val_mixture


def _prepare_sft(cfg, ctx, sequence_len, tokenizer):
    """kind="sft": builds the conversation mixture (see _build_sft_mixtures), renders each
    conversation to (ids, loss mask) via the tokenizer, and packs the result with BestFitPadPacker.
    Returns (dataset_name, dataset_dir, manifest)."""
    manager = ctx.data_manager
    dataset_name = cfg.get("dataset") or default_dataset_name("sft", sequence_len, tokenizer)
    dataset_dir = prepared_dir(dataset_name)
    store = FileSystemDatasetStore(dataset_dir)

    train_mixture, val_mixture = _build_sft_mixtures(cfg, ctx)
    print0(f"Preparing SFT dataset {dataset_name!r}: {len(train_mixture):,} train conversations, {len(val_mixture):,} val -> {dataset_dir}")

    max_tokens = cfg.get("max_tokens_per_conversation", DEFAULT_MAX_TOKENS_PER_CONVERSATION)
    bos_id = tokenizer.get_bos_token_id()
    render = lambda conversation: tokenizer.render_conversation(conversation, max_tokens=max_tokens)
    sources = {
        "train": ExampleTokenSource(train_mixture, render, "train"),
        "val": ExampleTokenSource(val_mixture, render, "val"),
    }
    packer = BestFitPadPacker(bos_token_id=bos_id, padding_id=cfg.get("sft_padding_id"), buffer_size=cfg.get("buffer_size", 1000))
    t0 = time.time()
    manifest = manager.prepare(
        store, sources=sources, tokenizer=tokenizer, sequence_len=sequence_len,
        sequences_per_volume=cfg.get("sequences_per_volume", 16384), packer=packer,
        num_threads=cfg.get("tokenizer_threads", os.cpu_count()),
    )
    print0(f"Prepared in {time.time() - t0:.1f}s")
    return dataset_name, dataset_dir, manifest


def _split_summary(split_manifest: dict) -> dict:
    """Every per-split stat datacore's manifest already computes (writer.SplitTotals, see
    datacore/docs/architecture.md), not just the num_sequences/num_tokens this used to cherry-pick
    -- num_documents_dropped/num_tokens_dropped (packing loss) were sitting on disk unreported.
    Adds chars_per_token, derived from num_chars_encoded (0 for a TokenSource split, e.g. SFT
    conversation rendering, which never has raw text pass through datacore -- chars_per_token is
    omitted rather than reported as a fake 0.0 in that case)."""
    num_chars = split_manifest.get("num_chars_encoded", 0)
    num_tokens_encoded = split_manifest["num_tokens_encoded"]
    summary = {
        "num_sequences": split_manifest["num_sequences"], "num_tokens": split_manifest["num_tokens"],
        "num_documents": split_manifest["num_documents"], "num_documents_dropped": split_manifest["num_documents_dropped"],
        "num_tokens_encoded": num_tokens_encoded, "num_tokens_dropped": split_manifest["num_tokens_dropped"],
    }
    if num_chars > 0 and num_tokens_encoded > 0:
        summary["chars_per_token"] = num_chars / num_tokens_encoded
    return summary


def run(cfg: dict, ctx) -> dict:
    """Runs one prepare step: cfg is a resolved job-file step (see docs/job-file.md for every
    key), ctx the shared Context for this job run. Returns {"op": "prepare", "kind", "dataset",
    "dataset_dir", "splits": {"train"|"val": _split_summary(...)}}."""
    assert "kind" in cfg, "prepare: 'kind' is required ('base' or 'sft')"
    kind = cfg["kind"]
    assert kind in ("base", "sft"), f"prepare: kind must be 'base' or 'sft', got {kind!r}"
    assert "sequence_len" in cfg, "prepare: 'sequence_len' is required (put it in \"defaults\" to share it with a matching train step)"
    sequence_len = cfg["sequence_len"]
    # This step's own tokenizer (see Context.tokenizer_for): cfg["tokenizer"] if this step sets
    # one, else the run's default -- a multi-tokenizer job (e.g. preparing base/sft data for two
    # differently-sized vocabularies) sets it per prepare step, not once in "defaults".
    tokenizer = ctx.tokenizer_for(cfg.get("tokenizer"))
    if kind == "base":
        dataset_name, dataset_dir, manifest = _prepare_base(cfg, ctx, sequence_len, tokenizer)
    else:
        dataset_name, dataset_dir, manifest = _prepare_sft(cfg, ctx, sequence_len, tokenizer)
    splits = {split: _split_summary(s) for split, s in manifest["splits"].items()}
    for split, s in splits.items():
        ratio = f" | {s['chars_per_token']:.3f} chars/token" if "chars_per_token" in s else ""
        print0(f"  {split}: {s['num_sequences']:,} sequences, {s['num_tokens']:,} tokens "
               f"({s['num_documents_dropped']:,} docs / {s['num_tokens_dropped']:,} tokens dropped){ratio}")
    return {"op": "prepare", "kind": kind, "dataset": dataset_name, "dataset_dir": dataset_dir, "splits": splits}
