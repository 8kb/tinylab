"""
The `prepare` op: tokenize + pack a corpus into a datacore dataset. Ported from nanochat's
scripts/data_prep.py -- kind="base" downloads ClimbMix shards (if not already present) and packs
them with BestFitCropPacker; kind="sft" builds the SmolTalk + MMLU + GSM8K conversation mixture and
packs it with BestFitPadPacker. See docs/job-file.md for every cfg key this module reads.
"""
import os
import time

from datacore import BestFitCropPacker, BestFitPadPacker, EncodedDoc, ExampleMixture, FileSystemDatasetStore, ParquetDirectorySource

from tinylab import data
from tinylab.runtime import get_base_dir, print0
from tinylab.tokenizer import DEFAULT_MAX_TOKENS_PER_CONVERSATION

# Keys shared by both kinds, plus each kind's own -- see accepted_keys() below, which is kind-
# aware so e.g. "mmlu_epochs" on a kind="base" step is caught as an error rather than silently
# accepted and ignored.
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


class TaskMixtureTokenSource:
    """Adapts a datacore.ExampleMixture into datacore's TokenSource protocol: each conversation is
    rendered (ids + per-token loss mask) here, in chunks, so DataManager.prepare gets a volume
    flush boundary every `chunk_size` conversations rather than one giant flush at the end."""

    def __init__(self, task_mixture, tokenizer, name, max_tokens=DEFAULT_MAX_TOKENS_PER_CONVERSATION, chunk_size=2000):
        self.task_mixture = task_mixture
        self.tokenizer = tokenizer
        self.name = name
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size

    def token_batches(self):
        n = len(self.task_mixture)
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            docs = []
            for i in range(start, end):
                conversation = self.task_mixture[i]
                ids, mask = self.tokenizer.render_conversation(conversation, max_tokens=self.max_tokens)
                docs.append(EncodedDoc(ids=ids, mask=mask))
            yield f"{self.name}[{start}:{end}]", docs


class _Truncated:
    """Caps an ExampleSet/ExampleMixture's apparent length, for smoke-test-sized mixtures."""
    def __init__(self, task, limit):
        self.task = task
        self.limit = min(limit, len(task))
    def __len__(self):
        return self.limit
    def __getitem__(self, index):
        return self.task[index]


def _prepare_base(cfg, ctx, sequence_len):
    """kind="base": ensures cfg["shards"] ClimbMix train shards (plus the fixed val shard) are on
    disk, downloading whatever's missing, then packs exactly those paths -- never however many
    shard files a larger previous run happened to leave around. Returns (dataset_name, dataset_dir,
    manifest)."""
    manager = ctx.data_manager
    dataset_name = cfg.get("dataset") or default_dataset_name("base", sequence_len, ctx.tokenizer)
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
        store, sources=sources, tokenizer=ctx.tokenizer, sequence_len=sequence_len,
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
    train_mixture = ExampleMixture(train_tasks)
    val_mixture = ExampleMixture([
        data.SmolTalk(split="test"),
        MMLU(subset="all", split="test", cache_dir=cache_dir, stop=5200),
        GSM8K(subset="main", split="test", cache_dir=cache_dir, stop=420),
    ])
    max_conversations = cfg.get("max_conversations")
    if max_conversations is not None:
        train_mixture = _Truncated(train_mixture, max_conversations)
        val_mixture = _Truncated(val_mixture, max_conversations)  # _Truncated itself clamps to len(val_mixture)
    return train_mixture, val_mixture


def _prepare_sft(cfg, ctx, sequence_len):
    """kind="sft": builds the conversation mixture (see _build_sft_mixtures), renders each
    conversation to (ids, loss mask) via the tokenizer, and packs the result with BestFitPadPacker.
    Returns (dataset_name, dataset_dir, manifest)."""
    manager = ctx.data_manager
    dataset_name = cfg.get("dataset") or default_dataset_name("sft", sequence_len, ctx.tokenizer)
    dataset_dir = prepared_dir(dataset_name)
    store = FileSystemDatasetStore(dataset_dir)

    train_mixture, val_mixture = _build_sft_mixtures(cfg, ctx)
    print0(f"Preparing SFT dataset {dataset_name!r}: {len(train_mixture):,} train conversations, {len(val_mixture):,} val -> {dataset_dir}")

    max_tokens = cfg.get("max_tokens_per_conversation", DEFAULT_MAX_TOKENS_PER_CONVERSATION)
    bos_id = ctx.tokenizer.get_bos_token_id()
    sources = {
        "train": TaskMixtureTokenSource(train_mixture, ctx.tokenizer, "train", max_tokens=max_tokens),
        "val": TaskMixtureTokenSource(val_mixture, ctx.tokenizer, "val", max_tokens=max_tokens),
    }
    packer = BestFitPadPacker(bos_token_id=bos_id, padding_id=cfg.get("sft_padding_id"), buffer_size=cfg.get("buffer_size", 1000))
    t0 = time.time()
    manifest = manager.prepare(
        store, sources=sources, tokenizer=ctx.tokenizer, sequence_len=sequence_len,
        sequences_per_volume=cfg.get("sequences_per_volume", 16384), packer=packer,
        num_threads=cfg.get("tokenizer_threads", os.cpu_count()),
    )
    print0(f"Prepared in {time.time() - t0:.1f}s")
    return dataset_name, dataset_dir, manifest


def run(cfg: dict, ctx) -> dict:
    """Runs one prepare step: cfg is a resolved job-file step (see docs/job-file.md for every
    key), ctx the shared Context for this job run. Returns {"op": "prepare", "kind", "dataset",
    "dataset_dir", "splits": {"train"|"val": {"num_sequences", "num_tokens"}}}."""
    assert "kind" in cfg, "prepare: 'kind' is required ('base' or 'sft')"
    kind = cfg["kind"]
    assert kind in ("base", "sft"), f"prepare: kind must be 'base' or 'sft', got {kind!r}"
    assert "sequence_len" in cfg, "prepare: 'sequence_len' is required (put it in \"defaults\" to share it with a matching train step)"
    sequence_len = cfg["sequence_len"]
    if kind == "base":
        dataset_name, dataset_dir, manifest = _prepare_base(cfg, ctx, sequence_len)
    else:
        dataset_name, dataset_dir, manifest = _prepare_sft(cfg, ctx, sequence_len)
    return {
        "op": "prepare", "kind": kind, "dataset": dataset_name, "dataset_dir": dataset_dir,
        "splits": {split: {"num_sequences": s["num_sequences"], "num_tokens": s["num_tokens"]}
                   for split, s in manifest["splits"].items()},
    }
