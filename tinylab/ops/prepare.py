"""
The `prepare` op: tokenize + pack a corpus into a datacore dataset. Ported from our nanochat fork's
scripts/data_prep.py -- kind="base" downloads the shards of a corpus (data.CORPORA, "corpus" key; if
not already present) and packs them with BestFitCropPacker; kind="sft" builds the conversation
mixture ("mixture"; SmolTalk + MMLU + GSM8K by default) and packs it with BestFitPadPacker. See docs/job-file.md for every cfg key this module reads.
"""
import os
import time

from datacore import (
    BestFitCropPacker, BestFitPadPacker, DatasetMismatch, ExampleMixture, ExampleTokenSource, FileSystemDatasetStore,
    ParquetDirectorySource,
)

from tinylab import data
from tinylab.runtime import get_base_dir, print0
from tinylab.tokenizer import bucket_entity

# Keys shared by both kinds, plus each kind's own -- see accepted_keys() below, which is kind-
# aware so e.g. "mmlu_epochs" on a kind="base" step is caught as an error rather than silently
# accepted and ignored. "tokenizer" is already in ops.COMMON_KEYS (every op accepts it -- it's the
# step's own tokenizer selection, see Context.tokenizer_for), so it isn't repeated here.
_COMMON_KEYS = {"kind", "dataset", "sequences_per_volume", "buffer_size", "tokenizer_threads", "push"}
_BASE_KEYS = {"shards", "corpus"}
_SFT_KEYS = {"max_conversations", "mixture", "mmlu_epochs", "gsm8k_epochs", "max_tokens_per_conversation", "sft_padding_id"}

# The SFT conversation mixture: entries are {"task", "epochs", "val_cap"} -- `task` is "smoltalk" or a
# benchcore chat task (see data.build_task); `epochs` passes of its train split go into the training
# data (default 1); `val_cap` caps its test split in the validation mixture (default: uncapped).
DEFAULT_MIXTURE = [
    {"task": "smoltalk"},
    {"task": "mmlu", "epochs": 3, "val_cap": 5200},
    {"task": "gsm8k", "epochs": 4, "val_cap": 420},
]
_MIXTURE_ENTRY_KEYS = {"task", "epochs", "val_cap"}


def accepted_keys(cfg: dict) -> set:
    kind = cfg.get("kind")
    kind_keys = _BASE_KEYS if kind == "base" else _SFT_KEYS if kind == "sft" else (_BASE_KEYS | _SFT_KEYS)
    return _COMMON_KEYS | kind_keys


def prepared_dir(name: str) -> str:
    return os.path.join(get_base_dir(), "prepared", name)


def default_dataset_name(kind: str, sequence_len: int, tokenizer, corpus: str = data.DEFAULT_CORPUS) -> str:
    stem = corpus if kind == "base" else "sft"
    return f"{stem}_t{sequence_len}_{tokenizer.fingerprint()}"


def _make_store(cfg, ctx, dataset_name, dataset_dir):
    """FileSystemDatasetStore, or -- with a "remote" and "push" (default true) -- one that uploads
    each shard as it's written and the manifest last."""
    if ctx.uploader is None or not cfg.get("push", True):
        return FileSystemDatasetStore(dataset_dir)
    from tinylab.remote_stores import UploadingDatasetStore
    spec = cfg.get("tokenizer")
    entity = bucket_entity(spec, ctx.tokenizer_name_for(spec))
    inputs = {} if entity is None else {"tokenizer": entity}
    return UploadingDatasetStore(dataset_dir, name=dataset_name, uploader=ctx.uploader, producer=ctx.producer(cfg["name"]),
                                 inputs=inputs, motivation=cfg.get("_comment", ""))


def open_prepared(cfg, ctx, kind, sequence_len, tokenizer):
    """Opens the dataset a train/bench step reads, raising a clear error (not letting datacore's own
    FileNotFoundError propagate unexplained) if it hasn't been prepared yet, or was prepared with a
    different sequence_len or tokenizer than this step is using. Returns (dataset_name, dataset,
    token_bytes) -- token_bytes is the per-token-id byte-length table evaluate_bpb needs."""
    dataset_name = cfg.get("dataset") or default_dataset_name(kind, sequence_len, tokenizer, cfg.get("corpus", data.DEFAULT_CORPUS))
    dataset_dir = prepared_dir(dataset_name)
    if ctx.remote is not None:
        # Missing shards stream in from the bucket while training runs on the ones already local.
        from tinylab.remote_stores import PrefetchingDatasetStore
        store = PrefetchingDatasetStore(dataset_dir, name=dataset_name, remote=ctx.remote)
    else:
        store = FileSystemDatasetStore(dataset_dir)
    try:
        # datacore compares only on request (expect_*): the host decides what must match.
        dataset = ctx.data_manager.open(store, expect_sequence_len=sequence_len, expect_fingerprint=tokenizer.fingerprint())
    except FileNotFoundError:
        raise SystemExit(
            f"No prepared dataset found at {dataset_dir}. Run a \"prepare\" step first, with "
            f"kind={kind!r} and matching \"sequence_len\"."
        )
    except DatasetMismatch as e:
        if e.reason == "sequence_len":
            raise SystemExit(f"Dataset {dataset_name!r} was prepared at sequence_len={e.actual}, but this step uses {sequence_len}.")
        raise SystemExit(
            f"Dataset {dataset_name!r} was prepared against tokenizer fingerprint "
            f"{e.actual}, but the local tokenizer's fingerprint is "
            f"{e.expected} -- refusing to train on it."
        )
    return dataset_name, dataset, ctx.data_manager.token_bytes(dataset)


def _prepare_base(cfg, ctx, sequence_len, tokenizer):
    """kind="base": ensures cfg["shards"] train shards of cfg["corpus"] (plus the fixed val shard) are on
    disk, downloading whatever's missing, then packs exactly those paths -- never however many
    shard files a larger previous run happened to leave around. Returns (dataset_name, dataset_dir,
    manifest)."""
    manager = ctx.data_manager
    corpus = cfg.get("corpus", data.DEFAULT_CORPUS)
    dataset_name = cfg.get("dataset") or default_dataset_name("base", sequence_len, tokenizer, corpus)
    dataset_dir = prepared_dir(dataset_name)
    store = _make_store(cfg, ctx, dataset_name, dataset_dir)

    shards = cfg.get("shards", 8)
    train_paths, val_paths = data.corpus_train_val_paths(shards, corpus)
    if any(not os.path.exists(p) for p in train_paths + val_paths):
        data.download_corpus_shards(shards, corpus, log=print0)
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


def resolve_mixture(cfg):
    """The step's mixture as a validated list of entries: "mixture" if given, else DEFAULT_MIXTURE
    with the older "mmlu_epochs"/"gsm8k_epochs" keys (kept as aliases) applied. Giving both forms is
    an error."""
    legacy = {name: cfg[key] for name, key in (("mmlu", "mmlu_epochs"), ("gsm8k", "gsm8k_epochs")) if key in cfg}
    if "mixture" in cfg:
        assert not legacy, 'prepare: "mixture" replaces "mmlu_epochs"/"gsm8k_epochs" -- give one or the other'
        mixture = cfg["mixture"]
    else:
        mixture = [dict(entry, epochs=legacy.get(entry["task"], entry.get("epochs", 1))) for entry in DEFAULT_MIXTURE]
    assert isinstance(mixture, list) and mixture, 'prepare: "mixture" must be a non-empty list of {"task": ...} entries'
    for entry in mixture:
        assert isinstance(entry, dict) and "task" in entry, f'prepare: mixture entry {entry!r} needs a "task"'
        unknown = set(entry) - _MIXTURE_ENTRY_KEYS
        assert not unknown, f"prepare: mixture entry {entry!r} has unknown key(s) {sorted(unknown)}; accepted: {sorted(_MIXTURE_ENTRY_KEYS)}"
        if entry["task"].lower() != "smoltalk":
            data.chat_task_name(entry["task"])  # raises ValueError for an unknown task
        assert entry.get("epochs", 1) >= 1, f"prepare: mixture entry {entry!r}: epochs must be >= 1"
    return mixture


def _build_sft_mixtures(cfg, ctx):
    """Builds (train_mixture, val_mixture) from resolve_mixture(cfg): each entry's train split, `epochs`
    times, for train; its test split (capped at `val_cap`, if given) for val. Each mixture is
    truncated to cfg["max_conversations"] if given, for a fast smoke-sized run."""
    mixture = resolve_mixture(cfg)
    train_tasks = [data.build_task(entry["task"], "train") for entry in mixture for _ in range(entry.get("epochs", 1))]
    val_tasks = [data.build_task(entry["task"], "test", **({"stop": entry["val_cap"]} if "val_cap" in entry else {}))
                 for entry in mixture]
    # max_conversations (smoke tests) caps both mixtures via ExampleMixture's own stop= kwarg --
    # __len__ clamps stop to each mixture's true length itself, see datacore.records.ExampleSet.
    max_conversations = cfg.get("max_conversations")
    return ExampleMixture(train_tasks, stop=max_conversations), ExampleMixture(val_tasks, stop=max_conversations)


def _prepare_sft(cfg, ctx, sequence_len, tokenizer):
    """kind="sft": builds the conversation mixture (see _build_sft_mixtures), renders each
    conversation to (ids, loss mask) via the tokenizer, and packs the result with BestFitPadPacker.
    Returns (dataset_name, dataset_dir, manifest)."""
    manager = ctx.data_manager
    dataset_name = cfg.get("dataset") or default_dataset_name("sft", sequence_len, tokenizer)
    dataset_dir = prepared_dir(dataset_name)
    store = _make_store(cfg, ctx, dataset_name, dataset_dir)

    train_mixture, val_mixture = _build_sft_mixtures(cfg, ctx)
    print0(f"Preparing SFT dataset {dataset_name!r}: {len(train_mixture):,} train conversations, {len(val_mixture):,} val -> {dataset_dir}")

    max_tokens = cfg.get("max_tokens_per_conversation", sequence_len)
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
