"""
Checkpoint naming policy: which directory, which step, which tag. Ported from nanochat's
nanochat/checkpoint_manager.py -- the actual model/optimizer artifact format belongs to modelcore
(modelcore.manager.ModelManager), this module hands it a modelcore.store.FileSystemStore over the
right directory+step. See docs/architecture.md's "On-disk layout" for meta.json's full field list.

There is one checkpoint namespace, <base_dir>/checkpoints/<tag>/, and the tag is the whole address:
arbitrary text, optionally with folders ("gpt-d12-base", "kvcache/d13-chat"). What *kind* of
checkpoint it is (pretrained, fine-tuned, whatever comes next) is the job author's to say in the
tag, not something the path encodes -- an earlier base_checkpoints/ + chatsft_checkpoints/ split
would have needed a new directory for every new kind.
"""
import json
import os
import re

from modelcore.store import FileSystemStore
from modelcore.store import last_step as _last_step

from tinylab.runtime import get_base_dir
from tinylab.tokenizer import get_tokenizer

CHECKPOINTS_DIR = "checkpoints"

# One name: a filename/directory component on its own (a step name, a job-file stem, one segment
# of a tag). Deliberately an allowlist, not a blocklist of what's unsafe -- short and predictable
# enough to type, read in a log line, and embed directly in a filename with no further escaping.
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,16}$")


def validate_name(name, *, label="name"):
    """A bare name: 1-16 characters, only letters/digits/"_"/"-". Used standalone for a step name
    or a job file's own stem, and as the building block validate_tag splits a tag into. Returns
    name."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"{label} must be 1-16 characters from [A-Za-z0-9_-], got {name!r}")
    return name


def validate_tag(tag, *, label="checkpoint tag"):
    """A tag is 1-4 "/"-joined names (see validate_name) -- e.g. "gpt-d12-base" or
    "kvcache/d13-chat". `label` only changes the wording of a raised error, so the same validator
    serves checkpoint tags (output_tag/source_tag/model_tag) and any other "/"-joined path segment
    under <base_dir> that needs the same treatment (e.g. a job's log_dir). Returns tag."""
    if not isinstance(tag, str) or not tag:
        raise ValueError(f"{label} must be a non-empty string, got {tag!r}")
    segments = tag.split("/")
    if not 1 <= len(segments) <= 4:
        raise ValueError(f"{label} must have 1-4 \"/\"-separated segments, got {len(segments)} in {tag!r}")
    for segment in segments:
        validate_name(segment, label=f"{label} segment")
    return tag


def resolve_checkpoint_dir(tag, base_dir=None):
    """<base_dir>/checkpoints/<tag>, with the tag's "/" folders as real subdirectories."""
    validate_tag(tag)
    return os.path.join(base_dir or get_base_dir(), CHECKPOINTS_DIR, *tag.split("/"))


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    """Writes model_data (a state_dict) and meta_data (a plain dict -- see docs/architecture.md's
    "On-disk layout" for the fields tinylab.ops.train populates) to checkpoint_dir at this step,
    merging into any existing meta.json there rather than overwriting it. optimizer_data (also a
    state_dict, or None to skip saving optimizer state) is written per-rank. Only rank 0 writes
    model/meta; every rank writes its own optimizer shard."""
    store = FileSystemStore(checkpoint_dir, step)
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        store.write_model_state(model_data)
        store.update_meta(meta_data)
        if "model_config" in meta_data:
            store.write_config(meta_data["model_config"])
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        store.write_optimizer_state(optimizer_data, rank=rank)


def build_model(checkpoint_dir, step, device, phase, config_override=None, tokenizer_spec=None):
    """Builds a model from a checkpoint. config_override, if given, replaces the checkpoint's own
    stored config (e.g. a hand-edited tree with adapters attached, loaded via
    tinylab.modelconfig.load_model_config) -- see ModelManager.load_model's own docstring for the
    adapter-reconciling load path this enables. Returns (model, tokenizer, meta_data).

    Which tokenizer: `tokenizer_spec` (a job's "tokenizer" value -- see tinylab.tokenizer.
    resolve_tokenizer_dir) if given, else the one the checkpoint's own config says it was trained
    with (its `tokenizer` block's name), else the default. The vocab-size and fingerprint checks
    below then catch a wrong *choice*, not just a wrong base dir -- a fingerprint-identical wrong
    tokenizer is impossible, but a differently-trained one with a matching vocab size is exactly
    what the fingerprint exists to refuse."""
    assert phase in ("train", "eval"), f"Invalid phase: {phase}"
    # A fresh ModelManager is cheap (it just wraps Runtime detection) and this is its only use
    # site -- no reason to hold one at module scope (see ops/__init__.py's "explicitly passed,
    # never a module global" rule, which a checkpoints-module-level manager would have violated).
    from modelcore import ModelManager
    manager = ModelManager()
    store = FileSystemStore(checkpoint_dir, step)
    model = manager.load_model(store, device=device, config=config_override, train=(phase == "train"))
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    if tokenizer_spec is None:
        tokenizer_spec = ((meta_data.get("model_config") or {}).get("tokenizer") or {}).get("name")
    tokenizer = get_tokenizer(tokenizer=tokenizer_spec)
    assert tokenizer.get_vocab_size() == model.config.vocab_size, (
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab "
        f"size {model.config.vocab_size}"
    )
    checkpoint_fingerprint = meta_data.get("tokenizer_fingerprint")
    if checkpoint_fingerprint is not None and checkpoint_fingerprint != tokenizer.fingerprint():
        raise ValueError(
            f"Tokenizer fingerprint mismatch: this checkpoint was trained with a different "
            f"tokenizer than the one loaded here ({checkpoint_fingerprint} != "
            f"{tokenizer.fingerprint()}). Vocab size matches, but token ids mean different "
            f"things -- refusing to load, rather than produce silent garbage."
        )
    return model, tokenizer, meta_data


def find_last_step(checkpoint_dir):
    """The highest step number among checkpoint_dir's model_<step>.pt files. Raises
    FileNotFoundError if there aren't any. Naming mechanics moved to modelcore.store.last_step --
    nanochat carried an identical copy."""
    return _last_step(checkpoint_dir)


def load_for_resume(checkpoint_dir, step, device, rank, manager):
    """Everything tinylab.ops.train needs to continue an interrupted run at `step`: the model
    (loaded with config=None, so it gets exactly the config that was checkpointed -- a resumed run
    never re-resolves "model_config"/"source_tag" for itself, see tinylab.ops.train's own
    docstring), this rank's optimizer state (None if it was never saved for this step/rank -- the
    caller decides whether that's fatal), and the full meta dict (dataloader position, val_bpb,
    world_size the checkpoint was saved at, ...)."""
    store = FileSystemStore(checkpoint_dir, step)
    model = manager.load_model(store, device=device, config=None, train=True)
    return model, store.read_optimizer_state(rank=rank, map_location=device), store.read_meta()


def load_model(model_tag, device, phase, step=None, config_override=None, tokenizer_spec=None):
    """model_tag is required -- tinylab has no auto-discovery, since a job file always names the
    tag it just trained or wants to load. config_override, tokenizer_spec: see build_model."""
    checkpoint_dir = resolve_checkpoint_dir(model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase, config_override=config_override,
                                              tokenizer_spec=tokenizer_spec)
    meta_data["model_tag"] = model_tag
    return model, tokenizer, meta_data


def load_optimizer_state(model_tag, step, device, rank):
    """This rank's optimizer shard from another tag's checkpoint, without re-loading its model --
    what a kind="sft" step's momentum warm-start (tinylab.ops.train) needs from its own
    "source_tag". Mirrors nanochat's checkpoint_manager.load_optimizer_state, minus the
    base/sft/rl directory-name mapping tinylab's flat tag namespace doesn't have -- resolve_
    checkpoint_dir(model_tag) is already the whole address. Returns None if this shard was never
    saved (e.g. an older checkpoint, or optimizer state genuinely absent) -- the caller decides
    whether that's fatal."""
    checkpoint_dir = resolve_checkpoint_dir(model_tag)
    store = FileSystemStore(checkpoint_dir, step)
    return store.read_optimizer_state(rank=rank, map_location=device)
