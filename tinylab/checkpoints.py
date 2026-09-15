"""
Checkpoint naming policy: which directory, which step, which tag. Ported from nanochat's
nanochat/checkpoint_manager.py -- the actual model/optimizer artifact format belongs to modelcore
(modelcore.manager.ModelManager), this module hands it a modelcore.store.FileSystemStore over the
right directory+step. See docs/architecture.md's "On-disk layout" for meta.json's full field list.
"""
import json
import os

from modelcore.store import FileSystemStore
from modelcore.store import last_step as _last_step

from tinylab.runtime import get_base_dir
from tinylab.tokenizer import get_tokenizer

CHECKPOINT_DIRS = {"base": "base_checkpoints", "sft": "chatsft_checkpoints"}


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


def build_model(checkpoint_dir, step, device, phase):
    """Builds a model from a checkpoint. Returns (model, tokenizer, meta_data)."""
    assert phase in ("train", "eval"), f"Invalid phase: {phase}"
    # A fresh ModelManager is cheap (it just wraps Runtime detection) and this is its only use
    # site -- no reason to hold one at module scope (see ops/__init__.py's "explicitly passed,
    # never a module global" rule, which a checkpoints-module-level manager would have violated).
    from modelcore import ModelManager
    manager = ModelManager()
    store = FileSystemStore(checkpoint_dir, step)
    model = manager.load_model(store, device=device, train=(phase == "train"))
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    tokenizer = get_tokenizer()
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


def load_model(source, device, phase, model_tag, step=None):
    """source: 'base' | 'sft'. model_tag is required -- tinylab has no auto-discovery, since a job
    file always names the tag it just trained or wants to load."""
    checkpoints_dir = os.path.join(get_base_dir(), CHECKPOINT_DIRS[source])
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    meta_data["model_tag"] = model_tag
    return model, tokenizer, meta_data
