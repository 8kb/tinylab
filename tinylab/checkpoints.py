"""
Checkpoint naming policy: which directory, which step, which tag, plus meta.json's extra fields
(step, val_bpb, tokenizer_fingerprint, model_config, user_config, device_batch_size, max_seq_len,
total_batch_size, dataloader_state_dict, total_training_time -- see tinylab/ops/train.py, which
writes them). Ported and trimmed from nanochat's nanochat/checkpoint_manager.py (llmllab/nanochat)
-- the actual model/optimizer artifact format belongs to modelcore (modelcore.manager.ModelManager),
this module hands it a modelcore.store.FileSystemStore over the right directory+step.

Dropped relative to nanochat's version: LegacyCheckpointStore (tinylab has no pre-modelcore
checkpoints to migrate), arch-aware tag naming (tinylab has only one preset family, "gpt"),
resume support (load_checkpoint/load_optimizer_state -- an already out-of-scope feature), and the
config_override load-time hook (served only PEFT/adapters, also out of scope).
"""
import json
import os
import re

from modelcore.store import FileSystemStore

from tinylab.runtime import get_base_dir
from tinylab.tokenizer import get_tokenizer

CHECKPOINT_DIRS = {"base": "base_checkpoints", "sft": "chatsft_checkpoints"}


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    store = FileSystemStore(checkpoint_dir, step)
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        store.write_model_state(model_data)
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        existing = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.update({k: v for k, v in meta_data.items() if k != "model_config"})
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
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
    pattern = re.compile(r"model_(\d+)\.pt$")
    steps = []
    for filename in os.listdir(checkpoint_dir):
        match = pattern.search(filename)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(steps)


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
