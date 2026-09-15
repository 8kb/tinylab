"""
The `train` op: one loop for both kind="base" (pretrain from scratch) and kind="sft" (finetune a
base checkpoint on conversation data). Ported from nanochat's scripts/base_train.py +
scripts/chat_sft.py + nanochat/scaling.py -- see docs/architecture.md for what's deliberately
dropped relative to that, and docs/job-file.md for every cfg key this module reads.
"""
import os
import time

from datacore import DatasetMismatch, FileSystemDatasetStore
from modelcore import OptimizerHparams, derive_training_plan
from modelcore.optim.schedules import lr_multiplier, muon_momentum
from modelcore.scaling import B_REF  # noqa: F401 -- re-exported for existing call sites

from tinylab import checkpoints, presets
from tinylab.ops import prepare
from tinylab.runtime import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, print0

# Keys shared by both kinds, plus each kind's own -- see accepted_keys() below, which is kind-
# aware so e.g. "target_flops" on a kind="sft" step (horizon derivation is a kind="base"-only
# concept) is caught as an error rather than silently accepted and ignored.
_COMMON_KEYS = {
    "kind", "dataset", "output_tag", "num_iterations", "device_batch_size", "total_batch_size",
    "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr", "weight_decay",
    "warmup_steps", "warmdown_ratio", "final_lr_frac", "eval_every", "eval_tokens",
}
_BASE_KEYS = {"target_flops", "target_param_data_ratio"}
_SFT_KEYS = {"source", "source_tag", "source_step"}


def accepted_keys(cfg: dict) -> set:
    kind = cfg.get("kind")
    kind_keys = _BASE_KEYS if kind == "base" else _SFT_KEYS if kind == "sft" else (_BASE_KEYS | _SFT_KEYS)
    return _COMMON_KEYS | kind_keys


# -----------------------------------------------------------------------------
# Scaling-law horizon derivation (kind="base" only) is modelcore.derive_training_plan -- this
# module's own copy was ported whole from nanochat/scaling.py; both now come from modelcore, which
# has no host dependencies at all. See modelcore's own docs/architecture.md.
# -----------------------------------------------------------------------------

def _open_dataset(cfg, ctx, kind, sequence_len):
    """Opens the dataset this step trains on, raising a clear error (not letting datacore's own
    FileNotFoundError propagate unexplained) if it hasn't been prepared yet, or was prepared with a
    different sequence_len or tokenizer than this step is using. Returns (dataset_name, dataset,
    token_bytes) -- token_bytes is the per-token-id byte-length table evaluate_bpb needs."""
    dataset_name = cfg.get("dataset") or prepare.default_dataset_name(kind, sequence_len, ctx.tokenizer)
    dataset_dir = prepare.prepared_dir(dataset_name)
    store = FileSystemDatasetStore(dataset_dir)
    try:
        # expect_sequence_len/expect_fingerprint: datacore's own (opt-in) comparison now, not a
        # hand-written check -- see datacore/AGENTS.md's "tokenizer fingerprint" invariant for why
        # datacore itself still never decides *to* compare.
        dataset = ctx.data_manager.open(store, expect_sequence_len=sequence_len, expect_fingerprint=ctx.tokenizer.fingerprint())
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


def _lr_schedule(num_iterations, warmup_steps, warmdown_ratio, final_lr_frac):
    """Builds the two per-step schedule functions a training loop needs, from
    modelcore.optim.schedules.lr_multiplier/muon_momentum (this module's own copies were ported
    from nanochat's scripts/base_train.py, which now shares the same source).

    num_iterations: total training steps (the horizon derive_training_plan settled on, or an
        explicit cfg["num_iterations"] for kind="sft").
    warmup_steps: linear LR warmup length, in steps, before holding at the full LR multiplier 1.0.
    warmdown_ratio: fraction of num_iterations spent linearly decaying the LR multiplier from 1.0
        down to final_lr_frac, at the end of the run.
    final_lr_frac: the LR multiplier warmdown decays to (not zero -- a small residual LR at the
        very end of training).

    Returns (get_lr_multiplier, get_muon_momentum): both take a 0-indexed step and return a float
    -- get_lr_multiplier scales every param group's base LR; get_muon_momentum sets Muon's own
    momentum directly (not a multiplier) for the "muon" param groups only.
    """
    # The Muon-momentum warmup ramps over this many early steps before holding at 0.97. Capped at
    # num_iterations // 3 (not modelcore's own default of 400) so a short run -- every job shipped
    # in this repo trains well under 400 steps -- doesn't spend its *entire* horizon inside the
    # warmup branch, which would make the warmdown branch below unreachable.
    momentum_warmup_steps = max(1, min(400, num_iterations // 3))

    def get_lr_multiplier(it):
        return lr_multiplier(it, num_iterations, warmup_steps, warmdown_ratio, final_lr_frac)

    def get_muon_momentum(it):
        return muon_momentum(it, num_iterations, warmdown_ratio, momentum_warmup_steps=momentum_warmup_steps)

    return get_lr_multiplier, get_muon_momentum


def run(cfg: dict, ctx) -> dict:
    """Runs one train step: cfg is a resolved job-file step (see docs/job-file.md for every key),
    ctx the shared Context for this job run. Returns {"op": "train", "kind", "output_tag", "step",
    "val_bpb", "total_training_time"}."""
    assert "kind" in cfg, "train: 'kind' is required ('base' or 'sft')"
    kind = cfg["kind"]
    assert kind in ("base", "sft"), f"train: kind must be 'base' or 'sft', got {kind!r}"
    assert "sequence_len" in cfg, "train: 'sequence_len' is required (put it in \"defaults\")"
    sequence_len = cfg["sequence_len"]

    manager = ctx.model_manager
    tokenizer = ctx.tokenizer
    vocab_size = tokenizer.get_vocab_size()
    device = ctx.device
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = ctx.device_info[:4]
    tokenizer_fingerprint = tokenizer.fingerprint()
    print0(f"[{cfg['name']}] compute dtype: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    dataset_name, dataset, token_bytes = _open_dataset(cfg, ctx, kind, sequence_len)
    print0(f"Dataset: {dataset_name} ({dataset.num_sequences('train'):,} train / {dataset.num_sequences('val'):,} val sequences)")

    weight_decay = cfg.get("weight_decay", 0.28 if kind == "base" else 0.0)
    device_batch_size = cfg.get("device_batch_size", 4)

    if kind == "base":
        model_cfg = cfg.get("model", {})
        depth = model_cfg.get("depth", 6)
        selector = model_cfg["config"] if "config" in model_cfg else model_cfg.get("preset", "gpt")
        real_config = presets.resolve_model_config(
            selector, depth, aspect_ratio=model_cfg.get("aspect_ratio", 64), head_dim=model_cfg.get("head_dim", 128),
            max_seq_len=sequence_len, vocab_size=vocab_size, window_pattern=model_cfg.get("window_pattern"),
            arch_opts=model_cfg.get("arch_opts"),
        )
        model_stats = manager.stats(real_config)
        d_ref_scaling_params = model_cfg.get("d_ref_scaling_params")
        if d_ref_scaling_params is None:
            d_ref_scaling_params = manager.stats(presets.resolve_reference_config(real_config, 12)).num_scaling_params

        plan = derive_training_plan(
            num_scaling_params=model_stats.num_scaling_params, d_ref_scaling_params=d_ref_scaling_params,
            num_flops_per_token=model_stats.flops_per_token,
            target_param_data_ratio=cfg.get("target_param_data_ratio", 12), target_flops=cfg.get("target_flops", -1.0),
            num_iterations=cfg.get("num_iterations", -1), total_batch_size=cfg.get("total_batch_size", -1),
            weight_decay=weight_decay,
        )
        num_iterations = plan.num_iterations
        total_batch_size = plan.total_batch_size
        weight_decay = plan.weight_decay_scaled
        batch_lr_scale = plan.batch_lr_scale
        print0(f"Horizon: {num_iterations:,} iterations, {plan.total_tokens:,} tokens ({plan.horizon_source})")

        model = manager.create_model(real_config, device=device, seed=42)
        base_model_tag = None
        base_model_step = None
    else:
        assert "source_tag" in cfg, "train: kind='sft' requires 'source_tag' (the tag of a prior 'base' train step)"
        source = cfg.get("source", "base")
        model, _source_tokenizer, meta = checkpoints.load_model(source, device, phase="train", model_tag=cfg["source_tag"], step=cfg.get("source_step"))
        real_config = model.config
        num_iterations = cfg.get("num_iterations")
        if num_iterations is None:
            num_iterations = max(1, dataset.num_sequences("train") // (device_batch_size * ddp_world_size))
        total_batch_size = cfg.get("total_batch_size", device_batch_size * sequence_len * ddp_world_size)
        batch_lr_scale = 1.0
        base_model_tag = meta.get("model_tag")
        base_model_step = meta.get("step")

    optimizer_hparams = OptimizerHparams(
        unembedding_lr=cfg.get("unembedding_lr", 0.008 if kind == "base" else 0.004) * batch_lr_scale,
        embedding_lr=cfg.get("embedding_lr", 0.3) * batch_lr_scale,
        scalar_lr=cfg.get("scalar_lr", 0.5) * batch_lr_scale,
        matrix_lr=cfg.get("matrix_lr", 0.02) * batch_lr_scale,
        weight_decay=weight_decay,
    )
    optimizer = manager.create_optimizer(model, optimizer_hparams)

    tokens_per_fwdbwd = device_batch_size * sequence_len * ddp_world_size
    assert total_batch_size % tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {tokens_per_fwdbwd}"
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    eval_every = cfg.get("eval_every", 50)
    eval_tokens = cfg.get("eval_tokens", device_batch_size * sequence_len * ddp_world_size * 4)
    get_lr_multiplier, get_muon_momentum = _lr_schedule(
        num_iterations, cfg.get("warmup_steps", 5 if kind == "base" else 0),
        cfg.get("warmdown_ratio", 0.65 if kind == "base" else 0.5), cfg.get("final_lr_frac", 0.05 if kind == "base" else 0.0),
    )

    train_loader = ctx.data_manager.batches(dataset, "train", device_batch_size, device=device, rank=ddp_rank, world_size=ddp_world_size, infinite=True)
    x, y, dataloader_state = next(train_loader)

    val_bpb = None
    t_start = time.time()
    for step in range(num_iterations + 1):
        last_step = step == num_iterations
        if eval_every > 0 and (last_step or step % eval_every == 0):
            model.eval()
            val_loader = ctx.data_manager.batches(dataset, "val", device_batch_size, device=device, rank=ddp_rank, world_size=ddp_world_size, infinite=True)
            eval_steps = max(1, eval_tokens // tokens_per_fwdbwd)
            val_bpb = manager.evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print0(f"[{cfg['name']}] step {step:05d} | val bpb: {val_bpb:.6f}")
            model.train()
        if last_step:
            break
        step_loss = 0.0  # grad-accum-averaged, not the last micro-batch's raw loss
        for _ in range(grad_accum_steps):
            loss = model(x, y)
            step_loss += loss.detach().item() / grad_accum_steps
            (loss / grad_accum_steps).backward()
            x, y, dataloader_state = next(train_loader)
        lrm = get_lr_multiplier(step)
        muon_momentum_value = get_muon_momentum(step)
        manager.apply_schedule(optimizer, lr_mult=lrm, muon_momentum=muon_momentum_value)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print0(f"[{cfg['name']}] step {step:05d}/{num_iterations} | loss {step_loss:.4f}")
    total_training_time = time.time() - t_start

    output_tag = cfg.get("output_tag", cfg["name"])
    checkpoint_dir = os.path.join(checkpoints.get_base_dir(), checkpoints.CHECKPOINT_DIRS[kind], output_tag)
    meta_data = {
        "step": num_iterations, "val_bpb": val_bpb, "tokenizer_fingerprint": tokenizer_fingerprint,
        "model_config": manager.config_to_dict(real_config),
        # "model" is excluded: it's the preset/depth *request* (train.py already resolved it into
        # model_config above), and would otherwise duplicate model_config's own content under a
        # different, unresolved shape.
        "user_config": {k: v for k, v in cfg.items() if k != "model"},
        "device_batch_size": device_batch_size, "max_seq_len": sequence_len, "total_batch_size": total_batch_size,
        "dataloader_state_dict": dataloader_state, "total_training_time": total_training_time,
    }
    if kind == "sft":
        meta_data["base_model_tag"] = base_model_tag
        meta_data["base_model_step"] = base_model_step
    checkpoints.save_checkpoint(checkpoint_dir, num_iterations, model.state_dict(), optimizer.state_dict(), meta_data, rank=ddp_rank)
    print0(f"[{cfg['name']}] saved checkpoint: {checkpoint_dir} (step {num_iterations})")

    return {
        "op": "train", "kind": kind, "output_tag": output_tag, "step": num_iterations,
        "val_bpb": val_bpb, "total_training_time": total_training_time,
    }
