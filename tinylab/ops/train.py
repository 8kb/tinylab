"""
The `train` op: one loop for both kind="base" (pretrain from scratch) and kind="sft" (finetune a
base checkpoint on conversation data). Ported from nanochat's scripts/base_train.py +
scripts/chat_sft.py -- see docs/architecture.md for what's deliberately dropped relative to that,
and docs/job-file.md for every cfg key this module reads.

Carries no derivation rule of its own: every horizon/batch-size/LR-scale number nanochat's
scaling-law math (target_flops, target_param_data_ratio, the muP batch-size/weight-decay
corrections) used to compute is now a required job-file key instead -- see AGENTS.md's "a config
tree carries only concrete, already-decided values, never a derivation rule" invariant, which now
covers the whole job file. Compute the removed numbers with nanochat's own
`scripts/model_info.py --target-flops=... --json` and paste the result in.
"""
import os
import time

from datacore import DatasetMismatch, FileSystemDatasetStore
from modelcore import OptimizerHparams
from modelcore.kernels.flash_attn import build_doc_args
from modelcore.optim.schedules import lr_multiplier, muon_momentum

from tinylab import checkpoints, modelconfig
from tinylab.ops import prepare
from tinylab.runtime import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, print0

# Keys shared by both kinds, plus each kind's own -- see accepted_keys() below, which is kind-
# aware so e.g. "source_tag" on a kind="base" step is caught as an error rather than silently
# accepted and ignored.
_COMMON_KEYS = {
    "kind", "dataset", "output_tag", "num_iterations", "device_batch_size", "total_batch_size",
    "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr", "weight_decay",
    "warmup_steps", "warmdown_ratio", "final_lr_frac", "muon_momentum_warmup_steps",
    "eval_every", "eval_tokens",
    "fp8", "fp8_recipe", "fp8_eval",
    "doc_masking", "doc_masking_max_docs_per_row",
    "adapter_lr", "adapter_scalar_lr",
}
_BASE_KEYS = set()
_SFT_KEYS = {"source", "source_tag", "source_step"}


def accepted_keys(cfg: dict) -> set:
    kind = cfg.get("kind")
    kind_keys = _BASE_KEYS if kind == "base" else _SFT_KEYS if kind == "sft" else (_BASE_KEYS | _SFT_KEYS)
    return _COMMON_KEYS | kind_keys


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


def _lr_schedule(num_iterations, warmup_steps, warmdown_ratio, final_lr_frac, momentum_warmup_steps):
    """Builds the two per-step schedule functions a training loop needs, from
    modelcore.optim.schedules.lr_multiplier/muon_momentum (this module's own copies were ported
    from nanochat's scripts/base_train.py, which now shares the same source).

    num_iterations: total training steps (an explicit cfg["num_iterations"] for kind="base", or
        the one-epoch derivation for kind="sft").
    warmup_steps: linear LR warmup length, in steps, before holding at the full LR multiplier 1.0.
    warmdown_ratio: fraction of num_iterations spent linearly decaying the LR multiplier from 1.0
        down to final_lr_frac, at the end of the run.
    final_lr_frac: the LR multiplier warmdown decays to (not zero -- a small residual LR at the
        very end of training).
    momentum_warmup_steps: how many early steps Muon's own momentum ramps over before holding at
        0.97 -- a literal cfg["muon_momentum_warmup_steps"], modelcore.optim.schedules.
        muon_momentum's own default (400) unless the job file overrides it. No longer derived from
        num_iterations (that was `min(400, num_iterations // 3)`, an unstated heuristic).

    Returns (get_lr_multiplier, get_muon_momentum): both take a 0-indexed step and return a float
    -- get_lr_multiplier scales every param group's base LR; get_muon_momentum sets Muon's own
    momentum directly (not a multiplier) for the "muon" param groups only.
    """
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
    assert "total_batch_size" in cfg, (
        "train: 'total_batch_size' is required -- tinylab derives no scaling law of its own; "
        "compute it with nanochat's scripts/model_info.py --json and paste the number in."
    )
    assert "eval_tokens" in cfg, "train: 'eval_tokens' is required"
    if kind == "base":
        assert "num_iterations" in cfg, (
            "train: 'num_iterations' is required for kind='base' -- tinylab derives no training "
            "horizon of its own; compute it with nanochat's scripts/model_info.py --target-flops=... "
            "--json (training_plan.num_iterations) and paste the number in."
        )
    assert "world_size" in cfg, (
        "train: 'world_size' is required -- the job file fixes the GPU count for a run; edit it "
        "if the launch's GPU configuration changes."
    )

    manager = ctx.model_manager
    tokenizer = ctx.tokenizer
    vocab_size = tokenizer.get_vocab_size()
    device = ctx.device
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = ctx.device_info[:4]
    assert cfg["world_size"] == ddp_world_size, (
        f"train: job file declares world_size={cfg['world_size']}, but this run was launched "
        f"with {ddp_world_size} rank(s) -- edit the job file's \"world_size\" to match the actual "
        f"launch (e.g. torchrun --nproc_per_node)."
    )
    tokenizer_fingerprint = tokenizer.fingerprint()
    print0(f"[{cfg['name']}] compute dtype: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    dataset_name, dataset, token_bytes = _open_dataset(cfg, ctx, kind, sequence_len)
    print0(f"Dataset: {dataset_name} ({dataset.num_sequences('train'):,} train / {dataset.num_sequences('val'):,} val sequences)")

    weight_decay = cfg.get("weight_decay", 0.28 if kind == "base" else 0.0)
    device_batch_size = cfg.get("device_batch_size", 4)
    total_batch_size = cfg["total_batch_size"]

    if kind == "base":
        assert "model_config" in cfg, (
            'train (kind=base): "model_config" is required -- a path to a materialized '
            "ModelConfig tree (see nanochat's scripts/model_info.py --dump-config)."
        )
        real_config = modelconfig.load_model_config(cfg["model_config"], sequence_len=sequence_len, vocab_size=vocab_size)
        model_stats = manager.stats(real_config)
        print0(f"Model: {model_stats.n_layer} layers, {model_stats.num_params:,} params ({model_stats.num_scaling_params:,} scaling)")
        num_iterations = cfg["num_iterations"]

        model = manager.create_model(real_config, device=device, seed=42)
        base_model_tag = None
        base_model_step = None
    else:
        assert "source_tag" in cfg, "train: kind='sft' requires 'source_tag' (the tag of a prior 'base' train step)"
        source = cfg.get("source", "base")
        # An sft step's "model_config", if given, is a config-override request (e.g. to attach
        # LoRA/DoRA adapters to an already-trained base) -- not a from-scratch build like kind=base
        # uses. Omitting it (the common case) loads the checkpoint's own stored config unchanged.
        config_override = None
        if "model_config" in cfg:
            config_override = modelconfig.load_model_config(cfg["model_config"], sequence_len=sequence_len, vocab_size=vocab_size)
        model, _source_tokenizer, meta = checkpoints.load_model(
            source, device, phase="train", model_tag=cfg["source_tag"], step=cfg.get("source_step"),
            config_override=config_override,
        )
        real_config = model.config
        num_iterations = cfg.get("num_iterations")
        if num_iterations is None:
            # One epoch over the training split, in optimizer steps (not micro-batches --
            # total_batch_size already accounts for grad-accum and world_size).
            num_iterations = max(1, dataset.num_sequences("train") * sequence_len // total_batch_size)
        base_model_tag = meta.get("model_tag")
        base_model_step = meta.get("step")

    if cfg.get("fp8"):
        fp8_report = manager.enable_fp8(model, recipe=cfg.get("fp8_recipe", "tensorwise"))
        print0(f"[{cfg['name']}] fp8: {fp8_report.num_converted}/{fp8_report.num_linear} Linear converted")
    fp8_eval = cfg.get("fp8_eval", True)

    optimizer_hparams_kwargs = dict(
        unembedding_lr=cfg.get("unembedding_lr", 0.008 if kind == "base" else 0.004),
        embedding_lr=cfg.get("embedding_lr", 0.3),
        scalar_lr=cfg.get("scalar_lr", 0.5),
        matrix_lr=cfg.get("matrix_lr", 0.02),
        weight_decay=weight_decay,
    )
    if "adapter_lr" in cfg:
        optimizer_hparams_kwargs["adapter_lr"] = cfg["adapter_lr"]
    if "adapter_scalar_lr" in cfg:
        optimizer_hparams_kwargs["adapter_scalar_lr"] = cfg["adapter_scalar_lr"]
    optimizer = manager.create_optimizer(model, OptimizerHparams(**optimizer_hparams_kwargs))

    tokens_per_fwdbwd = device_batch_size * sequence_len * ddp_world_size
    assert total_batch_size % tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {tokens_per_fwdbwd}"
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    eval_every = cfg.get("eval_every", 50)
    eval_tokens = cfg["eval_tokens"]

    # doc_args is built here, outside the compiled model, and passed in as plain data -- never
    # inside a compiled region (modelcore's invariant -- see modelcore/AGENTS.md's "Intra-document
    # masking's doc_args must be built outside torch.compile"; tinylab's own loop is eager anyway,
    # but the rule is about where the call sits, not whether this particular loop compiles).
    bos_token_id = tokenizer.get_bos_token_id() if cfg.get("doc_masking") else None
    padding_id = dataset.info.padding_id if cfg.get("doc_masking") else None
    doc_masking_max_docs_per_row = cfg.get("doc_masking_max_docs_per_row")

    def make_doc_args(x):
        if bos_token_id is None:
            return None
        max_docs = doc_masking_max_docs_per_row * x.size(0) if doc_masking_max_docs_per_row is not None else None
        return build_doc_args(x, bos_token_id, padding_id=padding_id, max_docs=max_docs)

    get_lr_multiplier, get_muon_momentum = _lr_schedule(
        num_iterations, cfg.get("warmup_steps", 5 if kind == "base" else 0),
        cfg.get("warmdown_ratio", 0.65 if kind == "base" else 0.5), cfg.get("final_lr_frac", 0.05 if kind == "base" else 0.0),
        cfg.get("muon_momentum_warmup_steps", 400),
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
            eval_kwargs = dict(bos_token_id=bos_token_id, doc_masking_max_docs_per_row=doc_masking_max_docs_per_row, padding_id=padding_id)
            if cfg.get("fp8") and not fp8_eval:
                with manager.fp8_disabled(model):
                    val_bpb = manager.evaluate_bpb(model, val_loader, eval_steps, token_bytes, **eval_kwargs)
            else:
                val_bpb = manager.evaluate_bpb(model, val_loader, eval_steps, token_bytes, **eval_kwargs)
            print0(f"[{cfg['name']}] step {step:05d} | val bpb: {val_bpb:.6f}")
            model.train()
        if last_step:
            break
        step_loss = 0.0  # grad-accum-averaged, not the last micro-batch's raw loss
        for _ in range(grad_accum_steps):
            doc_args = make_doc_args(x)
            loss = model(x, y, doc_args=doc_args)
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
        # cfg's own "model_config" is excluded from user_config: it's the *request* (a path,
        # already resolved into the "model_config" key above -- a different dict, same name by
        # coincidence), and would otherwise duplicate that key's content under a different,
        # unresolved shape.
        "user_config": {k: v for k, v in cfg.items() if k != "model_config"},
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
