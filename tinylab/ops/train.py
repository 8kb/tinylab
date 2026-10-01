"""
The `train` op: one loop for both kind="base" (pretrain from scratch) and kind="sft" (finetune a
base checkpoint on conversation data). Ported from our nanochat fork's scripts/base_train.py +
scripts/chat_sft.py -- see docs/architecture.md for what's deliberately dropped relative to that,
and docs/job-file.md for every cfg key this module reads.

Carries no derivation rule of its own: every horizon/batch-size/LR-scale number our nanochat fork's
scaling-law math (target_flops, target_param_data_ratio, the muP batch-size/weight-decay
corrections) used to compute is now a required job-file key instead -- see AGENTS.md's "a config
tree carries only concrete, already-decided values, never a derivation rule" invariant, which now
covers the whole job file. Compute the removed numbers with
`python -m tinylab info <config> --target-flops=... --json` and paste the result in.

Resume (ctx.resume, set by `python -m tinylab ... --resume`, never a job-file key -- see
tinylab.job) is a run-level fact, not a pipeline fact: when it's set, a step first asks
checkpoints.resume_point whether it already has a checkpoint under its own output_tag and, if so,
continues from it instead of starting fresh, regardless of kind. The step comes from the job state
file's record of the last checkpoint whose save *fully* returned (ctx.resume_checkpoint_step); a
bare Context (this op called directly, without job.run_file) falls back to a directory scan, and a
job-driven run with no record but a checkpoint on disk is an error. See "Resume" in the run()
docstring below.
"""
import time

import torch

from modelcore import build_doc_args, lr_multiplier, muon_momentum
from modelcore import runtime as modelcore_runtime

from tinylab import checkpoints, modelconfig
from tinylab import remote as remote_mod
from tinylab.ops import prepare
from tinylab.runtime import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, print0
from tinylab.tokenizer import bucket_entity

# Keys shared by both kinds, plus each kind's own -- see accepted_keys() below, which is kind-
# aware so e.g. "source_tag" on a kind="base" step is caught as an error rather than silently
# accepted and ignored.
_COMMON_KEYS = {
    "kind", "dataset", "output_tag", "num_iterations", "device_batch_size", "total_batch_size",
    "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr", "weight_decay",
    "warmup_steps", "warmdown_ratio", "final_lr_frac", "muon_momentum_warmup_steps",
    "eval_every", "eval_tokens", "save_every", "seed",
    "fp8", "fp8_eval",
    "doc_masking", "doc_masking_max_docs_per_row",
    "adapter_lr", "adapter_scalar_lr", "conv_lr", "ssm_lr",
    "push_model", "push_optim",
}
_BASE_KEYS = {"corpus"}  # only names the default dataset (see prepare.default_dataset_name)
# init_lr_frac/load_optimizer are sft-only, deliberately (as in our nanochat fork), and accepted_keys
# is kind-aware, so either key on a kind="base" step is a startup error instead of a silently-ignored one.
_SFT_KEYS = {"source_tag", "source_step", "init_lr_frac", "load_optimizer"}

# Every default this op applies, in one place (docs/job-file.md documents them; info.py reads the
# base weight decay from here). The optimizer LRs are the same for both kinds: sft uses what base
# pretrained at (our nanochat fork inherited them from the pretrain checkpoint; tinylab has no
# inheritance mechanism, a job file names every value it wants).
_SHARED_DEFAULTS = {
    "device_batch_size": 4, "eval_every": 50, "save_every": 0, "fp8_eval": True,
    "unembedding_lr": 0.008, "embedding_lr": 0.3, "matrix_lr": 0.02, "scalar_lr": 0.5,
    "muon_momentum_warmup_steps": 400, "seed": 42,
}
DEFAULTS = {
    "base": {**_SHARED_DEFAULTS, "weight_decay": 0.28, "warmup_steps": 5, "warmdown_ratio": 0.65,
             "final_lr_frac": 0.05, "init_lr_frac": 1.0},
    "sft": {**_SHARED_DEFAULTS, "weight_decay": 0.0, "warmup_steps": 0, "warmdown_ratio": 0.5,
            "final_lr_frac": 0.0, "init_lr_frac": 0.8, "load_optimizer": True},
}


def accepted_keys(cfg: dict) -> set:
    kind = cfg.get("kind")
    kind_keys = _BASE_KEYS if kind == "base" else _SFT_KEYS if kind == "sft" else (_BASE_KEYS | _SFT_KEYS)
    return _COMMON_KEYS | kind_keys


def _horizon(cfg, dataset, sequence_len, total_batch_size):
    """num_iterations: the step's explicit value (0 is a real value: eval only), else one epoch."""
    num_iterations = cfg.get("num_iterations")
    return _one_epoch(dataset, sequence_len, total_batch_size) if num_iterations is None else num_iterations


def _one_epoch(dataset, sequence_len, total_batch_size):
    """One epoch over the training split, in optimizer steps (not micro-batches -- total_batch_size
    already accounts for grad-accum and world_size). kind="sft"'s default training horizon when
    "num_iterations" is omitted -- used both on a fresh start and (identically) when resuming a
    step that never had an explicit "num_iterations" of its own."""
    return max(1, dataset.num_sequences("train") * sequence_len // total_batch_size)


def _lr_schedule(num_iterations, warmup_steps, warmdown_ratio, final_lr_frac, momentum_warmup_steps):
    """Builds the two per-step schedule functions a training loop needs, from
    modelcore.optim.schedules.lr_multiplier/muon_momentum.

    num_iterations: total training steps (an explicit cfg["num_iterations"] for kind="base", or
        the one-epoch derivation for kind="sft").
    warmup_steps: linear LR warmup length, in steps, before holding at the full LR multiplier 1.0.
    warmdown_ratio: fraction of num_iterations spent linearly decaying the LR multiplier from 1.0
        down to final_lr_frac, at the end of the run.
    final_lr_frac: the LR multiplier warmdown decays to (not zero -- a small residual LR at the
        very end of training).
    momentum_warmup_steps: how many early steps Muon's own momentum ramps over before holding at
        0.97 -- cfg["muon_momentum_warmup_steps"] (DEFAULTS), not derived from num_iterations.

    Returns (get_lr_multiplier, get_muon_momentum): both take a 0-indexed step and return a float
    -- get_lr_multiplier scales every param group's base LR; get_muon_momentum sets Muon's own
    momentum directly (not a multiplier) for the "muon" param groups only. Both are pure functions
    of the absolute step, so a resumed run needs no special-casing here -- the loop just starts
    part-way through range(...) and these compute the correct value from step 0 either way.
    """
    def get_lr_multiplier(it):
        return lr_multiplier(it, num_iterations, warmup_steps, warmdown_ratio, final_lr_frac)

    def get_muon_momentum(it):
        return muon_momentum(it, num_iterations, warmdown_ratio, momentum_warmup_steps=momentum_warmup_steps)

    return get_lr_multiplier, get_muon_momentum


def _checks(cfg):
    """The step's required keys and cross-key rules, as clear errors before anything is built."""
    assert "kind" in cfg, "train: 'kind' is required ('base' or 'sft')"
    kind = cfg["kind"]
    assert kind in ("base", "sft"), f"train: kind must be 'base' or 'sft', got {kind!r}"
    assert "sequence_len" in cfg, "train: 'sequence_len' is required (put it in \"defaults\")"
    assert "total_batch_size" in cfg, (
        "train: 'total_batch_size' is required -- tinylab derives no scaling law of its own; "
        "compute it with `python -m tinylab info <config> --json` and paste the number in."
    )
    assert "eval_tokens" in cfg, "train: 'eval_tokens' is required"
    if kind == "base":
        assert "num_iterations" in cfg, (
            "train: 'num_iterations' is required for kind='base' -- tinylab derives no training "
            "horizon of its own; compute it with `python -m tinylab info <config> "
            "--target-flops=... --json` (training_plan.num_iterations) and paste the number in."
        )
    assert "world_size" in cfg, (
        "train: 'world_size' is required -- the job file fixes the GPU count for a run; edit it "
        "if the launch's GPU configuration changes."
    )
    output_tag = cfg.get("output_tag", cfg["name"])
    # One checkpoint namespace means a base and an sft checkpoint can no longer share a tag by
    # living in different directories -- so an sft step must not write where it reads from, or its
    # source weights would be mixed with (and, resuming, mistaken for) its own output.
    assert kind != "sft" or cfg.get("source_tag") != output_tag, (
        f"train: sft step {cfg['name']!r} has output_tag == source_tag == {output_tag!r} -- it would "
        f"read its starting weights from, and write its result into, the same checkpoint directory. "
        f"Give it a distinct \"output_tag\" (checkpoint tags share one namespace; the tag is where "
        f"you say what kind of checkpoint it is, e.g. \"nanogpt-d12-base\" / \"nanogpt-d12-chat\")."
    )
    return kind, output_tag


def run(cfg: dict, ctx) -> dict:
    """Runs one train step: cfg is a resolved job-file step (see docs/job-file.md for every key),
    ctx the shared Context for this job run. Returns {"op": "train", "kind", "output_tag", "step",
    "val_bpb", "total_training_time"}.

    Resume (ctx.resume): if set, this step first asks checkpoints.resume_point whether it has a
    checkpoint of its own under output_tag, regardless of kind. If so, it loads that checkpoint's own
    model config (never re-resolving "model_config"/"source_tag" -- a resumed run is a continuation,
    not a re-interpretation), optimizer state, and dataloader position, and continues from its saved
    step instead of the kind-specific fresh-start path below. A world_size mismatch against the
    checkpoint's own recorded value is a hard error (MuonAdamW's optimizer state doesn't reshard
    across world_size), and so is a missing optimizer shard for this rank -- resume never silently
    falls back to a fresh optimizer. With no checkpoint under this tag yet, this is a normal fresh
    start (safe to pass --resume unconditionally in an unattended restart script)."""
    kind, output_tag = _checks(cfg)
    defaults = DEFAULTS[kind]
    sequence_len = cfg["sequence_len"]
    push_model, push_optim = cfg.get("push_model", "last"), cfg.get("push_optim", "last")
    remote_mod.validate_policy(push_model, key="push_model")
    remote_mod.validate_policy(push_optim, key="push_optim")

    manager = ctx.model_manager
    # This step's own tokenizer (see Context.tokenizer_for): cfg["tokenizer"] if this step sets
    # one, else the run's default -- e.g. training two differently-vocabbed models sets it per
    # train step, not once in "defaults".
    tokenizer_spec = cfg.get("tokenizer")
    tokenizer = ctx.tokenizer_for(tokenizer_spec)
    vocab_size = tokenizer.get_vocab_size()
    device = ctx.device
    ddp_rank, ddp_world_size = ctx.rank, ctx.world_size
    assert cfg["world_size"] == ddp_world_size, (
        f"train: job file declares world_size={cfg['world_size']}, but this run was launched "
        f"with {ddp_world_size} rank(s) -- edit the job file's \"world_size\" to match the actual "
        f"launch (e.g. torchrun --nproc_per_node)."
    )
    tokenizer_fingerprint = tokenizer.fingerprint()
    print0(f"[{cfg['name']}] compute dtype: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    dataset_name, dataset, token_bytes = prepare.open_prepared(cfg, ctx, kind, sequence_len, tokenizer)
    print0(f"Dataset: {dataset_name} ({dataset.num_sequences('train'):,} train / {dataset.num_sequences('val'):,} val sequences)")

    device_batch_size = cfg.get("device_batch_size", defaults["device_batch_size"])
    total_batch_size = cfg["total_batch_size"]
    checkpoint_dir = checkpoints.resolve_checkpoint_dir(output_tag)

    # -- resume: does this step already have a checkpoint of its own to continue from? --
    resumed_step = checkpoints.resume_point(ctx, cfg["name"], output_tag, ddp_world_size)
    base_model_tag = base_model_step = None
    prior_training_time = 0.0
    resume_dataloader_state = None
    resumed_val_bpb = resumed_min_val_bpb = resumed_smooth_train_loss = None
    optimizer_state = None
    # A fresh kind="sft" start's own momentum warm-start (from "source_tag"'s optimizer shard) --
    # deliberately a *separate* local from optimizer_state above: optimizer_state's resume-path
    # load must stay a bit-exact restore (no LR rescale), while this one is followed by an
    # LR-reset + init_lr_frac rescale below. Conflating the two would silently corrupt resume.
    warm_start_optimizer_state = None
    if resumed_step is not None:
        model, optimizer_state, meta = checkpoints.load_for_resume(checkpoint_dir, resumed_step, device, ddp_rank, manager)
        saved_world_size = meta.get("user_config", {}).get("world_size")
        assert saved_world_size == ddp_world_size, (
            f"train: resume found a checkpoint saved at world_size={saved_world_size}, but "
            f"this run was launched with {ddp_world_size} -- MuonAdamW's optimizer state "
            f"doesn't reshard across world_size; relaunch at the original world_size to resume."
        )
        assert optimizer_state is not None, (
            f"train: resume found no optimizer state for step {resumed_step} rank {ddp_rank} "
            f"in {checkpoint_dir} -- refusing to resume with a freshly-initialized optimizer."
        )
        real_config = model.config
        num_iterations = _horizon(cfg, dataset, sequence_len, total_batch_size)
        assert resumed_step <= num_iterations, (
            f"train: resume: checkpoint at {checkpoint_dir} is already at step {resumed_step}, "
            f"at or past num_iterations={num_iterations} -- raise \"num_iterations\" in the "
            f"job file to continue further, or omit --resume to start over."
        )
        base_model_tag, base_model_step = meta.get("base_model_tag"), meta.get("base_model_step")
        prior_training_time = meta.get("total_training_time", 0.0)
        resume_dataloader_state = meta.get("dataloader_state_dict")
        resumed_val_bpb = meta.get("val_bpb")
        resumed_min_val_bpb = meta.get("min_val_bpb")
        resumed_smooth_train_loss = meta.get("smooth_train_loss")
    elif kind == "base":
        assert "model_config" in cfg, (
            'train (kind=base): "model_config" is required -- a path to a materialized '
            "ModelConfig tree (see llmllab/tools/make_config.py)."
        )
        real_config = modelconfig.load_model_config(cfg["model_config"], sequence_len=sequence_len, vocab_size=vocab_size)
        num_iterations = cfg["num_iterations"]
        model = manager.create_model(real_config, device=device, seed=cfg.get("seed", defaults["seed"]))
    else:
        assert "source_tag" in cfg, "train: kind='sft' requires 'source_tag' (the tag of a prior 'base' train step)"
        # An sft step's "model_config", if given, is a config-override request (e.g. to attach
        # LoRA/DoRA adapters to an already-trained base); omitting it (the common case) loads the
        # checkpoint's own stored config unchanged.
        config_override = modelconfig.load_override(cfg, sequence_len=sequence_len, vocab_size=vocab_size)
        model, _source_tokenizer, meta = checkpoints.load_model(
            cfg["source_tag"], device, phase="train", step=cfg.get("source_step"),
            config_override=config_override, tokenizer_spec=tokenizer_spec, remote=ctx.remote,
        )
        real_config = model.config
        num_iterations = _horizon(cfg, dataset, sequence_len, total_batch_size)
        base_model_tag = meta.get("model_tag")
        base_model_step = meta.get("step")

        # Optimizer momentum warm-start (default on): load source_tag's own optimizer shard for
        # this rank -- kept as a separate warm_start_optimizer_state local (see its declaration
        # above), consumed after the optimizer below is built, then LR-reset by the init_lr_frac
        # block right after that.
        load_optimizer = cfg.get("load_optimizer", defaults["load_optimizer"])
        if load_optimizer and real_config.adapters:
            # The pretrained optimizer's param groups were built for a fully-trainable base model;
            # an adapter-augmented model's groups are shaped differently (a frozen base produces no
            # "matrix"/"embedding"/... groups at all, plus new "adapter"/"adapter_scalar" roles --
            # see modelcore.roles.build_param_groups). Loading the shard would apply momentum state
            # to the wrong parameters entirely, not just stale ones.
            assert "load_optimizer" not in cfg, (
                f"train: sft step {cfg['name']!r} has adapters and an explicit "
                f"\"load_optimizer\": true -- the pretrained optimizer's param-group layout "
                f"does not match an adapter-augmented model. Omit \"load_optimizer\" "
                f"(adapters force the warm-start off)."
            )
            print0(f"[{cfg['name']}] adapters active: skipping optimizer warm-start")
        elif load_optimizer:
            saved_world_size = meta.get("user_config", {}).get("world_size")
            assert saved_world_size == ddp_world_size, (
                f"train: sft step {cfg['name']!r} warm-start (\"load_optimizer\") found "
                f"source_tag={cfg['source_tag']!r} saved at world_size={saved_world_size}, "
                f"but this run was launched with {ddp_world_size} -- MuonAdamW's optimizer "
                f"state doesn't reshard across world_size. Relaunch at the "
                f"original world_size, or set \"load_optimizer\": false to start sft with a "
                f"fresh optimizer instead."
            )
            warm_start_optimizer_state = checkpoints.load_optimizer_state(
                cfg["source_tag"], base_model_step, device, ddp_rank, ddp_world_size, remote=ctx.remote)
            assert warm_start_optimizer_state is not None, (
                f"train: sft step {cfg['name']!r} warm-start (\"load_optimizer\") found no "
                f"optimizer state for source_tag={cfg['source_tag']!r} step {base_model_step} "
                f"rank {ddp_rank} -- refusing to warm-start with a missing shard. Set "
                f"\"load_optimizer\": false to start sft with a fresh optimizer instead."
            )
            print0(f"[{cfg['name']}] loaded optimizer state from {cfg['source_tag']!r} (step {base_model_step}, rank {ddp_rank})")

    # One model_stats computation for all three paths (resume / base-fresh / sft-fresh): printed
    # below and used as flops_per_token for the MFU calculation.
    model_stats = manager.stats(real_config)
    if resumed_step is not None:
        print0(f"[{cfg['name']}] resuming from step {resumed_step}: {model_stats.n_layer} layers, {model_stats.num_params:,} params")
    elif kind == "base":
        print0(f"Model: {model_stats.n_layer} layers, {model_stats.num_params:,} params ({model_stats.num_scaling_params:,} scaling)")
    else:
        print0(f"[{cfg['name']}] sft model: {model_stats.n_layer} layers, {model_stats.num_params:,} params")

    # MFU denominator: an unknown/non-CUDA device name makes peak_flops() return inf (MFU reads
    # 0% rather than a wrong guess) -- see modelcore.runtime.peak_flops's own docstring. This
    # laptop has no CUDA device at all (see AGENTS.md), so MFU is expected to read 0% here.
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    peak_flops = modelcore_runtime.peak_flops(device_name, log=print0)

    if cfg.get("fp8"):
        # modelcore only implements the "tensorwise" recipe, so there is no recipe key to plumb.
        fp8_report = manager.enable_fp8(model)
        print0(f"[{cfg['name']}] fp8: {fp8_report.num_converted}/{fp8_report.num_linear} Linear converted")
    fp8_eval = cfg.get("fp8_eval", defaults["fp8_eval"])

    # Compile for the train/eval forward (fp8 first, then compile -- the order matters).
    # orig_model (uncompiled) is what the optimizer and checkpoint save use: a compiled module's
    # state_dict keys gain an "_orig_mod." prefix. Compiling costs a one-time stall at step 0
    # (~80s measured on a real H200 run) for ~4x steady-state throughput; see docs/architecture.md's
    # "Why the training loop is compiled".
    orig_model = model
    # Every train step of a job file shares one process, and dynamo's recompile limit (8) is per code
    # object, not per model: model.forward and MuonAdamW.step recompile for each step's new model,
    # for train/eval, for doc_masking on/off. Once the limit is hit dynamo silently stops compiling
    # and the rest of the process runs eager (measured: the 5th step of a job at 4x the step time).
    # Drop the previous step's compiled code before compiling this one.
    torch._dynamo.reset()
    model = torch.compile(model, dynamic=False)

    optimizer = manager.create_optimizer(orig_model, modelconfig.optimizer_hparams(cfg, defaults))
    if optimizer_state is not None:
        # A true step-resume (this step's own prior checkpoint): restore exactly, no LR games --
        # this must reproduce bit-for-bit what the interrupted run had (see test_resume.py). The
        # sft momentum warm-start below is a different thing (a *different* source checkpoint's
        # optimizer, LRs deliberately reset after), which is why it lives in
        # warm_start_optimizer_state, a separate local.
        optimizer.load_state_dict(optimizer_state)
    else:
        if warm_start_optimizer_state is not None:
            # sft momentum warm-start: load_state_dict overwrites every group's own "lr"/"initial_lr"
            # with the SOURCE run's saved (warmed-down) values -- capture this run's
            # freshly-computed LRs first and restore them right after, so only the momentum/exp_avg
            # buffers carry over.
            base_lrs = [g["lr"] for g in optimizer.param_groups]
            optimizer.load_state_dict(warm_start_optimizer_state)
            for g, base_lr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base_lr
        # init_lr_frac scales the starting LR down from the (possibly just-warm-started-then-reset)
        # base LR above -- only on a genuine fresh start, never on a resumed step (the branch above).
        # Its default is 1.0 for base, where the key is not even accepted (see _BASE_KEYS). The
        # "initial_lr" write is load-bearing beyond the rescale: ModelManager.apply_schedule
        # computes every step's LR as group["initial_lr"] * lr_mult, and load_state_dict above
        # would otherwise have left it at the source run's own value.
        init_lr_frac = cfg.get("init_lr_frac", defaults["init_lr_frac"])
        for g in optimizer.param_groups:
            g["lr"] *= init_lr_frac
            g["initial_lr"] = g["lr"]

    tokens_per_fwdbwd = device_batch_size * sequence_len * ddp_world_size
    assert total_batch_size % tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {tokens_per_fwdbwd}"
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    eval_every = cfg.get("eval_every", defaults["eval_every"])
    eval_tokens = cfg["eval_tokens"]
    save_every = cfg.get("save_every", defaults["save_every"])  # <= 0: only the final step saves

    # doc_args is built here, outside the compiled model, and passed in as plain data -- never
    # inside a compiled region (see modelcore/AGENTS.md's "Intra-document masking's doc_args must
    # be built outside torch.compile").
    bos_token_id = tokenizer.get_bos_token_id() if cfg.get("doc_masking") else None
    padding_id = dataset.info.padding_id if cfg.get("doc_masking") else None
    doc_masking_max_docs_per_row = cfg.get("doc_masking_max_docs_per_row")

    def make_doc_args(x):
        if bos_token_id is None:
            return None
        max_docs = doc_masking_max_docs_per_row * x.size(0) if doc_masking_max_docs_per_row is not None else None
        return build_doc_args(x, bos_token_id, padding_id=padding_id, max_docs=max_docs)

    get_lr_multiplier, get_muon_momentum = _lr_schedule(
        num_iterations, cfg.get("warmup_steps", defaults["warmup_steps"]),
        cfg.get("warmdown_ratio", defaults["warmdown_ratio"]), cfg.get("final_lr_frac", defaults["final_lr_frac"]),
        cfg.get("muon_momentum_warmup_steps", defaults["muon_momentum_warmup_steps"]),
    )

    def _save(step, val_bpb, dataloader_state, elapsed, min_val_bpb, smooth_train_loss):
        saved_config = checkpoints.checkpoint_config(real_config, tokenizer, ctx.tokenizer_name_for(tokenizer_spec), chat=kind == "sft")
        meta_data = {
            "step": step, "val_bpb": val_bpb, "tokenizer_fingerprint": tokenizer_fingerprint,
            "model_config": manager.config_to_dict(saved_config),
            # cfg's own "model_config" is excluded from user_config: it's the *request* (a path,
            # already resolved into the "model_config" key above -- a different dict, same name by
            # coincidence), and would otherwise duplicate that key's content under a different,
            # unresolved shape.
            "user_config": {k: v for k, v in cfg.items() if k != "model_config"},
            "device_batch_size": device_batch_size, "max_seq_len": sequence_len, "total_batch_size": total_batch_size,
            "dataloader_state_dict": dataloader_state, "total_training_time": elapsed,
            # Running best val bpb and an EMA of the per-step train loss, across the whole run
            # (resume picks both up from the loaded checkpoint's own meta rather than resetting them).
            "min_val_bpb": min_val_bpb, "smooth_train_loss": smooth_train_loss,
        }
        if kind == "sft":
            meta_data["base_model_tag"] = base_model_tag
            meta_data["base_model_step"] = base_model_step
        inputs = {"dataset": f"{remote_mod.PREPARED}/{dataset_name}"}
        tokenizer_entity = bucket_entity(tokenizer_spec, ctx.tokenizer_name_for(tokenizer_spec))
        if tokenizer_entity is not None:
            inputs["tokenizer"] = tokenizer_entity
        if kind == "sft":
            inputs["source"] = f"{remote_mod.CHECKPOINTS}/{cfg['source_tag']}@{base_model_step}"
        inputs["model_config_sha256"] = checkpoints.model_config_sha256(meta_data["model_config"])
        push = checkpoints.make_push(ctx, cfg, output_tag=output_tag, checkpoint_dir=checkpoint_dir, push_model=push_model,
                                     push_optim=push_optim, inputs=inputs, metrics={"val_bpb": val_bpb})
        checkpoints.save_checkpoint(checkpoint_dir, step, orig_model.state_dict(), optimizer.state_dict(), meta_data,
                                    rank=ddp_rank, push=push, barrier=ctx.uploader is not None)
        # Only after the save above has fully returned -- this is the signal job.run_file's job
        # state file trusts as "this step is safely resumable from here" (see ctx.record_checkpoint
        # and checkpoints.resume_point). Firing it any earlier would defeat the whole point.
        ctx.record_checkpoint(cfg["name"], step)
        print0(f"[{cfg['name']}] saved checkpoint: {checkpoint_dir} (step {step})")

    train_loader = ctx.data_manager.batches(dataset, "train", device_batch_size, device=device, rank=ddp_rank, world_size=ddp_world_size, resume=resume_dataloader_state, infinite=True)
    x, y, dataloader_state = next(train_loader)

    val_bpb = resumed_val_bpb
    min_val_bpb = resumed_min_val_bpb
    smooth_train_loss = resumed_smooth_train_loss
    start_step = resumed_step or 0
    t_start = time.time()
    dt = 0.0
    tokens_per_sec = 0.0
    mfu = 0.0
    for step in range(start_step, num_iterations + 1):
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
            min_val_bpb = val_bpb if min_val_bpb is None else min(min_val_bpb, val_bpb)
            print0(f"[{cfg['name']}] step {step:05d} | val bpb: {val_bpb:.6f} (min {min_val_bpb:.6f})")
            model.train()
        # save: always at the end of the run, or every save_every steps -- but never at step 0
        # (no training has happened yet) and never re-saving the step this run just resumed from
        # (it's already on disk, unchanged). `step` here already means "this many iterations have
        # completed" (0 at the very start), the same convention the eval check above uses, so no
        # off-by-one between what gets evaluated and what gets saved under the same step number.
        if last_step or (save_every > 0 and step > 0 and step != resumed_step and step % save_every == 0):
            _save(step, val_bpb, dataloader_state, prior_training_time + (time.time() - t_start), min_val_bpb, smooth_train_loss)
        if last_step:
            break
        step_t0 = time.time()
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
        # A plain EMA (no bias-correction warm-up), seeded from the resumed checkpoint's own value
        # when there is one, else from this run's very first step's own loss.
        smooth_train_loss = step_loss if smooth_train_loss is None else 0.9 * smooth_train_loss + 0.1 * step_loss
        dt = max(time.time() - step_t0, 1e-9)
        tokens_per_sec = total_batch_size / dt
        # peak_flops is inf for an unrecognized/non-CUDA device (see modelcore.runtime.peak_flops),
        # so this divides out to 0.0 rather than raising -- MFU just reads 0% there.
        mfu = model_stats.flops_per_token * total_batch_size / dt / peak_flops
        if step % 10 == 0:
            print0(f"[{cfg['name']}] step {step:05d}/{num_iterations} | loss {step_loss:.4f} | "
                   f"dt {dt * 1000:.0f}ms | tok/s {tokens_per_sec:,.0f} | mfu {mfu * 100:.1f}%")
    total_training_time = prior_training_time + (time.time() - t_start)

    return {
        "op": "train", "kind": kind, "output_tag": output_tag, "step": num_iterations,
        "val_bpb": val_bpb, "total_training_time": total_training_time,
    }
