"""
The `rl` op: reinforcement learning on GSM8K, starting from an sft checkpoint. Ported from
our nanochat fork's scripts/chat_rl.py (see llmllab/docs/history.md). The algorithm is deliberately plainer
than "GRPO", closer to REINFORCE:

1) no trust region -- no KL regularization to a reference model;
2) on-policy, so no PPO ratio/clip;
3) DAPO-style normalization: token-level, not sequence-level;
4) the advantage is (r - mean(r)) over an example's samples, not the z-score (r - mu) / sigma.

Each step draws `examples_per_step` GSM8K problems (split across ranks), samples `num_samples`
completions per problem through tinylab.engine.Engine, scores them with benchcore's GSM8K reward
(1.0 if the final answer is right), and takes one optimizer step on sum(logp * advantage) over the
sampled, non-forced tokens. The LR ramps linearly down to zero over the run.

Reuses train.py's plumbing: one checkpoint namespace (output_tag, kind="rl" in the meta, template
stamped "nanochat" like sft), the optimizer saved with each checkpoint, and step-level resume
(ctx.resume): a resumed run reloads model + optimizer, checks world_size, and continues the
deterministic example/seed schedule from where it stopped. Unlike train, no dataset has to be
prepared: the problems come from benchcore.GSM8K (cached under the base dir; pulled from the
bucket's task_data/ when a "remote" is set). Not exercised with adapters ("model_config" override)
beyond the same wiring sft has -- as in nanochat.
"""
import dataclasses
import hashlib
import itertools
import json
import time

import torch

from modelcore import OptimizerHparams

from tinylab import checkpoints, modelconfig
from tinylab import remote as remote_mod
from tinylab.engine import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TOP_K, Engine
from tinylab.runtime import get_base_dir, print0

_KEYS = {
    "source_tag", "source_step", "output_tag", "num_epochs", "num_iterations", "examples_per_step", "num_samples", "device_batch_size",
    "max_new_tokens", "temperature", "top_k", "embedding_lr", "unembedding_lr", "matrix_lr", "weight_decay",
    "init_lr_frac", "adapter_lr", "adapter_scalar_lr", "eval_every", "eval_examples", "save_every",
    "push_model", "push_optim",
}


def accepted_keys(cfg: dict) -> set:
    return _KEYS


def _gsm8k_tasks(ctx):
    """(train task, val task): GSM8K "main", train and test splits."""
    from benchcore import GSM8K
    if ctx.remote is not None:
        remote_mod.pull_prefix(ctx.remote, "task_data")
    return (GSM8K(subset="main", split="train", cache_dir=get_base_dir()),
            GSM8K(subset="main", split="test", cache_dir=get_base_dir()))


@torch.no_grad()
def _rollout(engine, tokenizer, task, example_idx, step, *, num_samples, device_batch_size, max_new_tokens,
             temperature, top_k, device):
    """Samples num_samples completions of one training problem and turns them into one training
    batch: (sequences, inputs, targets, rewards, advantages). targets is -1 (ignore) wherever the
    Engine's mask is 0 -- the prompt and any tool-forced tokens -- so only sampled tokens train."""
    assistant_end = tokenizer.encode_special("<|assistant_end|>")  # padding only; masked out of the loss
    conversation = task[example_idx]
    # Prime the Assistant for a completion: keep <|assistant_start|>, drop everything after it.
    tokens = tokenizer.render_for_completion(conversation)
    prefix_length = len(tokens)

    engine.model.eval()
    sequences, masks = [], []
    for sampling_step in range(num_samples // device_batch_size):  # sequential, to prevent OOMs
        seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF  # positive half of int32; new per sampling step
        batch_sequences, batch_masks = engine.generate_batch(
            tokens, num_samples=device_batch_size, max_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k or None, seed=seed)
        sequences.extend(batch_sequences)
        masks.extend(batch_masks)

    rewards = [task.reward(conversation, tokenizer.decode(seq[prefix_length:])) for seq in sequences]

    max_length = max(len(seq) for seq in sequences)
    ids = torch.tensor([seq + [assistant_end] * (max_length - len(seq)) for seq in sequences], dtype=torch.long, device=device)
    mask_ids = torch.tensor([m + [0] * (max_length - len(m)) for m in masks], dtype=torch.long, device=device)
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()  # clone: the next line writes in place
    targets[mask_ids[:, 1:] == 0] = -1
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    advantages = rewards - rewards.mean()  # (r - mu), not (r - mu) / sigma
    return sequences, inputs, targets, rewards, advantages


def _evaluate_pass_at_k(engine, tokenizer, task, ctx, *, max_examples, k, max_new_tokens):
    """pass@1..k on the first max_examples problems of `task` (k samples each, temperature 1.0):
    the fraction of problems with at least one correct answer among the first j samples. Ranks
    split the problems and all-reduce the counts. Returns a list of k floats."""
    engine.model.eval()
    max_examples = min(max_examples, len(task))
    passk = torch.zeros(k, device=ctx.device)
    count = 0
    for idx in range(ctx.rank, max_examples, ctx.world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        sequences, _masks = engine.generate_batch(tokens, num_samples=k, max_tokens=max_new_tokens, temperature=1.0, top_k=DEFAULT_TOP_K)
        correct = [bool(task.evaluate(conversation, tokenizer.decode(seq[len(tokens):]))) for seq in sequences]
        for j in range(1, k + 1):
            passk[j - 1] += any(correct[:j])
        count += 1
    count = torch.tensor(count, dtype=torch.long, device=ctx.device)
    if ctx.world_size > 1:
        import torch.distributed as dist
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(passk, op=dist.ReduceOp.SUM)
    return (passk / max(count.item(), 1)).tolist()


def run(cfg: dict, ctx) -> dict:
    """Runs one rl step: cfg is a resolved job-file step (see docs/job-file.md for every key), ctx
    the shared Context for this job run. Returns {"op": "rl", "output_tag", "step", "pass_at_k",
    "mean_reward", "total_training_time"} (pass_at_k: the final evaluation, [] if eval_every is 0).

    "step" counts completed optimizer updates, so a checkpoint at step N has had N updates and a
    resumed run continues with update N+1. Resume (ctx.resume) is the same contract as train's: a
    world_size mismatch or a missing optimizer shard is a hard error, never a silent fresh start;
    no checkpoint under output_tag yet is an ordinary fresh start."""
    assert "source_tag" in cfg, "rl: 'source_tag' is required (the tag of an sft checkpoint)"
    assert "world_size" in cfg, (
        "rl: 'world_size' is required -- the job file fixes the GPU count for a run; edit it if the launch's GPU configuration changes.")
    output_tag = cfg.get("output_tag", cfg["name"])
    assert cfg["source_tag"] != output_tag, (
        f"rl: step {cfg['name']!r} has output_tag == source_tag == {output_tag!r} -- give it a distinct \"output_tag\".")
    push_model, push_optim = cfg.get("push_model", "last"), cfg.get("push_optim", "last")
    remote_mod.validate_policy(push_model, key="push_model")
    remote_mod.validate_policy(push_optim, key="push_optim")

    manager = ctx.model_manager
    tokenizer_spec = cfg.get("tokenizer")
    tokenizer = ctx.tokenizer_for(tokenizer_spec)
    device = ctx.device
    ddp_rank, ddp_world_size = ctx.rank, ctx.world_size
    assert cfg["world_size"] == ddp_world_size, (
        f"rl: job file declares world_size={cfg['world_size']}, but this run was launched with {ddp_world_size} rank(s) -- "
        f"edit the job file's \"world_size\" to match the actual launch.")

    num_epochs = cfg.get("num_epochs", 1)
    examples_per_step = cfg.get("examples_per_step", 16)
    num_samples = cfg.get("num_samples", 16)
    device_batch_size = cfg.get("device_batch_size", 8)
    max_new_tokens = cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    temperature = cfg.get("temperature", 1.0)
    top_k = cfg.get("top_k", DEFAULT_TOP_K)
    eval_every = cfg.get("eval_every", 60)
    eval_examples = cfg.get("eval_examples", 400)
    save_every = cfg.get("save_every", 60)
    assert num_samples % device_batch_size == 0, f"rl: num_samples ({num_samples}) must be a multiple of device_batch_size ({device_batch_size})"
    assert examples_per_step % ddp_world_size == 0, f"rl: examples_per_step ({examples_per_step}) must be divisible by world_size ({ddp_world_size})"
    examples_per_rank = examples_per_step // ddp_world_size

    train_task, val_task = _gsm8k_tasks(ctx)
    num_steps = cfg.get("num_iterations") or (len(train_task) // examples_per_step) * num_epochs
    assert num_steps > 0, f"rl: {len(train_task)} training problems is fewer than examples_per_step={examples_per_step}"
    print0(f"[{cfg['name']}] {num_steps} steps, {examples_per_step * num_samples} sequences per step")

    checkpoint_dir = checkpoints.resolve_checkpoint_dir(output_tag)
    resumed_step = None
    optimizer_state = None
    prior_training_time = 0.0
    if ctx.resume:
        resumed_step = ctx.resume_checkpoint_step(cfg["name"])
        if ctx.remote is not None:
            try:
                checkpoints.ensure_local(ctx.remote, output_tag, resumed_step, optim=True, ranks=range(ddp_world_size))
            except FileNotFoundError:
                pass
        if resumed_step is None:
            try:
                resumed_step = checkpoints.find_last_step(checkpoint_dir)
            except FileNotFoundError:
                pass
    if resumed_step is not None:
        model, optimizer_state, meta = checkpoints.load_for_resume(checkpoint_dir, resumed_step, device, ddp_rank, manager)
        saved_world_size = meta.get("user_config", {}).get("world_size")
        assert saved_world_size == ddp_world_size, (
            f"rl: resume found a checkpoint saved at world_size={saved_world_size}, but this run was launched with "
            f"{ddp_world_size} -- MuonAdamW's optimizer state doesn't reshard across world_size; relaunch at the original one.")
        assert optimizer_state is not None, f"rl: resume found no optimizer state for step {resumed_step} rank {ddp_rank} in {checkpoint_dir}"
        assert resumed_step <= num_steps, f"rl: checkpoint at {checkpoint_dir} is at step {resumed_step}, past the {num_steps} steps this run has"
        base_model_tag, base_model_step = meta.get("base_model_tag"), meta.get("base_model_step")
        prior_training_time = meta.get("total_training_time", 0.0)
        print0(f"[{cfg['name']}] resuming from step {resumed_step}")
    else:
        # A "model_config" here is an override request (attach LoRA/DoRA adapters), as in sft.
        config_override = None
        if "model_config" in cfg:
            config_override = modelconfig.load_model_config(
                cfg["model_config"], sequence_len=cfg["sequence_len"], vocab_size=tokenizer.get_vocab_size())
        model, _tok, meta = checkpoints.load_model(
            cfg["source_tag"], device, phase="train", step=cfg.get("source_step"), config_override=config_override,
            tokenizer_spec=tokenizer_spec, remote=ctx.remote)
        base_model_tag, base_model_step = meta.get("model_tag"), meta.get("step")
    real_config = model.config
    engine = Engine(model, tokenizer, manager=manager)

    hparams = dict(unembedding_lr=cfg.get("unembedding_lr", 0.004), embedding_lr=cfg.get("embedding_lr", 0.2),
                   matrix_lr=cfg.get("matrix_lr", 0.02), weight_decay=cfg.get("weight_decay", 0.0))
    for key in ("adapter_lr", "adapter_scalar_lr"):
        if key in cfg:
            hparams[key] = cfg[key]
    optimizer = manager.create_optimizer(model, OptimizerHparams(**hparams))
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)  # bit-exact restore: LRs and initial_lr come with it
    else:
        # The RL LR is a small fraction of the base LR, then ramps linearly to zero.
        for group in optimizer.param_groups:
            group["lr"] *= cfg.get("init_lr_frac", 0.05)
            group["initial_lr"] = group["lr"]

    tokenizer_fingerprint = tokenizer.fingerprint()

    def _save(step, pass_at_k, elapsed):
        saved_config = dataclasses.replace(real_config, tokenizer=tokenizer.descriptor(ctx.tokenizer_name_for(tokenizer_spec)),
                                           template="nanochat")  # RL trains in the chat format, like sft
        meta_data = {
            "step": step, "kind": "rl", "tokenizer_fingerprint": tokenizer_fingerprint,
            "model_config": manager.config_to_dict(saved_config),
            "user_config": {k: v for k, v in cfg.items() if k != "model_config"},
            "base_model_tag": base_model_tag, "base_model_step": base_model_step,
            "total_training_time": elapsed, "pass_at_k": pass_at_k,
        }
        push = None
        if ctx.uploader is not None and (push_model != "none" or push_optim != "none"):
            def push(step_):
                inputs = {"source": f"checkpoints/{cfg['source_tag']}@{base_model_step}",
                          "model_config_sha256": hashlib.sha256(json.dumps(meta_data["model_config"], sort_keys=True).encode()).hexdigest()}
                if tokenizer_spec is None or "/" not in tokenizer_spec:
                    inputs["tokenizer"] = f"tokenizers/{ctx.tokenizer_name_for(tokenizer_spec)}"
                producer = ctx.producer(cfg["name"])
                ctx.uploader.submit(
                    f"checkpoints/{output_tag}",
                    lambda remote: remote_mod.push_checkpoint_step(
                        remote, checkpoint_dir, output_tag, step_, push_model=push_model, push_optim=push_optim,
                        ranks=range(ddp_world_size), producer=producer, inputs=inputs, motivation=cfg.get("_comment", ""),
                        metrics={"step": step_, "pass_at_1": (pass_at_k or [None])[0]}),
                    label=f"checkpoints/{output_tag}@{step_}")
        checkpoints.save_checkpoint(checkpoint_dir, step, model.state_dict(), optimizer.state_dict(), meta_data,
                                    rank=ddp_rank, push=push, barrier=ctx.uploader is not None)
        ctx.record_checkpoint(cfg["name"], step)  # only after the save fully returned
        print0(f"[{cfg['name']}] saved checkpoint: {checkpoint_dir} (step {step})")

    # Each rank owns every world_size-th problem; the cycle is a pure function of the step, so a
    # resumed run picks up exactly where the interrupted one stopped.
    rank_indices = list(range(ddp_rank, len(train_task), ddp_world_size))
    start_step = resumed_step or 0
    example_iter = itertools.islice(itertools.cycle(rank_indices), (start_step * examples_per_rank) % len(rank_indices), None)

    eval_kwargs = dict(max_examples=eval_examples, k=device_batch_size, max_new_tokens=max_new_tokens)
    pass_at_k, mean_reward = [], None
    t_start = time.time()
    for step in range(start_step, num_steps):
        if eval_every > 0 and step % eval_every == 0:
            pass_at_k = _evaluate_pass_at_k(engine, tokenizer, val_task, ctx, **eval_kwargs)
            print0(f"[{cfg['name']}] step {step} | " + ", ".join(f"pass@{j + 1}: {v:.4f}" for j, v in enumerate(pass_at_k)))

        rewards_list, sequence_lengths = [], []
        for _ in range(examples_per_rank):
            sequences, inputs_all, targets_all, rewards_all, advantages_all = _rollout(
                engine, tokenizer, train_task, next(example_iter), step, num_samples=num_samples,
                device_batch_size=device_batch_size, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, device=device)
            model.train()
            num_passes = inputs_all.size(0) // device_batch_size
            for pass_idx in range(num_passes):
                b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
                inputs, targets, advantages = inputs_all[b0:b1], targets_all[b0:b1], advantages_all[b0:b1]
                logp = -model(inputs, targets, loss_reduction="none").view_as(inputs)  # (B, T); the loss is NLL = -logp
                # ignore_index=-1 makes invalid tokens' loss 0. Normalized by valid tokens, passes and examples per rank.
                pg_obj = (logp * advantages.unsqueeze(-1)).sum()
                pg_obj = pg_obj / ((targets >= 0).sum().clamp(min=1) * num_passes * examples_per_rank)
                loss = -pg_obj  # on-policy: no PPO ratio/clip needed
                loss.backward()
            rewards_list.append(rewards_all.mean().item())
            sequence_lengths.extend(len(seq) for seq in sequences)

        mean_reward = sum(rewards_list) / len(rewards_list)
        mean_length = sum(sequence_lengths) / len(sequence_lengths)
        if ddp_world_size > 1:
            import torch.distributed as dist
            stats = torch.tensor([mean_reward, mean_length], dtype=torch.float, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.AVG)
            mean_reward, mean_length = stats.tolist()
        lrm = 1.0 - step / num_steps  # linear rampdown to zero
        manager.apply_schedule(optimizer, lr_mult=lrm)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        print0(f"[{cfg['name']}] step {step + 1}/{num_steps} | reward {mean_reward:.4f} | sequence length {mean_length:.1f} | lrm {lrm:.3f}")

        completed = step + 1
        if completed == num_steps or (save_every > 0 and completed % save_every == 0):
            if completed == num_steps and eval_every > 0:
                pass_at_k = _evaluate_pass_at_k(engine, tokenizer, val_task, ctx, **eval_kwargs)
                print0(f"[{cfg['name']}] final | " + ", ".join(f"pass@{j + 1}: {v:.4f}" for j, v in enumerate(pass_at_k)))
            _save(completed, pass_at_k, prior_training_time + (time.time() - t_start))

    return {"op": "rl", "output_tag": output_tag, "step": num_steps, "pass_at_k": pass_at_k, "mean_reward": mean_reward,
            "total_training_time": prior_training_time + (time.time() - t_start)}
