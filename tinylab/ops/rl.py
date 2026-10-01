"""
The `rl` op: reinforcement learning on a benchcore chat task (GSM8K by default, the "task" key),
starting from an sft checkpoint. Ported from
our nanochat fork's scripts/chat_rl.py (see llmllab/docs/history.md). The algorithm is deliberately plainer
than "GRPO", closer to REINFORCE:

1) no trust region -- no KL regularization to a reference model;
2) on-policy, so no PPO ratio/clip;
3) DAPO-style normalization: token-level, not sequence-level;
4) the advantage is (r - mean(r)) over an example's samples, not the z-score (r - mu) / sigma.

Each step draws `examples_per_step` training problems (split across ranks), samples `num_samples`
completions per problem through tinylab.engine.Engine, scores them with the task's own `reward()`
(for GSM8K: 1.0 if the final answer is right), and takes one optimizer step on sum(logp * advantage) over the
sampled, non-forced tokens. The LR ramps linearly down to zero over the run.

Shares train's plumbing (checkpoints.py): one checkpoint namespace (output_tag, kind="rl" in the
meta, the chat template stamped like sft), the optimizer saved with each checkpoint, and step-level
resume: a resumed run reloads model + optimizer, checks world_size, and continues the deterministic
example/seed schedule from where it stopped. Unlike train, no dataset has to be
prepared: the problems come from the benchcore task (cached under the base dir; pulled from the
bucket's task_data/ when a "remote" is set). A task without a `reward()` cannot be used. Not exercised with adapters ("model_config" override)
beyond the same wiring sft has.
"""
import itertools
import time

import torch

from tinylab import checkpoints, data, modelconfig
from tinylab import remote as remote_mod
from tinylab.engine import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TOP_K, Engine
from tinylab.runtime import print0
from tinylab.tokenizer import bucket_entity

_KEYS = {
    "task", "source_tag", "source_step", "output_tag", "num_epochs", "num_iterations", "examples_per_step", "num_samples", "device_batch_size",
    "max_new_tokens", "temperature", "top_k", "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr", "weight_decay",
    "init_lr_frac", "adapter_lr", "adapter_scalar_lr", "conv_lr", "ssm_lr", "eval_every", "eval_examples", "save_every",
    "push_model", "push_optim",
}

# Every default this op applies, in one place (docs/job-file.md documents them). The LR dials are
# smaller than train's: rl fine-tunes an already-sft'd model, and init_lr_frac then scales them down
# again before the linear ramp to zero.
DEFAULTS = {
    "task": "gsm8k", "num_epochs": 1, "examples_per_step": 16, "num_samples": 16, "device_batch_size": 8, "temperature": 1.0,
    "eval_every": 60, "eval_examples": 400, "save_every": 60, "init_lr_frac": 0.05,
    "unembedding_lr": 0.004, "embedding_lr": 0.2, "matrix_lr": 0.02, "scalar_lr": 0.5, "weight_decay": 0.0,
}


def accepted_keys(cfg: dict) -> set:
    return _KEYS


def _tasks(ctx, name):
    """(train task, val task) of the benchcore chat task `name`. The task must score a completion
    with a reward (`reward(conversation, completion)`), which is what the policy gradient trains on."""
    if ctx.remote is not None:
        remote_mod.pull_prefix(ctx.remote, remote_mod.TASK_DATA)
    train, val = data.build_task(name, "train"), data.build_task(name, "test")
    assert hasattr(train, "reward"), f"rl: task {name!r} has no reward() -- only a task that defines one can be trained on"
    return train, val


@torch.no_grad()
def _rollout(engine, tokenizer, task, example_idx, step, *, num_samples, device_batch_size, max_new_tokens,
             temperature, top_k, device):
    """Samples num_samples completions of one training problem and turns them into one training
    batch: (sequences, inputs, targets, rewards, advantages). targets is -1 (ignore) wherever the
    Engine's mask is 0 -- the prompt and any tool-forced tokens -- so only sampled tokens train."""
    assistant_end = tokenizer.ids.assistant_end  # padding only; masked out of the loss
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
    defaults = DEFAULTS
    assert cfg["world_size"] == ddp_world_size, (
        f"rl: job file declares world_size={cfg['world_size']}, but this run was launched with {ddp_world_size} rank(s) -- "
        f"edit the job file's \"world_size\" to match the actual launch.")

    num_epochs = cfg.get("num_epochs", defaults["num_epochs"])
    examples_per_step = cfg.get("examples_per_step", defaults["examples_per_step"])
    num_samples = cfg.get("num_samples", defaults["num_samples"])
    device_batch_size = cfg.get("device_batch_size", defaults["device_batch_size"])
    max_new_tokens = cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    temperature = cfg.get("temperature", defaults["temperature"])
    top_k = cfg.get("top_k", DEFAULT_TOP_K)
    eval_every = cfg.get("eval_every", defaults["eval_every"])
    eval_examples = cfg.get("eval_examples", defaults["eval_examples"])
    save_every = cfg.get("save_every", defaults["save_every"])  # <= 0: only the final step saves
    assert num_samples % device_batch_size == 0, f"rl: num_samples ({num_samples}) must be a multiple of device_batch_size ({device_batch_size})"
    assert examples_per_step % ddp_world_size == 0, f"rl: examples_per_step ({examples_per_step}) must be divisible by world_size ({ddp_world_size})"
    examples_per_rank = examples_per_step // ddp_world_size

    train_task, val_task = _tasks(ctx, cfg.get("task", defaults["task"]))
    num_steps = cfg.get("num_iterations") or (len(train_task) // examples_per_step) * num_epochs
    assert num_steps > 0, f"rl: {len(train_task)} training problems is fewer than examples_per_step={examples_per_step}"
    print0(f"[{cfg['name']}] {num_steps} steps, {examples_per_step * num_samples} sequences per step")

    checkpoint_dir = checkpoints.resolve_checkpoint_dir(output_tag)
    resumed_step = checkpoints.resume_point(ctx, cfg["name"], output_tag, ddp_world_size)
    optimizer_state = None
    prior_training_time = 0.0
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
        config_override = modelconfig.load_override(
            cfg, sequence_len=cfg["sequence_len"], vocab_size=tokenizer.get_vocab_size())
        model, _tok, meta = checkpoints.load_model(
            cfg["source_tag"], device, phase="train", step=cfg.get("source_step"), config_override=config_override,
            tokenizer_spec=tokenizer_spec, remote=ctx.remote)
        base_model_tag, base_model_step = meta.get("model_tag"), meta.get("step")
    real_config = model.config
    engine = Engine(model, tokenizer, manager=manager)

    optimizer = manager.create_optimizer(model, modelconfig.optimizer_hparams(cfg, defaults))
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)  # bit-exact restore: LRs and initial_lr come with it
    else:
        # The RL LR is a small fraction of the base LR, then ramps linearly to zero.
        for group in optimizer.param_groups:
            group["lr"] *= cfg.get("init_lr_frac", defaults["init_lr_frac"])
            group["initial_lr"] = group["lr"]

    tokenizer_fingerprint = tokenizer.fingerprint()

    def _save(step, pass_at_k, elapsed):
        saved_config = checkpoints.checkpoint_config(real_config, tokenizer, ctx.tokenizer_name_for(tokenizer_spec), chat=True)
        meta_data = {
            "step": step, "kind": "rl", "tokenizer_fingerprint": tokenizer_fingerprint,
            "model_config": manager.config_to_dict(saved_config),
            "user_config": {k: v for k, v in cfg.items() if k != "model_config"},
            "base_model_tag": base_model_tag, "base_model_step": base_model_step,
            "total_training_time": elapsed, "pass_at_k": pass_at_k,
        }
        inputs = {"source": f"{remote_mod.CHECKPOINTS}/{cfg['source_tag']}@{base_model_step}",
                  "model_config_sha256": checkpoints.model_config_sha256(meta_data["model_config"])}
        tokenizer_entity = bucket_entity(tokenizer_spec, ctx.tokenizer_name_for(tokenizer_spec))
        if tokenizer_entity is not None:
            inputs["tokenizer"] = tokenizer_entity
        push = checkpoints.make_push(ctx, cfg, output_tag=output_tag, checkpoint_dir=checkpoint_dir, push_model=push_model,
                                     push_optim=push_optim, inputs=inputs, metrics={"pass_at_1": (pass_at_k or [None])[0]})
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
