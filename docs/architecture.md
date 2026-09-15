# Architecture

How a job file becomes a run, what each module owns, and where tinylab's own code ends and a
subsystem's begins.

```
tinylab/
  __main__.py           the CLI: parses argv, dispatches to job.run_file or chat.main
  job.py                loads a job file, deep-merges "defaults", validates keys, runs steps in order
  runtime.py             base dir, device/DDP init, print0
  tokenizer.py           RustBPETokenizer: BPE encode/decode, special tokens, conversation rendering
  default_tokenizer/     committed tokenizer.pkl + token_bytes.pt (a fixed, pre-trained vocab)
  modelconfig.py          loads + validates a materialized modelcore.ModelConfig tree (no
                          preset/depth-dial derivation -- that's nanochat's job, see below)
  checkpoints.py         checkpoint tag/step naming over modelcore's FileSystemStore
  engine.py               KV-cached generation + calculator tool use
  chat.py                 the `chat` CLI command
  data.py                 corpus identity: ClimbMix shard URLs, SmolTalk
  ops/
    __init__.py            OPS registry + Context (device/managers, passed explicitly)
    prepare.py              op: prepare
    train.py                op: train (kind base|sft, one shared loop)
    bench.py                op: bench (suite core|chat)
jobs/
  smoke.json              tiny end-to-end pipeline, runs on a laptop in minutes
  speedrun.json            real-scale template for a multi-GPU pod
  contest.json             two-architecture comparison, matching nanochat's runs/contest_d12.sh
  configs/                 materialized ModelConfig trees the job files above point at, dumped by
                          nanochat's scripts/model_info.py --dump-config (see AGENTS.md)
```

## From a job file to a run

```
__main__.main()
  -> job.run_file(path, only=..., dry_run=...)
       -> job.load(path)                 # parse JSON, check top-level keys
       -> job.resolve_steps(job, only)   # deep-merge defaults, validate each step's keys
       -> for each step: OPS[step["op"]].run(step, ctx)
```

`resolve_steps` deep-merges `job["defaults"]` into every step, then validates the merged result
against that op's own `accepted_keys(cfg)` — a *function*, not a static set, because `prepare` and
`train` accept a different key set depending on the step's own `"kind"` (`"base"` vs `"sft"`), and
`bench` depending on `"suite"` (`"core"` vs `"chat"`). A key that's real for the wrong kind (e.g.
`"mmlu_epochs"` on a `kind: base` prepare step) is caught the same way a genuinely unknown key is —
see `docs/job-file.md` for the full key reference this validates against.

`chat.main()` follows the identical `job.load` -> `job.resolve_chat` -> `job.check_known_keys`
shape, but is never reachable through `OPS` — it's a separate CLI command (`python -m tinylab
chat <job.json>`), not a pipeline step. A pipeline step runs to completion unattended; a REPL
blocks on a human at a prompt, so it can't be one.

## `Context`

`tinylab.ops.Context` carries the device, rank/world_size, and lazily-built `ModelManager`/
`DataManager`/`BenchManager`/tokenizer singletons for one run. It's constructed once in
`job.run_file` and passed explicitly to every op's `run(cfg, ctx)` — never a module-level global,
matching the subsystems' own no-ambient-globals rule (see
[`llmllab/docs/subsystem-conventions.md`](../../llmllab/docs/subsystem-conventions.md), which
resolves in a local checkout of the whole family). This is what lets a
multi-step job share one device/process-group setup instead of re-initializing it per step.

## Where each subsystem's boundary sits

- **`modelcore.ModelManager`** creates, loads, and saves models and optimizers, and computes their
  FLOPs/param/KV-cache stats. `tinylab.modelconfig` loads and validates the materialized
  `ModelConfig` tree a job file's `"model_config"` names — tinylab has no depth-dial layer of its
  own; the tree is produced entirely by nanochat's `scripts/model_info.py --dump-config` (or hand-
  written) before it ever reaches tinylab. modelcore itself never sees a depth dial either way,
  only the resolved tree.
- **`datacore.DataManager`** prepares a raw corpus into a packed, on-disk dataset and reads it back
  as batches. `tinylab.data` is the layer that knows *which* corpus (ClimbMix's URL, SmolTalk's
  HuggingFace path) — datacore itself has no opinion on where data comes from.
- **`benchcore.BenchManager`** scores a model against CORE or the chat-task suite. A `modelcore`
  model already satisfies `benchcore.Model` (it's `__call__(input_ids) -> logits` plus
  `get_device()`); `tinylab.engine.Engine.generate_batch` is the adapter that additionally
  satisfies `benchcore.Generator`, which GSM8K/HumanEval's generative scoring needs — this adapter
  has to live in the host, not in either subsystem, since it's the one place that knows both a
  model and a tokenizer.
- **`tinylab.checkpoints`** owns tag/step naming and `meta.json`'s extra fields on top of
  `modelcore.store.FileSystemStore`, which owns the actual model/optimizer artifact format.

## On-disk layout

Everything lives under `<base_dir>` (`~/.cache/tinylab/`, or `TINYLAB_BASE_DIR` to override) —
separate from `nanochat`'s `~/.cache/nanochat/`, even though the two produce identical token ids
with a compatible tokenizer.

```
<base_dir>/
  tokenizer/                 tokenizer.pkl (+ token_bytes.pt), copied from the bundled default on first use
  base_data_climbmix/        downloaded ClimbMix parquet shards
  prepared/<dataset_name>/   a datacore dataset: packed sequences + manifest
  base_checkpoints/<tag>/    model_<step>.pt, meta_<step>.json, config_<step>.json, optim_<step>_rank<r>.pt
  chatsft_checkpoints/<tag>/ same shape, for SFT checkpoints
```

`meta_<step>.json` carries: `step`, `val_bpb`, `tokenizer_fingerprint`, `user_config` (the resolved
job-file step, minus `"model_config"` — that's already captured in `config_<step>.json`), `device_batch_size`,
`max_seq_len`, `total_batch_size`, `dataloader_state_dict` (for exact-resume of the data reader,
though tinylab itself has no `--resume` flag today), `total_training_time`, and for SFT checkpoints,
`base_model_tag`/`base_model_step`. `tinylab.checkpoints.build_model` cross-checks
`tokenizer_fingerprint` against the currently-loaded tokenizer before returning a model — a vocab-
size match alone isn't enough to prove two tokenizers assign ids the same way.

## No derivation rules, not just no depth dial

`tinylab.presets` (a `"depth"` dial → concrete `ModelConfig` tree, one preset registered) is gone.
`tinylab.modelconfig` replaced it: a job file's `"model_config"` names a path to an already-
materialized tree — produced by nanochat's `scripts/model_info.py --dump-config`, since that's
where the preset registry and the depth-dial derivation rules (`mup_dims`, `compute_window_sizes`,
`gpt_lambda_schedule`, …) actually live. This isn't scope-trimming, it's the same rule
`modelcore/AGENTS.md` states for its own tree — "a config tree carries only concrete,
already-decided values, never a derivation rule" — applied to the whole job file, not just the
`ModelConfig` part of it.

The same rule removed `modelcore.scaling.derive_training_plan` from `tinylab.ops.train` too:
`target_flops`/`target_param_data_ratio` and the muP batch-size/weight-decay corrections they fed
are gone along with the reference-model machinery (`resolve_reference_config`,
`d_ref_scaling_params`) that made them work. `total_batch_size`, `num_iterations` (for `kind:
base`), and `eval_tokens` are now required job-file keys — compute them with nanochat's
`scripts/model_info.py --target-flops=... --json` and paste the numbers in. `world_size` is a new
required key for the same reason: a job file fixes the GPU count a run assumes (its
`total_batch_size`/grad-accum arithmetic depends on it), checked against the actual launch rather
than inferred from it. The literal defaults that remain (`matrix_lr: 0.02`, `warmup_steps: 5`, …)
are concrete values, not rules, and stay.

## What tinylab deliberately doesn't do

tinylab is the minimal host: one job file, four things it can do. Relative to `nanochat` (the
architecture-playground host built on the same three subsystems), tinylab now has fp8, doc-masking,
and LoRA/DoRA adapters (ported in — see `ops/train.py`'s `fp8`/`doc_masking`/`model_config`
adapter-override keys), but still no wandb logging, no fp16 `GradScaler`, no `--resume-from-step`,
no mid-training CORE/sample eval (only a final save), no periodic checkpointing, and no
`torch.compile` in its own training loop (see "Why the training loop is eager" below). None of
these are bugs — they're `nanochat`'s job, not tinylab's; adding one back means porting it the same
way everything else here was ported, not inventing it fresh.

## What one job file still can't replace: `runs/contest_d12.sh`

`jobs/contest.json` runs a two-architecture comparison (matching nanochat's
`runs/contest_d12.sh` rows) as one tinylab pipeline — every architecture nanochat can dump is now
reachable, which presets never allowed. It is not a drop-in replacement for the shell driver,
though:

- **No row matrix/sweep.** N architectures means N × 4 hand-written steps (prepare is shared, but
  each architecture needs its own base-train, sft-train, and two bench steps). There is no
  sweep/matrix/foreach construct anywhere in the job-file schema.
- **No preflight budget.** `--dry-run` only echoes resolved step JSON — no GPU-hours/cost estimate.
  Run nanochat's `scripts/model_info.py --json` per architecture before spending.
- **No results aggregation.** Each op returns a dict `job.run_file` prints as one JSON line; there
  is no `results.csv`, no joined comparison table.
- **No skip-if-done resume.** `contest.sh` greps its CSV to skip finished rows after an
  interruption; tinylab's only re-run granularity is `--only <step>`, one step at a time.

## Why the training loop is eager

`tinylab.ops.train`'s own forward/backward loop never calls `torch.compile` — deliberately, so a
job's very first training step can't stall inside a cold compiled-kernel cache the first time it
runs in a fresh environment (real, once-per-environment behavior on some backends, not a bug).
`modelcore.optim.MuonAdamW.step` is unconditionally compiled internally regardless — that's an
invariant owned by modelcore, not something tinylab's own loop controls or can opt out of.

## Where things come from

`tokenizer.py`, `checkpoints.py`, `engine.py`, `chat.py`, `data.py`, `runtime.py`, and
`ops/{prepare,train,bench}.py` are each ported from a corresponding file in
[`nanochat`](https://github.com/8kb/nanochat) (see each module's own docstring for exactly which
one) and trimmed to what a job-file-driven pipeline needs — `nanochat` is a virtual uv project (no
`[build-system]`) and can't be a real dependency, so this is a port, not an import.
`tests/test_no_nanochat.py` mechanically guards against an accidental `import nanochat` slipping in.

Several pieces that started as ports (byte-identical copies of nanochat code, since neither host
could depend on the other) have since moved into the subsystems all three repos already share,
once the duplication became visible with both ports in hand — modelcore has no host dependencies
at all, so this isn't a new coupling, just recognizing mechanism that belonged there from the
start:

- `runtime.py`'s device/DDP/seed bring-up (`compute_init`/`compute_cleanup`/
  `autodetect_device_type`) → `modelcore.runtime`.
- `ops/train.py`'s LR-multiplier/Muon-momentum schedule shapes → `modelcore.optim.schedules`
  (the param-group mutation itself is `ModelManager.apply_schedule`, since it touches the
  optimizer's own on-disk format). Scaling-law horizon derivation (`derive_training_plan`/
  `TrainingPlan`/`B_REF`) lives in `modelcore.scaling` too, but tinylab no longer calls it at all —
  see "No derivation rules, not just no depth dial" below.
- `checkpoints.py`'s `meta.json` merge and `model_<step>.pt` step scan → `modelcore.store`'s
  `FileSystemStore.update_meta`/`last_step`.
- `engine.py`'s tool-use decode loop (`RowState`, the forced-token deque, the tool start/end state
  machine) → `modelcore.generate.generate_with_tools`/`collect_batch`, driven by a `ToolSpec` whose
  `run=` is `Engine._run_calculator` — `use_calculator` itself (the `eval()` sandbox) stays here,
  it's the one thing that's actually this repo's own.
- `ops/train.py`'s dataset-open validation → `DataManager.open(..., expect_sequence_len=,
  expect_fingerprint=)`, raising `datacore.DatasetMismatch`.
- `ops/prepare.py`'s `TaskMixtureTokenSource` → `datacore.ExampleTokenSource(mixture, render=...,
  name=...)`; its `_Truncated` wrapper is gone entirely -- `datacore.ExampleMixture`'s own `stop=`
  kwarg already clamps to the true length.
- `ops/bench.py`'s chat-task-name → task-class dict → `benchcore.build_chat_tasks`; its CORE-suite
  load-then-score → `BenchManager.core_suite`.

What's left in each of these files is what's actually specific to tinylab: naming/tag policy,
job-file key handling, the cosine weight-decay schedule (no modelcore equivalent), and the
mixture/corpus recipes themselves.
