# Architecture

How a job file becomes a run, what each module owns, and where tinylab's own code ends and a
subsystem's begins.

```
tinylab/
  __main__.py           the CLI: parses argv, dispatches to job.run_file, chat.main, info.main or remote_cli
  job.py                loads a job file, deep-merges "defaults", validates keys, runs steps in order
  runtime.py             base dir, device/DDP init, print0
  tokenizer.py           RustBPETokenizer: BPE encode/decode, special tokens, conversation rendering
  default_tokenizer/     committed tokenizer.pkl (a fixed, pre-trained vocab)
  modelconfig.py          loads + validates a materialized modelcore.ModelConfig tree (no
                          preset/depth-dial derivation -- that's llmllab/tools/, see below)
  info.py                 `python -m tinylab info`: read-only params/FLOPs/KV/plan report for a
                          config file or a checkpoint tag (meta only, no weights)
  checkpoints.py         checkpoint tag/step naming over modelcore's FileSystemStore
  remote.py              the HF bucket: Remote/Uploader/Prefetcher, push/pull rules (docs/remote.md)
  remote_stores.py       datacore DatasetStore wrappers that upload / stream shards
  remote_cli.py          `python -m tinylab remote ls|pull|push|check|rm|push-docs`
  readme.py              the per-entity README.md format (code-owned frontmatter + History)
  engine.py               KV-cached generation + calculator tool use
  chat.py                 the `chat` CLI command
  data.py                 corpus identity: ClimbMix shard URLs, SmolTalk
  ops/
    __init__.py            OPS registry + Context (device/managers, resume flag, passed explicitly)
    prepare.py              op: prepare
    train.py                op: train (kind base|sft, one shared loop)
    bench.py                op: bench (suite core|chat|bpb|sample|infer|tokenizer)
    rl.py                    op: rl -- REINFORCE on GSM8K from an sft checkpoint
    tokenizer.py             op: tokenizer -- trains a fresh BPE vocab
jobs/
  smoke.json              tiny end-to-end pipeline, runs on a laptop in minutes
  speedrun.json            real-scale template for a multi-GPU pod
  contest.json             two-architecture comparison (the rows of our nanochat fork's runs/contest_d12.sh)
  configs/                 materialized ModelConfig trees the job files above point at, generated
                          by llmllab/tools/make_config.py (see AGENTS.md)
```

## From a job file to a run

```
__main__.main()
  -> job.run_file(path, only=..., dry_run=..., resume=...)
       -> job.load(path)                 # parse JSON, check top-level keys
       -> job.resolve_steps(job, only)   # deep-merge defaults, validate each step's keys
       -> [job state file dance -- see "Resume: the job state file" below]
       -> for each step: OPS[step["op"]].run(step, ctx)
```

`resolve_steps` deep-merges `job["defaults"]` into every step, then validates the merged result
against that op's own `accepted_keys(cfg)` — a *function*, not a static set, because `prepare` and
`train` accept a different key set depending on the step's own `"kind"` (`"base"` vs `"sft"`), and
`bench` depending on `"suite"` (`core`, `chat`, `bpb`, `sample`, `infer`, `tokenizer`). A key that's real for the wrong kind (e.g.
`"mmlu_epochs"` on a `kind: base` prepare step) is caught the same way a genuinely unknown key is —
see `docs/job-file.md` for the full key reference this validates against.

`chat.main()` follows the identical `job.load` -> `job.resolve_chat` -> `job.check_known_keys`
shape, but is never reachable through `OPS` — it's a separate CLI command (`python -m tinylab
chat <job.json>`), not a pipeline step. A pipeline step runs to completion unattended; a REPL
blocks on a human at a prompt, so it can't be one.

## Resume: the job state file

`--resume` is a CLI flag (`run_file`'s `resume` param), never a job-file key — whether a given
invocation is a fresh start or a continuation is a fact about the invocation, not the pipeline (see
`job.py`'s own module docstring). It drives two independent things, both backed by one job state
file under `<base_dir>/job_state/` (named from the job file's own path, so two different job files
never collide):

- **Pipeline-level: skip whichever steps already completed.** A real (`not dry_run`), full-pipeline
  (`only is None`) run tracks, per step, `{"status": "done"}` once that step's `run()` returns
  without raising — a later `--resume` skips it outright. Starting a job whose state file already
  exists *without* `--resume` is a hard, immediate `JobError` — a previous run may still be alive,
  or died without cleanup — pointing at `--resume` (to continue) or deleting the state file (to
  force a fresh start). `--resume` with no state file present is a no-op (an ordinary fresh run),
  so it's safe to pass unconditionally from an unattended restart script. `--only` bypasses this
  entire mechanism — a deliberate single-step override never touches the state file, even while
  one from a previous full-pipeline run exists.
- **Step-level: a `train` step continues its own interrupted run — from a checkpoint the state
  file itself confirms, not a directory guess.** `Context.resume` (set from the same flag) reaches
  every op; `ops/train.py`'s `run()` checks it first, before its usual kind-specific fresh-start
  path. Whenever a periodic checkpoint's save (model + optimizer + meta, every rank) fully returns,
  `ctx.record_checkpoint(step_name, step)` records `{"status": "in_progress", "checkpoint_step":
  step}` for that step — *before* `--resume` is even a question, this is what makes resume possible
  at all. On `--resume`, `ops/train.py` asks `ctx.resume_checkpoint_step(step_name)` for that exact
  value and loads precisely that checkpoint — it never scans the checkpoint directory and guesses
  the highest step number, because a directory scan can't tell a fully-written checkpoint from one
  that started writing and crashed mid-`torch.save` (still matches the `model_<step>.pt` naming
  pattern, so a scan would happily pick it and then fail to load, even with a perfectly good,
  slightly older checkpoint sitting right next to it). The logic is `checkpoints.resume_point`, shared
  by `train` and `rl`. A directory scan (`checkpoints.find_last_step`) is only a fallback for a call
  not driven through `job.run_file` at all (a bare `Context`, as this repo's own tests use). In a
  job-driven run, `--resume` with *no* record for a step but a checkpoint already on local disk
  (the state file was deleted when an earlier run finished, or the step was added since) is a hard
  error: whether that checkpoint is finished or half-written is unknown, so nothing is guessed — delete
  it, give the step another `output_tag`, or run just that step with `--only`. With a `remote`, a clean
  disk first pulls the newest resumable step from the bucket, which is not an error. Either way, resume only has something to continue from if
  the step had already taken at least one periodic checkpoint (`"save_every"`, see
  `docs/job-file.md`) — without one, the step just restarts, which is the best any resume can do. A
  `world_size` mismatch against the checkpoint's own recorded value is a hard error (`MuonAdamW`'s
  optimizer state doesn't reshard across `world_size`), and so is a missing
  optimizer shard for this rank; resume never silently falls back to a fresh optimizer.

### The state file's own crash-safety

The state file update above — after every periodic checkpoint, and after every whole step — has to
survive a crash *during its own write*, or the fix just moves the corruption problem one file over.
Every write (`job._write_state`) goes through: serialize the new content to a temp file, `flush()` +
`os.fsync()` it (durable on disk before anything else happens), delete the previous `.old`
generation if one exists, atomically rename the current file to `.old` (`os.replace`, which works
identically on POSIX and Windows, unlike bare `os.rename`), then atomically rename the temp file to
current. A crash at any point in that sequence leaves either the untouched previous file or the
untouched, already-durable `.old` readable — never a torn write, regardless of how reliable the
filesystem's rename semantics actually are (matters on network/overlay storage, which a GPU pod's
mount can be). Reading (`job._read_state`) tries the current file first, falls back to `.old` if
the current one is missing or fails to parse, and returns `None` (treated as "nothing to resume
from", not an error) if neither is usable. A monotonic `"counter"` field increments on every write,
mainly so a human inspecting the file by hand can tell at a glance whether `.old` is one generation
behind or something's actually gone wrong — the rotation order alone already guarantees which file
is newer, no comparison needed for the read logic itself.

**What this does not cover**: `modelcore.store.FileSystemStore`'s own writes (the actual
`model_<step>.pt`/`optim_<step>_rank<r>.pt`/`meta_<step>.json` files) are not atomic — a separate
repo, its own versioning discipline, out of scope here. The state file mechanism above closes the
gap for *knowing which checkpoint step to trust*; it doesn't make an individual checkpoint file's
own write crash-safe. A directory-scan-only resume (the no-state-file fallback path) is still
exposed to that gap.

## `Context`

`tinylab.ops.Context` carries the device, rank/world_size, and lazily-built `ModelManager`/
`DataManager`/`BenchManager`/tokenizer singletons for one run. It's constructed once in
`job.run_file` and passed explicitly to every op's `run(cfg, ctx)` — never a module-level global,
matching the subsystems' own no-ambient-globals rule (see
[`llmllab/docs/subsystem-conventions.md`](../../llmllab/docs/subsystem-conventions.md), which
resolves in a local checkout of the whole family). This is what lets a
multi-step job share one device/process-group setup instead of re-initializing it per step. It also
carries the resume hooks (`resume_checkpoint_step`/`record_checkpoint`) `job.run_file` wires up to
the job state file — see "Resume: the job state file" above.

## Where each subsystem's boundary sits

- **`modelcore.ModelManager`** creates, loads, and saves models and optimizers, and computes their
  FLOPs/param/KV-cache stats. `tinylab.modelconfig` loads and validates the materialized
  `ModelConfig` tree a job file's `"model_config"` names — tinylab has no depth-dial layer of its
  own; the tree is produced entirely by `llmllab/tools/make_config.py` (or hand-
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
  model and a tokenizer. `Engine.generate_batch_multi` is the optional half of that protocol:
  several *different* prompts decoded together in one right-ragged batch (`modelcore.generate.
  collect_batch_multi`) rather than one problem at a time. `bench`'s `suite: "chat"` step always
  passes a real `Engine`, so setting the job key `generative_batch_size` above `1` is what turns
  batching on for GSM8K/HumanEval — identical results at `temperature: 0`, and the point of it:
  uncapped GSM8K (~1319 problems) and HumanEval (~164) one-at-a-time can cost more than training
  the model did (see `llmllab/docs/history.md`'s Stage notes on `chat_eval`'s `-B` flag, the same
  mechanism this ports).
- **`tinylab.checkpoints`** owns tag/step naming and `meta.json`'s extra fields on top of
  `modelcore.store.FileSystemStore`, which owns the actual model/optimizer artifact format.

## On-disk layout

Everything lives under `<base_dir>` (`~/.cache/tinylab/`, or `TINYLAB_BASE_DIR` to override) —
separate from our nanochat fork's `~/.cache/nanochat/`, even though the two produce identical token ids
with a compatible tokenizer.

```
<base_dir>/
  tokenizers/<name>/         tokenizer.pkl. Any number side by side; "default" is
                             copied from the bundled vocab on first use, every other name comes
                             from a "tokenizer" step
  base_data_climbmix/        downloaded ClimbMix parquet shards
  prepared/<dataset_name>/   a datacore dataset: packed sequences + manifest
  checkpoints/<tag>/         model_<step>.pt, meta_<step>.json (its own "model_config" key holds
                             the tree -- there is no separate config_<step>.json file),
                             optim_<step>_rank<r>.pt; multiple steps coexist, none pruned locally.
                             <tag> is 1-4 "/"-joined names, each 1-16 chars from [A-Za-z0-9_-]
                             ("gpt-d12-base", "kvcache/d13-chat")
  job_state/                 one state file (+.old/.tmp) per in-progress or crashed job run -- see "Resume" above
  experiments/<experiment>/  a job's required "experiment" key (1-64 chars from [A-Za-z0-9_-], the
                             name of the experiment card's folder in git, or "scratch"; no built-in
                             default -- see docs/job-file.md). logs/<job_name>.log is the whole
                             run's general log (start/finish, each step's own header/result, a
                             top-level failure, the final job-state dump); logs/<job_name>-
                             <step_name>.log is one executed step's own console output, nothing
                             else. <job_name> is the job file's own basename, no directory or
                             extension. With a "remote" set, jobs/, configs/ and results/ siblings
                             hold snapshots and step results -- see docs/remote.md.
```

### Checkpoint tags: one namespace, the tag is the address

There used to be `base_checkpoints/` and `chatsft_checkpoints/`, chosen by a `source` job key. Two
kinds today, more later — a new directory (and a new `source` value) per kind doesn't scale — so
there is now one `checkpoints/` directory and the *tag* says what a checkpoint is. `output_tag`,
`source_tag` and `model_tag` are each a complete address; `tinylab.checkpoints.resolve_checkpoint_dir`
turns one into a path and `validate_tag` rejects anything that could escape it (empty, absolute,
backslash, an empty/`.`/`..` segment). `job.resolve_steps` runs the same check up front, so a bad tag
is a `JobError` before a `prepare` step has spent an hour, not a `ValueError` after.

One consequence to know: base and sft used to be able to share a tag because they lived in different
directories (our nanochat fork's convention reuses `d12` for both). They can't now, so an sft step whose
`output_tag` equals its own `source_tag` is refused outright, rather than reading its starting weights
from, and writing its result into, the same directory.

### sft's optimizer: `init_lr_frac` and `load_optimizer` (momentum warm-start)

Ported from our nanochat fork's `chat_sft.py` (`--init-lr-frac`/`--load-optimizer`), both **sft-only**
job-file keys (`_SFT_KEYS`, not accepted on `kind: base`). A fresh `kind: sft` step:

1. Builds a normal, cold optimizer from this step's own `OptimizerHparams` (same as `base`).
2. `load_optimizer` (default `true`): loads `source_tag`'s own last-saved optimizer shard for this
   rank via `checkpoints.load_optimizer_state`, then immediately restores every param group's `lr`
   back to what step 1 computed — `load_state_dict` overwrites a group's whole metadata dict
   (`lr`, `initial_lr`, ...), and only the momentum/`exp_avg` buffers are meant to carry over, not
   the source run's own (usually warmed-down-to-near-zero) LRs. Skipped, with a printed reason,
   when the model has `adapters`: an adapter-augmented model's param groups are shaped completely
   differently from the base checkpoint's (`modelcore.roles.build_param_groups`), so the shard
   would apply momentum state to the wrong parameters. A world_size mismatch or a missing shard is
   a hard error (same stance as the resume path's own optimizer checks) — set `load_optimizer:
   false` to start with a fresh optimizer instead.
3. `init_lr_frac` (default `0.8`): scales every group's now-current `lr` down by this fraction and
   restates `initial_lr` to match — load-bearing beyond the sft case, since `ModelManager.
   apply_schedule` computes every step's LR as `group["initial_lr"] * lr_mult`.

Both steps happen only on a **genuine fresh start** — a resumed step (`ctx.resume` finding this
step's own prior checkpoint) restores its optimizer state exactly as saved, with neither the
warm-start nor the rescale re-applied; conflating the two would silently rescale an already-scaled
LR on every resume. This is why the code carries the source-tag warm-start and the resumed-step
restore as two separate locals (`warm_start_optimizer_state` vs `optimizer_state`) even though both
eventually reach the same `optimizer.load_state_dict` call site.

What's deliberately **not** ported: our nanochat fork's hyperparameter *inheritance* (`chat_sft.py` reads
`embedding_lr`/`unembedding_lr`/`matrix_lr` back out of the pretrain checkpoint's own
`user_config` when the CLI doesn't override them). tinylab has no "inherit a value from a
checkpoint" mechanism anywhere else (a job file names every value it wants — see this module's own
docstring), so `unembedding_lr`'s sft default is a literal `0.008`, matching what our nanochat fork's
inheritance actually resolves to for a checkpoint pretrained with `base_train.py`'s own default,
not a new inheritance rule.

### Several tokenizers

A job's `"tokenizer"` (a `COMMON_KEYS` entry) is a bare name — `<base_dir>/tokenizers/<name>/` —
or, if it contains a path separator, a path (relative ones made absolute against the job file's
own directory in `job.resolve_steps`, the same rule `model_config` follows). `Context.tokenizer_for
(spec)` loads and memoizes one instance *per resolved directory*, so two specs naming the same
directory share one loaded instance.

Unlike `"device"`, `"tokenizer"` is resolved **per step**, not once for the whole run:
`ops.{prepare,train,bench}.run` each call `ctx.tokenizer_for(cfg.get("tokenizer"))` themselves,
falling back to the run's default (`Context.tokenizer`/`tokenizer_spec`, still read off the first
resolved step, same as `"device"`) only when a step doesn't set its own. This is what lets one job
train several differently-vocabbed models side by side — a `prepare`/`train`/`bench` step for each
tokenizer, rather than one job file per tokenizer (see docs/job-file.md's "Training two
differently-vocabbed models"). It's also what keeps the dataset↔model fingerprint agreement a run
depends on: a step's dataset (via `prepare.default_dataset_name`) and its model are checked against
*that step's own* tokenizer, whichever one it names, not against some other step's.

Only the `default` tokenizer is ever created for you (copied from the bundled vocab). A missing
*named* one raises instead of copying the bundled vocab under that name: the two would be
fingerprint-identical, so no check downstream could tell the user got the wrong tokenizer.

A checkpoint records which tokenizer it was trained with — `ops.train` writes
`model_config["tokenizer"] = {"name", "fingerprint", "vocab_size", "special_tokens"}` via
`Context.tokenizer_name_for(spec)` (modelcore carries that block opaquely; see its own docs), naming
*that train step's own* tokenizer, not the run's default — and `checkpoints.build_model` resolves
the tokenizer to load in order: an explicit `tokenizer_spec` argument (the caller's own step's
`"tokenizer"`, if set) if given, else that recorded name, else the default. The existing vocab-size
assert and fingerprint check then catch a wrong *selection*.

`ops.train` also stamps `template`: a `kind: sft` checkpoint declares `"nanochat"` (the chat format
`tinylab.tokenizer.render_conversation` renders), a base one keeps whatever its `model_config` said.
Declarative only for now — modelcore validates it is a known template and nothing else reads it.

`meta_<step>.json` carries: `step`, `val_bpb`, `min_val_bpb` (the run's best `val_bpb` so far, across
resumes), `smooth_train_loss` (an EMA of the per-step train loss, across resumes),
`tokenizer_fingerprint`, `model_config` (the
materialized tree this checkpoint was built from), `user_config` (the resolved job-file step, minus
`"model_config"` — that would just duplicate the sibling `model_config` key under a different,
unresolved shape), `device_batch_size`, `max_seq_len`, `total_batch_size`, `dataloader_state_dict`
(a datacore resume cursor -- see "Resume" above), `total_training_time`, and for SFT checkpoints,
`base_model_tag`/`base_model_step`. `model_config` is a `modelcore.v3` tree, including its `template`
and `tokenizer` blocks. `tinylab.checkpoints.build_model` cross-checks
`tokenizer_fingerprint` against the currently-loaded tokenizer before returning a model — a vocab-
size match alone isn't enough to prove two tokenizers assign ids the same way.

## No derivation rules, not just no depth dial

`tinylab.presets` (a `"depth"` dial → concrete `ModelConfig` tree, one preset registered) is gone.
`tinylab.modelconfig` replaced it: a job file's `"model_config"` names a path to an already-
materialized tree — produced by `llmllab/tools/make_config.py`, since that's
where the preset registry and the depth-dial derivation rules (`mup_dims`, `compute_window_sizes`,
`gpt_lambda_schedule`, …) actually live. This isn't scope-trimming, it's the same rule
`modelcore/AGENTS.md` states for its own tree — "a config tree carries only concrete,
already-decided values, never a derivation rule" — applied to the whole job file, not just the
`ModelConfig` part of it.

The same rule removed `modelcore.scaling.derive_training_plan` from `tinylab.ops.train` too:
`target_flops`/`target_param_data_ratio` and the muP batch-size/weight-decay corrections they fed
are gone along with the reference-model machinery (`resolve_reference_config`,
`d_ref_scaling_params`) that made them work. `total_batch_size`, `num_iterations` (for `kind:
base`), and `eval_tokens` are now required job-file keys — compute them with
`python -m tinylab info <config> --target-flops=... --json` (the one place the plan math runs
here: a report that reads `modelcore.scaling.derive_training_plan`, never written into a job file)
and paste the numbers in. `world_size` is a new
required key for the same reason: a job file fixes the GPU count a run assumes (its
`total_batch_size`/grad-accum arithmetic depends on it), checked against the actual launch rather
than inferred from it. The literal defaults that remain (`matrix_lr: 0.02`, `warmup_steps: 5`, …)
are concrete values, not rules, and stay.

## What tinylab deliberately doesn't do

tinylab is the minimal host: one job file, five ops (`prepare`, `train`, `bench`, `tokenizer`, `rl`)
plus the `chat`, `info` and `remote` commands. Relative to our nanochat fork (the
archived architecture-playground host that preceded it), tinylab has fp8, doc-masking,
LoRA/DoRA adapters (ported in — see `ops/train.py`'s `fp8`/`doc_masking`/`model_config`
adapter-override keys), periodic checkpointing and resume (`"save_every"`, `--resume` — see
"Resume: the job state file" above), tokenizer training (the `tokenizer` op), and `torch.compile`
in its own training loop (see "Why the training loop is compiled" below), but still no wandb
logging, no fp16 `GradScaler`, and no mid-training CORE/sample eval (only periodic val-bpb). None of
these are bugs — they were left behind in our archived nanochat fork; adding one back means porting it
the same way everything else here was ported, not inventing it fresh. What was ported late, when
our nanochat fork was retired: the `rl` op, the `bpb`/`sample`/`infer`/`tokenizer` bench suites, and the
`info` inspector (see `llmllab/docs/history.md`).

## What one job file still can't replace: `runs/contest_d12.sh`

`jobs/contest.json` runs a two-architecture comparison (matching our archived nanochat fork's
`runs/contest_d12.sh` rows) as one tinylab pipeline — every architecture `make_config.py` can
generate (or anyone can write by hand) is reachable, which presets never allowed. It is not a drop-in replacement for the shell driver,
though:

- **No row matrix/sweep.** N architectures means N × 4 hand-written steps (prepare is shared, but
  each architecture needs its own base-train, sft-train, and two bench steps). There is no
  sweep/matrix/foreach construct anywhere in the job-file schema.
- **No preflight budget.** `--dry-run` only echoes resolved step JSON — no GPU-hours/cost estimate.
  Run `python -m tinylab info <config> --gpu ... --json` per architecture before spending.
- **No results aggregation.** Each op returns a dict `job.run_file` prints as one JSON line; there
  is no `results.csv`, no joined comparison table.

`--resume` (see "Resume: the job state file" above) does cover skip-if-done at the step level now —
close to what `contest.sh`'s CSV-grep achieves, though `contest.sh` skips finished *architecture
rows* (a coarser unit than tinylab's individual steps) and additionally resumes a `train` step's
own interrupted training loop, not just whole-step granularity.

## Why the training loop is compiled

`tinylab.ops.train`'s forward/backward loop calls `torch.compile(model, dynamic=False)` right
after fp8 conversion (ordering matters -- fp8 must wrap the model's Linears first, as in our nanochat fork's
`base_train.py`), using an `orig_model`
reference (uncompiled) for the optimizer and every checkpoint save -- a compiled module's
`state_dict()` keys gain an `_orig_mod.` prefix otherwise (our nanochat fork's checkpoint manager had the same
strip-hack). `modelcore.optim.MuonAdamW.step` is additionally,
unconditionally compiled internally regardless -- that's an invariant owned by modelcore, not
something tinylab's own loop controls or can opt out of.

This was deliberately *not* the case at first, so that a job's very first training step couldn't
stall inside a cold compiled-kernel cache the first time it runs in a fresh environment (real,
once-per-environment behavior on some backends, not a bug). Reversed after a real measurement on
experiment 01 (`llmllab/experiments/01-ffn-width-vs-heads`, 1x H200 SXM, d12 GPT, fp8): the eager
loop measured ~11% MFU (3.67s/step) against a documented ~40% planning assumption; adding the
compile call raised it to ~47% MFU (0.86s/step steady-state) -- a 4.27x speedup. The stall is real but one-time and bounded (~80s measured for this
model/GPU, once, at step 0) -- not worth paying roughly 4x the wall-clock and $ on every subsequent
step, on every real run this host exists to make cheap and unattended, to avoid it. `pytest`'s own
CPU run pays a version of this cost too (Inductor's CPU backend, not CUDA-specific), but only once
per machine: Inductor persists compiled artifacts to an on-disk cache keyed by graph/shape, so a
cold-cache suite run took ~28s (vs. ~7s before this change) and every run after that was back down
to ~7-8s, reusing the cache. Comfortably fast enough either way, not worth special-casing off for
tests.

## Where things come from

Most modules are ported from the corresponding file of our nanochat fork (archived, see
`llmllab/docs/history.md`; each module's docstring has the one-line provenance) and trimmed to what a
job-file-driven pipeline needs. The fork was a virtual uv project and could never be a dependency, so
this is a port, not an import; `tests/test_no_nanochat.py` mechanically guards against an accidental
`import nanochat` slipping in. `tokenizer.py`'s `train_from_iterator`/`save` (via
`ops/tokenizer.py`) are the one place this repo depends on `rustbpe` directly.

What the three subsystems own instead of tinylab:

- device/DDP/seed bring-up (`compute_init`/`compute_cleanup`/`autodetect_device_type`) →
  `modelcore.runtime`;
- the LR-multiplier/Muon-momentum schedule shapes → `modelcore.optim.schedules` (the param-group
  mutation itself is `ModelManager.apply_schedule`, since it touches the optimizer's own on-disk
  format); scaling-law horizon derivation (`derive_training_plan`) → `modelcore.scaling`, read only by
  the `info` report;
- the `meta.json` merge and `model_<step>.pt` step scan → `modelcore.store`;
- the tool-use decode loop (`RowState`, the forced-token deque, the tool state machine) →
  `modelcore.generate.generate_with_tools`/`collect_batch`, driven by a `ToolSpec` whose `run=` is
  `Engine._run_calculator` (the `eval()` sandbox stays here);
- dataset-open validation → `DataManager.open(..., expect_sequence_len=, expect_fingerprint=)`;
- the mixture-to-token-source glue → `datacore.ExampleTokenSource`; capping a mixture →
  `ExampleMixture(stop=)`;
- the chat-task registry → `benchcore.build_chat_tasks`; the CORE load-then-score →
  `BenchManager.core_suite`.

What's left in tinylab is what is specific to a host: naming/tag policy, job-file key handling, and
the mixture/corpus recipes.
