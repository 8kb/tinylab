# AGENTS.md

Non-obvious invariants for anyone (human or agent) working in this repo. Read
[`docs/architecture.md`](docs/architecture.md) first for the module map and how a job file becomes
a run, and [`docs/job-file.md`](docs/job-file.md) for the full key reference — this file covers
what isn't obvious from either. Read [`../llmllab/AGENTS.md`](../llmllab/AGENTS.md) for conventions
that apply across the whole family (style, docs-placement rules), and each subsystem's own
`AGENTS.md`/`docs/architecture.md` — [modelcore](https://github.com/8kb/modelcore),
[datacore](https://github.com/8kb/datacore), [benchcore](https://github.com/8kb/benchcore) — before
touching the code that calls into it. (The `../llmllab/...` links in this file resolve in a local
checkout of the whole family; `llmllab` has no public remote yet, so they won't resolve for someone
reading this on GitHub — the modelcore/datacore/benchcore links are real GitHub URLs and work
anywhere.)

`modelcore`/`datacore`/`benchcore` are pinned by git tag in `pyproject.toml`'s
`[tool.uv.sources]`: `modelcore` `v0.3.0`, `datacore` `v0.2.1`, `benchcore` `v0.1.1`.

## Invariants owned elsewhere

- **A config tree carries only concrete, already-decided values, never a derivation rule.** This
  is modelcore's invariant for `ModelConfig`, but tinylab applies it to the whole job file: there
  is no `presets.py` here (deleted — depth-dial derivation, muP scaling-law horizon/batch-size/LR
  math, all of it) and never will be. A job file's `"model_config"` names a path to an
  already-materialized tree; every number tinylab used to derive (`total_batch_size`,
  `num_iterations`, …) is a required key instead. Presets and the training-plan math still exist —
  they're nanochat's job (`nanochat/architectures/`, `modelcore.scaling`), not tinylab's. Compute a
  number with nanochat's `scripts/model_info.py --dump-config` / `--target-flops=... --json` and
  paste it into the job file; don't add a derivation rule back into tinylab to avoid that step. See
  [modelcore/AGENTS.md](https://github.com/8kb/modelcore/blob/main/AGENTS.md) and
  [docs/architecture.md](docs/architecture.md#no-derivation-rules-not-just-no-depth-dial).
- **Optimizer state is checkpointed and reloaded positionally** — the role-order
  `ModelManager.create_optimizer` builds internally is part of the on-disk format; never
  reconstruct a param-group list by hand.
- **datacore doesn't compare tokenizer identity itself** — a prepared dataset's
  `tokenizer_fingerprint` is host-checked, never datacore's job. See
  [datacore/AGENTS.md](https://github.com/8kb/datacore/blob/main/AGENTS.md).
- **A model already satisfies `benchcore.Model`; a `Generator` doesn't come for free.**
  `tinylab.engine.Engine.generate_batch` is that adapter — see
  [benchcore/AGENTS.md](https://github.com/8kb/benchcore/blob/main/AGENTS.md)'s `protocols.py`.

## Invariants that will bite you (tinylab's own)

- **`nanochat` cannot be a real dependency.** It's a *virtual* uv project (no `[build-system]` in
  its `pyproject.toml`), so it's never built or installable. Everything tinylab needs from it is
  **ported, not imported** — each ported file's docstring names its origin.
  `tests/test_no_nanochat.py` mechanically guards against an accidental `import nanochat` slipping
  in.
- **tinylab keeps its own cache directory**, `~/.cache/tinylab/` (`TINYLAB_BASE_DIR` to override) —
  separate from nanochat's, even though the ported tokenizer produces identical token ids. A
  prepared dataset, checkpoint, or downloaded shard from one is never read by the other.
- **An unrecognized job-file key is a hard error, with a close-match suggestion when there is
  one.** `prepare`/`train`/`bench` each accept a different key set depending on their step's own `"kind"`/`"suite"` (see
  each module's `accepted_keys(cfg)`), so a key that's real for one kind but nonsensical for
  another is also caught, not silently ignored. See `docs/job-file.md` for the full reference and
  `tests/test_docs.py` for the guard keeping it accurate.
- **`chat` is a separate CLI command, never a job op.** A pipeline step must run to completion
  unattended; a REPL blocks on a human at a prompt. Don't add `"chat"` to `tinylab.ops.OPS` —
  `tests/test_job.py` pins this (a step with `"op": "chat"` is rejected as an unknown op).
- **A `train` step's output checkpoint tag defaults to the step's own `"name"`.** This is what
  lets an `"sft"` step's `"source_tag": "pre"` and a later `bench` step's `"model_tag": "sft"`
  reference each other by the pipeline's own step names, with no separate tag field to keep in
  sync — see `docs/job-file.md`'s glossary.
- **`prepare` and `train` share one `"sequence_len"`, not a per-op field**, precisely because a
  prepared dataset's `sequence_len` is fixed at prepare time and a training step must match it
  exactly (`tinylab.ops.train._open_dataset` raises if they disagree) — put it once, in
  `"defaults"`.
- **A `bench` step's `"max_per_task"` must leave enough examples for every CORE task's own
  few-shot count**, or benchcore's few-shot sampling raises a `ValueError` (it draws from the
  *other* examples in the same truncated task, so a `max_per_task` of N leaves a pool of only
  N - 1). Owned by benchcore, not tinylab — leave `"max_per_task"` unset for a real (non-smoke)
  CORE run.
- **A CORE `bench` step needs a model trained with enough `"sequence_len"` headroom for its
  few-shot prompts, not just enough for training itself.** A rendered CORE prompt (up to ~10
  few-shot examples concatenated) can run past 2,000 tokens; modelcore's rotary embedding cache is
  sized at `sequence_len * 10` when the model is built, and exceeding it is a hard error, not a
  truncation. `jobs/smoke.json` trains at `sequence_len: 2048` for exactly this reason, even
  though the model itself is tiny (`configs/gpt_d4.json`, a 4-layer tree).
- **`train`'s own forward/backward loop is eager — no `torch.compile`.** This keeps a job's very
  first step from being the one that first triggers a cold compiled-kernel cache on whatever
  backend it's running on. `modelcore.optim.MuonAdamW.step` is compiled regardless, internally —
  tinylab's own loop doesn't control or opt out of that. See `docs/architecture.md`.
- **`train` never warm-starts its optimizer** — every step builds a fresh one
  (`ModelManager.create_optimizer`), unlike nanochat's `chat_sft.py --load-optimizer`. This is why
  attaching an adapter via a `kind: sft` step's `"model_config"` override needs no
  matching guard here: nanochat forces `--load-optimizer` off for `--adapters` because a frozen
  base's param groups are shaped completely differently from a fully-trainable one, and loading a
  pretrained optimizer shard into that layout would corrupt momentum state — a failure mode that
  simply can't occur when there's no warm-start to begin with.
- **A `train` step's `"world_size"` is checked against the actual launch, not derived from it.**
  The job file fixes the GPU count a run assumes (`total_batch_size`/grad-accum arithmetic depends
  on it); a mismatched `torchrun --nproc_per_node` is a hard error, not a silently different
  effective batch size. Edit the job file when the GPU configuration changes.

## Testing

```bash
uv run pytest -q                 # everything: a couple of seconds, incl. one real training run
uv run pytest -q -m "not slow"   # skip that one real run: well under a second
```

`tests/conftest.py` has the one shared `base_dir` fixture (isolates `TINYLAB_BASE_DIR` to a
tmp_path) that most tests use. `test_smoke.py` (marked `slow`) is the only test that trains
anything real — a few steps on a synthetic in-memory corpus, no network — and is what proves
`train` -> `checkpoints` -> `engine` actually works together, not just each in isolation.

A change here that a subsystem's own suite wouldn't catch (e.g. a new op, a job-schema change)
needs `uv run pytest -q` green against the *pinned* install (not an editable local sibling
checkout) before it's considered landed — see
[`llmllab/docs/subsystem-conventions.md`](../llmllab/docs/subsystem-conventions.md)'s tag-bump rule
for why that distinction matters when bumping a pin.

## Style

See [llmllab/AGENTS.md](../llmllab/AGENTS.md#style).
