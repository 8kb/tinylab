# AGENTS.md

tinylab is the second host application in the llmllab family, alongside `nanochat` -- the same
four things a language-model lab does (prepare data, train, chat, bench) over the three standalone
subsystems, with every task declared in one JSON job file instead of CLI flags. Read
[`../llmllab/AGENTS.md`](../llmllab/AGENTS.md) first for conventions that apply across the whole
family, then each subsystem's own `AGENTS.md` + `docs/architecture.md` (linked below) before
touching the code that calls into them. This file covers what's specific to tinylab itself.

Note on the `../llmllab/...` and `../nanochat/...` links in this file: they resolve in a local
checkout of the whole family (this repo's own dev machine has one), but `llmllab` has no public
remote yet, so they won't resolve for someone reading this on GitHub. The modelcore/datacore/
benchcore links below are real GitHub URLs and work anywhere.

`modelcore`/`datacore`/`benchcore` are pinned by git tag in `pyproject.toml`'s
`[tool.uv.sources]`, exactly as nanochat pins them -- see their repos:
[modelcore](https://github.com/8kb/modelcore) `v0.3.0`,
[datacore](https://github.com/8kb/datacore) `v0.2.1`,
[benchcore](https://github.com/8kb/benchcore) `v0.1.1`.

## Repo map

```
tinylab/
  __main__.py           the entire CLI (two commands, no other flags anywhere)
  job.py                job-file loading, defaults deep-merge, key validation, dispatch
  runtime.py            base dir, device/DDP init, print0                 (from nanochat/common.py)
  tokenizer.py          RustBPETokenizer, inference-only                  (from nanochat/tokenizer.py)
  default_tokenizer/    committed tokenizer.pkl + token_bytes.pt          (byte copy of nanochat's)
  presets.py            arch presets + depth-dial derivation, merged      (from architectures/{presets,derive}.py)
  checkpoints.py        tag/step naming over modelcore's FileSystemStore  (from checkpoint_manager.py)
  engine.py             KV-cached generation + calculator tool use        (from nanochat/engine.py)
  chat.py               the `chat` CLI command -- NOT a job op
  data.py               corpus identity: ClimbMix shards, SmolTalk        (from dataset.py + sft_data.py)
  ops/
    __init__.py          OPS registry + Context (device/managers, explicit, never a global)
    prepare.py           op: prepare
    train.py             op: train  (kind base|sft, one shared loop; scaling math from scaling.py)
    bench.py             op: bench  (suite core|chat)
jobs/
  smoke.json            tiny end-to-end pipeline, runs on this laptop
  speedrun.json         real-scale template for a multi-GPU pod, not run here
```

## Invariants owned elsewhere

- **A config tree carries only concrete, already-decided values, never a derivation rule.** The
  rules (`presets.py`) are tinylab's own job to run once, at expansion time -- see
  [modelcore/AGENTS.md](https://github.com/8kb/modelcore/blob/main/AGENTS.md).
- **Optimizer state is checkpointed and reloaded positionally** -- the role-order
  `ModelManager.create_optimizer` builds internally is part of the on-disk format; never
  reconstruct a param-group list by hand.
- **datacore doesn't compare tokenizer identity itself** -- a prepared dataset's
  `tokenizer_fingerprint` is host-checked, never datacore's job. See
  [datacore/AGENTS.md](https://github.com/8kb/datacore/blob/main/AGENTS.md).
- **A model already satisfies `benchcore.Model`; a `Generator` doesn't come for free.**
  `tinylab.engine.Engine.generate_batch` is that adapter -- see
  [benchcore/AGENTS.md](https://github.com/8kb/benchcore/blob/main/AGENTS.md)'s `protocols.py`.

## Invariants that will bite you (tinylab's own)

- **`nanochat` cannot be a real dependency.** It's a *virtual* uv project (no `[build-system]` in
  its `pyproject.toml`; `uv.lock` records `source = { virtual = "." }`), so it's never built or
  installable. Everything tinylab needs from it (tokenizer, presets, engine, checkpoint naming,
  scaling-law math) is **ported, not imported** -- each ported file's docstring names its origin.
  `tests/test_no_nanochat.py` mechanically guards against an accidental `import nanochat` slipping
  in (easy to write by hand since the two repos sit side by side on disk).
- **tinylab keeps its own cache directory**, `~/.cache/tinylab/` (`TINYLAB_BASE_DIR` to override)
  -- separate from nanochat's `~/.cache/nanochat/`, even though the ported tokenizer produces
  identical token ids. A prepared dataset, checkpoint, or downloaded shard from one is never read
  by the other.
- **An unrecognized job-file key is a hard error, with a close-match suggestion when there is
  one** -- including inside a `"model"` block, not just at a step's top level. There's no
  `argparse` here to catch a typo -- `tinylab.job.check_known_keys` is the only thing that will.
  A key inherited purely from `"defaults"` is still checked against the *merged* step's accepted
  keys; only a leading-underscore key (`"_comment"`) is exempt, deliberately, as a JSON comment.
  `prepare`/`train`/`bench` each accept a different key set depending on their step's own
  `"kind"`/`"suite"` (see each module's `accepted_keys(cfg)`), so a key that's real for one kind
  but nonsensical for another (e.g. `"mmlu_epochs"` on a `"kind": "base"` prepare step) is also
  caught, not silently ignored.
- **`chat` is a separate CLI command, never a job op.** A pipeline step must run to completion
  unattended; a REPL blocks on a human at a prompt. Don't add `"chat"` to `tinylab.ops.OPS` --
  `tinylab/job.py`'s tests pin this (a step with `"op": "chat"` is rejected as an unknown op).
- **`train`'s own forward/backward loop is eager -- no `torch.compile`.** On this dev machine,
  MPS's first real `torch.compile`'d op after a cold Metal shader cache can sit in
  `waitUntilCompleted` for *minutes* with no stdout; skipping compile in the loop itself keeps a
  job's very first step from ever hitting that stall. (`modelcore.optim.MuonAdamW.step` is
  compiled regardless, internally -- see ".python-version" below for what that costs on this
  machine specifically.)
- **A `train` step's output checkpoint tag defaults to the step's own `"name"`.** This is what
  lets an `"sft"` step's `"source_tag": "pre"` and a later `bench` step's `"model_tag": "sft"`
  reference each other by the pipeline's own step names, with no separate tag field to keep in
  sync -- see `README.md`'s glossary.
- **`prepare` and `train` share one `"sequence_len"`, not a per-op field**, precisely because a
  prepared dataset's `sequence_len` is fixed at prepare time and a training step must match it
  exactly (`tinylab.ops.train._open_dataset` raises if they disagree) -- put it once, in
  `"defaults"`.
- **A `bench` step's `"max_per_task"` must leave enough examples for every CORE task's own
  few-shot count**, or benchcore's few-shot sampling raises a `ValueError` (it draws from the
  *other* examples in the same truncated task, so a `max_per_task` of N leaves a pool of only
  N - 1). Bit `jobs/smoke.json`'s first real run directly (`8` was too small; `24` isn't). Owned
  by benchcore, not tinylab -- leave `"max_per_task"` unset for a real (non-smoke) CORE run.
- **A CORE `bench` step needs a model trained with enough `"sequence_len"` headroom for its
  few-shot prompts, not just enough for training itself.** A rendered CORE prompt (up to ~10
  few-shot examples concatenated) can run past 2,000 tokens; modelcore's rotary embedding cache is
  sized at `sequence_len * 10` when the model is built, and exceeding it is a hard error, not a
  truncation. This is why `jobs/smoke.json` trains at `sequence_len: 2048` (matching nanochat's
  own real default) even though the model itself is tiny (`depth: 4`) -- a smaller `sequence_len`
  bit this exact job's first real CORE run.

## Testing

```bash
uv run pytest -q                 # everything: ~2s on this machine, incl. one real training run
uv run pytest -q -m "not slow"   # skip that one real run: well under a second
```

`tests/conftest.py` has the one shared `base_dir` fixture (isolates `TINYLAB_BASE_DIR` to a
tmp_path) that most tests use. `test_smoke.py` (marked `slow`) is the only test that trains
anything real -- a few steps on a synthetic in-memory corpus, no network -- and is what proves
`train` -> `checkpoints` -> `engine` actually works together, not just each in isolation.

A change here that a subsystem's own suite wouldn't catch (e.g. a new op, a job-schema change)
needs `uv run pytest -q` green against the *pinned* install (not an editable local sibling
checkout) before it's considered landed -- see
[`llmllab/docs/subsystem-conventions.md`](../llmllab/docs/subsystem-conventions.md)'s tag-bump
rule for why that distinction matters when bumping a pin.

## This laptop

Dev machine: Apple Silicon (M4), macOS, no CUDA -- see
[`../llmllab/AGENTS.md`](../llmllab/AGENTS.md#this-laptop) for the shared facts (MPS's cold-compile
stall, `print()` block-buffering when stdout is redirected, no long local training runs).
`jobs/smoke.json` is sized for this machine in minutes; `jobs/speedrun.json` is a real-scale
template meant for a multi-GPU pod, never run here.

- **`.python-version` is pinned to `3.11`, not the family's usual `3.10`, for a real reason:**
  `modelcore.optim.MuonAdamW.step` is unconditionally `@torch.compile`d (not something tinylab's
  own training loop controls or can opt out of). On this machine, uv's own managed CPython 3.10
  installs under `~/Library/Application Support/uv/python/...` -- the space in "Application
  Support" breaks Inductor's generated `clang++ -L<path>` command, and every `MuonAdamW.step()`
  call fails with a `CppCompileError`. Homebrew's `python3.11` (no spaces in its real path)
  doesn't hit this. `requires-python` stays `>=3.10` in `pyproject.toml` -- this is purely a local
  `.venv` choice on this machine, not a real dependency floor, and may well not apply to yours.
  Revisit if a `uv`/PyTorch fix lands, or you're on a machine without this path shape.

## Style

See [llmllab/AGENTS.md](../llmllab/AGENTS.md#style).
