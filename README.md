# tinylab

tinylab trains a small language model end to end: it downloads and tokenizes data, pretrains a
model, fine-tunes it on conversations, scores it against standard benchmarks, and lets you chat
with it. Every task is described in one JSON file instead of command-line flags.

It's built on three components, each its own repo: [`modelcore`](https://github.com/8kb/modelcore)
(models), [`datacore`](https://github.com/8kb/datacore) (data), and
[`benchcore`](https://github.com/8kb/benchcore) (evaluation). Their own docs cover what they are and
where they came from; this page covers using tinylab.

## Install

```bash
git clone https://github.com/8kb/tinylab
cd tinylab && uv sync --extra cpu --group dev
```

(`uv sync --extra gpu --group dev` on a CUDA machine.)

## The whole CLI

```
python -m tinylab <job.json> [--only NAME] [--dry-run] [--resume]   # run a pipeline
python -m tinylab chat <job.json>                                   # talk to what it trained
```

That's it — two commands, no other flags beyond these three. `--only NAME` re-runs one named step
on its own (useful once that step's inputs, like a prepared dataset or an earlier checkpoint,
already exist on disk). `--dry-run` validates the file and prints every step's fully resolved
config without touching disk, the network, or a GPU. `--resume` continues a previous, interrupted
run of the same job file — skips whichever steps already completed, and continues a `train` step's
own training loop from its last periodic checkpoint (see `"save_every"` below) if it has one. It
can't be combined with `--only`, and it's the one thing here that's a flag rather than a job-file
key — see [`docs/architecture.md`](docs/architecture.md#resume-the-job-state-file).

## Quickstart

```bash
uv run python -m tinylab jobs/smoke.json --dry-run   # validate, spend nothing
uv run python -m tinylab jobs/smoke.json             # the real (tiny) end-to-end run
uv run python -m tinylab chat jobs/smoke.json         # talk to what it just trained
```

The first run downloads a couple of ClimbMix shards and some HuggingFace datasets, then trains a
tiny model for a few dozen steps. It takes a few minutes and produces a genuinely bad model — the
point is to prove the plumbing works end to end, not to produce something worth talking to.

## A job file

```json
{
  "defaults": {
    "device": "auto",
    "sequence_len": 2048,
    "world_size": 1,
    "model_config": "configs/gpt_d4.json"
  },
  "steps": [
    { "name": "data", "op": "prepare", "kind": "base", "shards": 2 },
    { "name": "pre",  "op": "train",   "kind": "base", "num_iterations": 30, "device_batch_size": 1, "total_batch_size": 2048, "eval_tokens": 2048 },
    { "name": "sftdata", "op": "prepare", "kind": "sft", "max_conversations": 200 },
    { "name": "sft",  "op": "train",   "kind": "sft", "source_tag": "pre", "num_iterations": 20, "device_batch_size": 1, "total_batch_size": 2048, "eval_tokens": 2048 },
    { "name": "core", "op": "bench",   "suite": "core", "model_tag": "sft", "max_per_task": 24 }
  ],
  "chat": { "model_tag": "sft", "temperature": 0.6 }
}
```

`"model_config"` names a materialized `modelcore.ModelConfig` tree — tinylab does no preset/
depth-dial derivation of its own; dump one with nanochat's `scripts/model_info.py --dump-config`
(see `jobs/configs/`). This is (a trimmed copy of) `jobs/smoke.json` — a real, runnable pipeline.
`jobs/speedrun.json` is the same shape at production scale, meant for a GPU pod; `jobs/contest.json`
compares two architectures in one pipeline.

`"defaults"` deep-merges into every step and into `"chat"`; a step's own keys always win. Steps run
top to bottom, with no dependency graph — each step instead names what it needs by checkpoint tag
(`"source_tag"`, `"model_tag"`). An unrecognized key anywhere is a hard error with a did-you-mean
suggestion. **See [`docs/job-file.md`](docs/job-file.md) for every
key, its meaning, and its default** — this example only shows a handful.

## Ops

| `op` | what it does |
|---|---|
| `prepare` | Tokenizes and packs a corpus into a dataset. `"kind": "base"` downloads ClimbMix shards; `"kind": "sft"` builds a SmolTalk + MMLU + GSM8K conversation mixture. |
| `train` | One training loop for both `"kind": "base"` (pretrain from scratch) and `"kind": "sft"` (fine-tune a `source_tag`'d base checkpoint). `"save_every": N` checkpoints periodically, not just at the end. |
| `bench` | Scores a checkpoint: `"suite": "core"` (DCLM's CORE benchmark) or `"suite": "chat"` (ARC, MMLU, GSM8K, HumanEval, plus the combined ChatCORE metric). |
| `tokenizer` | Trains a fresh BPE vocab and writes it to the tokenizer it names (`"output"`, else the run's `"tokenizer"`, else `default`) under `<base_dir>/tokenizers/`. Most job files never need this — tinylab ships a committed default vocab; if used, it has to come before any step that loads that same tokenizer. |

## Chatting

`python -m tinylab chat <job.json>` opens an interactive prompt against the checkpoint named in
that file's `"chat"` block. Add `"prompt": "..."` to the block to get one response and exit
instead of a REPL — useful for scripting or a quick sanity check.

## Tests

```bash
uv run pytest -q                 # everything: a couple of seconds, including one real (tiny) training/generation run
uv run pytest -q -m "not slow"   # skip that one real run: well under a second
```

## More

- [`docs/job-file.md`](docs/job-file.md) — every job-file key: meaning, default, which op/kind it
  applies to.
- [`docs/architecture.md`](docs/architecture.md) — module map, how a job file turns into a run, and
  where tinylab's own code ends and each subsystem's begins.
- [`AGENTS.md`](AGENTS.md) — invariants worth knowing before changing the code.

## Development

tinylab pins `modelcore`/`datacore`/`benchcore` as git-tagged dependencies (see `pyproject.toml`'s
`[tool.uv.sources]`). To develop against a local sibling checkout of one of them instead: `uv sync`
once, then `uv pip install -e ../modelcore` (or `../datacore`, `../benchcore`) from here.
