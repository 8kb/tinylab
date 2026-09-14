# tinylab

tinylab trains a small language model end to end: it downloads and tokenizes data, pretrains a
model, fine-tunes it on conversations, scores it against standard benchmarks, and lets you chat
with it. Every task is described in one JSON file instead of command-line flags.

It's the second [llmllab](https://github.com/8kb/llmllab) host application, alongside
[nanochat](https://github.com/8kb/nanochat) (an architecture-playground fork of
karpathy/nanochat). Where nanochat exposes dozens of flags for trying new architectures, tinylab
is the minimal opposite: one job file, four things it can do.

## The whole CLI

```
python -m tinylab <job.json> [--only NAME] [--dry-run]   # run a pipeline
python -m tinylab chat <job.json>                        # talk to what it trained
```

That's it -- two commands, no other flags. `--only NAME` re-runs one named step on its own (useful
once that step's inputs, like a prepared dataset or an earlier checkpoint, already exist on disk).
`--dry-run` validates the file and prints every step's fully resolved config without touching
disk, the network, or a GPU.

## A job file

```json
{
  "defaults": {
    "device": "auto",
    "sequence_len": 2048,
    "model": { "preset": "gpt", "depth": 4, "aspect_ratio": 64 }
  },
  "steps": [
    { "name": "data", "op": "prepare", "kind": "base", "shards": 2 },
    { "name": "pre",  "op": "train",   "kind": "base", "num_iterations": 30, "device_batch_size": 1, "total_batch_size": 2048 },
    { "name": "sftdata", "op": "prepare", "kind": "sft", "max_conversations": 200 },
    { "name": "sft",  "op": "train",   "kind": "sft", "source_tag": "pre", "num_iterations": 20 },
    { "name": "core", "op": "bench",   "suite": "core", "source": "sft", "model_tag": "sft", "max_per_task": 24 }
  ],
  "chat": { "source": "sft", "model_tag": "sft", "temperature": 0.6 }
}
```

This is (a trimmed copy of) `jobs/smoke.json` -- a real, runnable pipeline sized for a laptop.
`jobs/speedrun.json` is the same shape at production scale, meant for a GPU pod.

A few things worth knowing about the format:

- `"defaults"` deep-merges into every step and into `"chat"`; a step's own keys always win.
- Steps run top to bottom. There's no dependency graph -- instead, each step names what it needs:
  `"sft"` above finds `"pre"`'s checkpoint via `"source_tag": "pre"`, and the `"core"` bench step
  and the `"chat"` block both find `"sft"`'s checkpoint via `"model_tag": "sft"`. That works
  because **a `train` step's output checkpoint tag defaults to its own `"name"`** -- `"pre"`
  writes to a checkpoint tagged `pre`, unless you set `"output_tag"` explicitly.
- An unrecognized key anywhere -- including inside `"model"` -- is a hard error, with a
  did-you-mean suggestion when one is close. There's no `argparse` here to catch a typo otherwise.
- A key starting with `_` (like this README's own `"_comment"` convention in `jobs/*.json`) is a
  freeform comment, always ignored, since JSON has no comment syntax of its own.

### Glossary: source, model_tag, output_tag

Three related but distinct ideas show up across `train`, `bench`, and `chat`:

- **`output_tag`** (a `train` step): the checkpoint tag this step *writes to*. Defaults to the
  step's own `"name"`.
- **`source_tag`** (a `train` step with `"kind": "sft"`): the checkpoint tag this step *reads its
  starting weights from* -- normally an earlier `"kind": "base"` step's `output_tag`.
- **`source`** (any of `train`/`bench`/`chat`): which checkpoint *namespace* to read from,
  `"base"` or `"sft"` -- these are two separate directories, since a base (pretrained) checkpoint
  and an SFT (fine-tuned) checkpoint are never the same tag. `train` defaults `source` to
  `"base"` (SFT normally starts from a base checkpoint); `bench` and `chat` default it to `"sft"`
  (you normally score or talk to the fine-tuned model).
- **`model_tag`** (`bench`/`chat`): the checkpoint tag to load, inside whichever `source`
  namespace -- the read-side counterpart to `output_tag`.

### Chatting

`python -m tinylab chat <job.json>` opens an interactive prompt against the checkpoint named in
that file's `"chat"` block. Add `"prompt": "..."` to the block to get one response and exit
instead of a REPL -- useful for scripting or a quick sanity check.

## Ops

| `op` | what it does |
|---|---|
| `prepare` | Tokenizes and packs a corpus into a dataset. `"kind": "base"` downloads ClimbMix shards; `"kind": "sft"` builds a SmolTalk + MMLU + GSM8K conversation mixture. |
| `train` | One training loop for both `"kind": "base"` (pretrain from scratch) and `"kind": "sft"` (fine-tune a `source_tag`'d base checkpoint). |
| `bench` | Scores a checkpoint: `"suite": "core"` (DCLM's CORE benchmark) or `"suite": "chat"` (ARC, MMLU, GSM8K, HumanEval, plus the combined ChatCORE metric). |

## Quickstart

```bash
uv sync --extra cpu --group dev
uv run python -m tinylab jobs/smoke.json --dry-run   # validate, spend nothing
uv run python -m tinylab jobs/smoke.json             # the real (tiny) end-to-end run
uv run python -m tinylab chat jobs/smoke.json         # talk to what it just trained
```

The first run downloads a couple of ClimbMix shards and some HuggingFace datasets, then trains a
tiny model for a few dozen steps. It takes a few minutes and produces a genuinely bad model --
that's expected, the point is to prove the plumbing works end to end, not to produce something
worth talking to. If `uv sync` picks a different Python interpreter than expected and training
fails with a C++ compiler error, see `AGENTS.md` -- that's a known, machine-specific issue with
one particular Python install path, not a tinylab bug.

## Tests

```bash
uv run pytest -q                 # everything: ~2s, including one real (tiny) training/generation run
uv run pytest -q -m "not slow"   # skip that one real run: under a second
```

## Development

```bash
git clone https://github.com/8kb/tinylab
cd tinylab && uv sync --extra cpu --group dev
```

tinylab pins `modelcore`/`datacore`/`benchcore` as git-tagged dependencies (see `pyproject.toml`'s
`[tool.uv.sources]`). To develop against a local sibling checkout of one of them instead: `uv sync`
once, then `uv pip install -e ../modelcore` (or `../datacore`, `../benchcore`) from here.
