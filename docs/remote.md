# The central bucket

`hf://buckets/mendel-il/tinylab-data` is the one source of truth for everything tinylab produces
that outlives a pod: checkpoints, tokenizers, prepared datasets, experiment logs, benchmark data.
Disks (a laptop's `~/.cache/tinylab/`, a RunPod volume) are caches of it. Set
`"remote": "hf://buckets/mendel-il/tinylab-data"` in a job file's `defaults` to use it; without the
key nothing changes and nothing touches the network.

Code: [`tinylab/remote.py`](../tinylab/remote.py) (bucket wrapper, `Uploader`, `Prefetcher`, push/pull),
[`remote_stores.py`](../tinylab/remote_stores.py) (datacore store wrappers),
[`readme.py`](../tinylab/readme.py), [`remote_cli.py`](../tinylab/remote_cli.py).
The bucket is public and **not versioned**: an overwrite or delete is permanent.

## Layout

A path in the bucket is the path under `TINYLAB_BASE_DIR`; there is no mapping.

```
README.md                          what this is, the map, the rules
tokenizers/<name>/                 README.md, tokenizer.pkl (marker, last)
prepared/<dataset>/                README.md, *.npy shards, manifest.json (marker, last)
checkpoints/<tag>/                 README.md, model_<step>.pt, optim_<step>_rank<r>.pt,
                                   meta_<step>.json (marker for that step, last)
experiments/<experiment>/          README.md (the card), jobs/, configs/, logs/, results/
experiments/scratch/               smoke runs; may be cleaned wholesale
eval_bundle/                       the unpacked CORE bundle (no .zip)
task_data/<repo--slug>/<subset>/<split>/   chat-benchmark sets only (never SmolTalk)
```

Never synchronized, either way: `base_data_*` (raw upstream data), `job_state/`, `locks/`, `*.tmp`, `*.lock`,
`*.tmp.npy`, `*.part`, `eval_bundle.zip`, SmolTalk. (`tinylab.remote.NEVER_SYNC_GLOBS`.)

Each *entity* — a tokenizer, a dataset, a checkpoint tag, an experiment — is one folder with a
`README.md`. Intermediate folders (`checkpoints/exp01/`) get a group README by hand; type folders
(`checkpoints/`) have a type README, kept in git under [`tinylab/remote_readmes/`](../tinylab/remote_readmes)
and uploaded with `python -m tinylab remote push-docs`.

### Entity README

Frontmatter written and updated **only by code** (one `key: <json>` per line, strict YAML subset):
`kind` (`tokenizer|dataset|checkpoint|experiment|bench-data`), `id`, `created`,
`producer` (`repo, version, git, experiment, job, step, hardware`), `inputs` (the lineage: tokenizer,
dataset, source checkpoint, `model_config_sha256`), `retention`, `steps`, `metrics`. Body sections:
`Motivation` (seeded from the step's `"_comment"`, then yours), `History` (append-only, code adds lines),
`Related` (relative links to the inputs; written once), `Notes` (yours). Code never rewrites a body
section other than appending to History. `remote check` lists READMEs whose Motivation is still empty.

## Rules

1. **Namespaces.** A job's `experiment` is the name of its card folder in git (`01-ffn-width-vs-heads`);
   a checkpoint tag's first segment is a short experiment prefix (`exp01` — tag segments are capped at
   16 characters), recorded as `tag_prefix` in the experiment README; `remote check` warns about a tag
   with a prefix no experiment claims. Smoke runs use `experiment: "scratch"` and tags under `scratch/`.
2. **Immutability.** Once an artifact upload is *complete* (its marker exists), none of its files is
   overwritten with different content — a size mismatch is `ImmutableError` (an identical re-push is a
   no-op). A partial upload (no marker) may be replaced. An entity whose README names a different
   `producer` (experiment, job, step) refuses the push (`ProducerConflict`). `--force` exists only on
   the CLI, never as a job-file key. (Logs, results and snapshots under `experiments/` are exempt: they
   are appended to, so they are re-uploaded as they change.)
3. **Completion marker.** A checkpoint step is complete only with `meta_<step>.json` (and its model);
   a dataset only with `manifest.json`; a tokenizer with `tokenizer.pkl`. Markers upload last, after
   everything else in their group, and a pull never sees an entity without its marker. If any earlier
   upload of an entity failed, its marker is withheld. Locally this is datacore's "manifest last".
4. **Deletion.** Only a step's own retention policy (`push_model`/`push_optim`), or an explicit
   `remote rm` with confirmation, deletes. There is no `sync --delete`. Retention deletes the older
   step's marker before its model, so an old step never looks complete without its weights.
5. **Public.** No secrets, tokens or environment dumps in logs or READMEs; tinylab never prints its
   environment.
6. **Identity.** Pulling a tokenizer next to a differing local one with the same name is a hard error
   (`tokenizer.pkl` is compared by hash), never an overwrite.

## What a run does

**Pull (automatic, only what's needed, only what's missing locally):**

| Input | Pulled |
|---|---|
| a named `tokenizer` | the whole `tokenizers/<name>/` |
| `source_tag` / `model_tag` (+ `source_step`/`model_step`, else the newest complete step) | `model_<step>.pt` + `meta_<step>.json` |
| `optim_*` | only for an sft `load_optimizer` warm-start, and for `--resume` |
| a `dataset` | `manifest.json`, `token_bytes` and the val shards up front; train shards stream in a lookahead window ahead of the dataloader (`PrefetchingDatasetStore`) |
| CORE | `eval_bundle/` |
| chat benches | `task_data/` |

A local copy always wins and is never refreshed. Downloads land atomically (`.part` then rename), so a
local file that exists is complete.

**Push (background, parallel with the next step):** one non-daemon `Uploader` thread per run drains a
FIFO queue.
- `train`: each checkpoint save queues `[model, optim_rank*] → meta`; then retention; then the README.
  `push_model`/`push_optim` are `"last"` (default: the bucket always holds the newest resumable state),
  `"all"`, `"none"` or `[steps]`. DDP: rank 0 uploads, after a barrier so every rank's shard is on
  disk (single node, shared disk; multi-node is out of scope).
- `prepare`: each shard is queued as it is written, the manifest last (`push`, default true).
- `tokenizer`: the folder, `tokenizer.pkl` last (`push`, default true).
- `experiments/<experiment>/`: snapshots of the job file and model configs at start (a changed job file
  gets a timestamped sibling, never an overwrite), logs and results synced every 3 minutes and after
  each step, and once more when the job ends. `results/<job>.results.jsonl` gets one line per step.

An upload error never kills training: it is retried with backoff, then remembered and raised when the
job ends (`UploadError`), after which `python -m tinylab remote push <entity>` redoes it from local disk.
Before the first step, a job that pushes checks `whoami` plus a write probe and refuses to start without
a working token — so it fails before spending GPU hours. A job that only pulls (no pushing step) runs
read-only without a token.

## CLI

```
python -m tinylab remote ls [prefix]
python -m tinylab remote pull <entity> [--step N] [--optim]
python -m tinylab remote push <entity> [--steps 1,2] [--optim] [--force]
python -m tinylab remote check
python -m tinylab remote rm <entity>[@step] [--yes]
python -m tinylab remote push-docs
```

`--remote URL` / `$TINYLAB_REMOTE` override the default bucket. `check` verifies write access and lists
READMEs to write, incomplete steps/datasets, and foreign tag prefixes.

## Tokens

Reading needs none (public bucket). Writing needs an `HF_TOKEN` with write access to the bucket: on a
laptop, `hf auth login` (a separate token), on a pod the RunPod secret `hf_bucket_write` exposed as
`HF_TOKEN={{ RUNPOD_SECRET_hf_bucket_write }}` (recipe in
[`llmllab/docs/runpod-ops.md`](../../llmllab/docs/runpod-ops.md)). The token never goes into a job file,
git or a log. Today one token can write anywhere in the bucket, so rules 2–4 above protect against
mistakes, not against a hostile pod; rotate the token after each campaign.
