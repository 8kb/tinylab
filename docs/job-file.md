# Job-file reference

Every key a job file can carry, what it means, and its default when omitted. This is the complete
reference `tinylab.job.check_known_keys` validates against — an unrecognized key anywhere (a typo,
a key that's real for a different `kind`/`suite`) is a hard error with a did-you-mean suggestion, so
this list is exhaustive by construction: `tests/test_docs.py` fails if the code and this file
disagree.

A key starting with `_` (e.g. `"_comment"`, `"_note"`) is a freeform comment, ignored everywhere.

`--resume` is the one exception to "everything is a JSON key" — it's a CLI flag
(`python -m tinylab job.json --resume`), not a job-file key, since whether a given invocation is a
fresh start or a continuation is a fact about the invocation, not the pipeline. See
`docs/architecture.md`'s "Resume: the job state file" section for the mechanism.

## Top level

| Key | Meaning |
|---|---|
| `defaults` | Deep-merged into every step and into `chat` (a step's own keys win; nested dicts merge, everything else is replaced wholesale). |
| `steps` | An ordered list of pipeline steps. Required. Each needs its own `name` (unique) and `op` (`prepare`/`train`/`bench`/`tokenizer`). |
| `chat` | The block `python -m tinylab chat` reads. Optional — omit it if this job file has nothing to chat with. |

## Common to every op (`prepare`/`train`/`bench`/`tokenizer`)

| Key | Meaning | Default |
|---|---|---|
| `device` | `"cuda"` / `"mps"` / `"cpu"` / `"auto"`. One value for a whole run — read off the first resolved step, so put it in `defaults`. | `"auto"` |
| `sequence_len` | Context length. Shared between `prepare` and `train`: a dataset is packed to a fixed `sequence_len`, and a `train` step raises if its own `sequence_len` doesn't match. Required by `prepare`/`train`; irrelevant to `bench`. | — |
| `model_config` | Path to a materialized `modelcore.ModelConfig` tree (a dict with `"#type"` markers) — dumped by nanochat's `scripts/model_info.py --dump-config`, or hand-written. Relative paths resolve against the job file's own directory. tinylab does no preset/depth-dial derivation of its own — see AGENTS.md. Required for `train` with `kind: base`; optional for `kind: sft` (an adapter-override request on top of the loaded checkpoint's own config). Loading checks the tree's own `sequence_len`/`vocab_size` (baked in when it was dumped) match the step's `"sequence_len"` and the local tokenizer's vocab size — a mismatch is a hard error. Irrelevant to `prepare`/`bench`, harmless if present. | — |
| `world_size` | The GPU count a `train` step's `total_batch_size`/grad-accum arithmetic assumes — checked against the actual launch (e.g. `torchrun --nproc_per_node`) and a hard error on mismatch, since the job file fixes GPU count rather than tinylab inferring it. Irrelevant to `prepare`/`bench`. Required by `train`. | — |

## `prepare`

`"kind": "base"` downloads and packs ClimbMix shards; `"kind": "sft"` builds the SmolTalk + MMLU +
GSM8K conversation mixture. Required: `kind`.

| Key | Meaning | Default | Which `kind` |
|---|---|---|---|
| `kind` | `"base"` or `"sft"`. | required | both |
| `dataset` | Output dataset name under `<base_dir>/prepared/`. | `{climbmix\|sft}_t<sequence_len>_<tokenizer fingerprint>` | both |
| `sequences_per_volume` | Datacore's on-disk shard granularity. | `16384` | both |
| `buffer_size` | Packer's best-fit lookback window. | `1000` | both |
| `tokenizer_threads` | Parallel tokenization threads. | `os.cpu_count()` | both |
| `shards` | Number of ClimbMix train shards to download and pack (plus one fixed validation shard). | `8` | base |
| `max_conversations` | Caps both the train and val conversation mixtures (for a fast smoke run). | unset (full mixture) | sft |
| `mmlu_epochs` | How many passes of MMLU's auxiliary-train split to mix into SFT training data. | `3` | sft |
| `gsm8k_epochs` | Same, for GSM8K's train split. | `4` | sft |
| `max_tokens_per_conversation` | Truncates any single rendered conversation past this many tokens. | `2048` | sft |
| `sft_padding_id` | Token id `BestFitPadPacker` pads with. | packer's own default | sft |

## `train`

One loop for both `"kind": "base"` (pretrain from scratch) and `"kind": "sft"` (fine-tune a
`source_tag`'d base checkpoint). Required: `kind`, `sequence_len` (in `defaults`), `world_size`,
`total_batch_size`, `eval_tokens`; `base` also requires `model_config` and `num_iterations`; `sft`
also requires `source_tag`.

Launched with `--resume`, a step first checks whether it already has a checkpoint of its own
(under `output_tag`) and, if so, continues training from it — model, optimizer state, and
dataloader position all restored, `model_config`/`source_tag` not re-read. See
`docs/architecture.md`'s "Resume: the job state file" section.

tinylab derives no training-horizon or batch-size scaling law of its own — nanochat's
`target_flops`/`target_param_data_ratio` math (muP batch-size and weight-decay corrections
included) lives only in `modelcore.scaling`, for nanochat's own use. Compute `total_batch_size`/
`num_iterations` with nanochat's `scripts/model_info.py --target-flops=... --json`
(`training_plan.total_batch_size` / `.num_iterations`) and paste the numbers in — see
AGENTS.md.

| Key | Meaning | Default | Which `kind` |
|---|---|---|---|
| `kind` | `"base"` or `"sft"`. | required | both |
| `dataset` | Which prepared dataset to train on. | same auto-name as `prepare` | both |
| `output_tag` | Checkpoint tag this step writes to, under `<base_dir>/{base_checkpoints\|chatsft_checkpoints}/`. | the step's own `name` | both |
| `num_iterations` | Training horizon, in steps. | required (base) / one epoch over the dataset (sft, unset) | both |
| `device_batch_size` | Micro-batch size per device, per forward/backward. | `4` | both |
| `total_batch_size` | Tokens per optimizer step, across grad-accum and all ranks; must be a multiple of `device_batch_size * sequence_len * world_size`. | required | both |
| `embedding_lr` | Embedding-table learning rate. | `0.3` | both |
| `unembedding_lr` | Unembedding (LM head) learning rate. | `0.008` (base) / `0.004` (sft) | both |
| `matrix_lr` | Muon (matrix-parameter) learning rate. | `0.02` | both |
| `scalar_lr` | Scalar-parameter learning rate. | `0.5` | both |
| `weight_decay` | AdamW weight decay, used verbatim (no batch-size rescale). | `0.28` (base) / `0.0` (sft) | both |
| `warmup_steps` | Linear LR warmup length, in steps. | `5` (base) / `0` (sft) | both |
| `warmdown_ratio` | Fraction of the run spent linearly decaying LR to `final_lr_frac`. | `0.65` (base) / `0.5` (sft) | both |
| `final_lr_frac` | LR multiplier at the end of warmdown. | `0.05` (base) / `0.0` (sft) | both |
| `muon_momentum_warmup_steps` | Steps Muon's own momentum ramps over before holding at 0.97. | `400` (`modelcore.optim.schedules.muon_momentum`'s own default) | both |
| `eval_every` | Run a val-bpb pass every N steps (plus always at the final step). `0` disables eval entirely, including at the final step. | `50` | both |
| `eval_tokens` | Tokens to evaluate per val pass. | required | both |
| `save_every` | Save a checkpoint every N steps, in addition to the always-saved final step. `-1` disables periodic saving (today's behavior: final step only). No pruning — each save is a new, permanent `model_<step>.pt`/`meta_<step>.json`/`optim_<step>_rank<r>.pt` triple; a large model at a small `save_every` grows disk usage without bound. | `-1` | both |
| `fp8` | Enable FP8 training (`modelcore.ModelManager.enable_fp8`; needs an H100+ GPU). | `false` | both |
| `fp8_recipe` | FP8 scaling recipe (only `"tensorwise"` is implemented). | `"tensorwise"` | both |
| `fp8_eval` | When `fp8` is on, measure val bpb directly in fp8 (`true`) instead of converting back to bf16 first (`false`). Irrelevant without `fp8`. | `true` | both |
| `doc_masking` | Restrict attention to within each packed row's own document (BOS-delimited), instead of allowing attention across document boundaries within a row. | `false` | both |
| `doc_masking_max_docs_per_row` | Override `build_doc_args`'s default per-row document budget (`DEFAULT_MAX_DOCS_PER_ROW=64`). | unset | both |
| `adapter_lr` | Learning rate for adapter (LoRA/DoRA A/B) params. Only meaningful when `model_config` carries `adapters`. | `modelcore.OptimizerHparams.adapter_lr` | both |
| `adapter_scalar_lr` | Learning rate for DoRA's per-channel magnitude param. | `modelcore.OptimizerHparams.adapter_scalar_lr` | both |
| `source` | Checkpoint namespace to read the starting weights from (`"base"` or `"sft"`). | `"base"` | sft |
| `source_tag` | Checkpoint tag to fine-tune from — normally an earlier `kind: base` step's `output_tag`. | required | sft |
| `source_step` | A specific step of that checkpoint, instead of its latest. | latest | sft |

## `tokenizer`

Trains a fresh BPE vocab and writes it to `<base_dir>/tokenizer/`, overwriting whatever was there
(the bundled default, or an earlier trained one) — no skip-if-exists guard. Most job files never
need this: tinylab ships a committed default vocab. If used, put it first in `"steps"` — it must
run before any earlier step has already read `ctx.tokenizer` (lazily loaded and cached on first
use). No required keys; every default below matches nanochat's own `scripts/tok_train.py`.

| Key | Meaning | Default |
|---|---|---|
| `shards` | Number of ClimbMix train shards to download and read (plus the fixed validation shard) — same corpus `prepare`'s `kind: base` uses. | `8` |
| `vocab_size` | Target vocab size, including the 9 special tokens (appended after training, never trained themselves). Must leave at least 256 ordinary tokens. | `32768` |
| `doc_cap` | Truncates any single document to this many characters before it reaches the trainer. | `10000` |
| `max_chars` | Stops reading once this many characters (post-`doc_cap`) have been seen. | `2000000000` |

## `bench`

Scores a checkpoint. `"suite": "core"` runs DCLM's CORE benchmark; `"suite": "chat"` runs the ARC/
MMLU/GSM8K/HumanEval chat-task suite plus the combined ChatCORE metric. Required: `suite`,
`model_tag`.

| Key | Meaning | Default | Which `suite` |
|---|---|---|---|
| `suite` | `"core"` or `"chat"`. | required | both |
| `source` | Checkpoint namespace to load from (`"base"` or `"sft"`). | `"sft"` | both |
| `model_tag` | Checkpoint tag to load. | required | both |
| `model_step` | A specific step of that checkpoint, instead of its latest. | latest | both |
| `max_per_task` | Caps examples per CORE task. Must leave enough for that task's own few-shot count, or benchcore's few-shot sampling raises — see `docs/architecture.md`. | unset (every example) | core |
| `tasks` | Which chat tasks to run. | all of `ARC-Easy`, `ARC-Challenge`, `MMLU`, `GSM8K`, `HumanEval` | chat |
| `batch_size` | Generation batch size. | `1` | chat |
| `num_samples` | Samples per problem. | `1` | chat |
| `max_new_tokens` | Generation length cap. | `256` | chat |
| `temperature` | Sampling temperature. | `0.0` (greedy) | chat |
| `top_k` | Sampling top-k. | `50` | chat |
| `max_problems` | Caps problems per task (for a fast smoke run). | unset (every problem) | chat |

## `chat`

Read by `python -m tinylab chat <job.json>`, not by the pipeline. Required: `model_tag`.

| Key | Meaning | Default |
|---|---|---|
| `source` | Checkpoint namespace to load from (`"base"` or `"sft"`). | `"sft"` |
| `model_tag` | Checkpoint tag to load. | required |
| `model_step` | A specific step of that checkpoint, instead of its latest. | latest |
| `temperature` | Sampling temperature. | `0.6` |
| `top_k` | Sampling top-k. | `50` |
| `max_tokens` | Generation length cap, per turn. | `256` |
| `prompt` | If present: send this one message, print the reply, and exit — instead of an interactive REPL. | unset (REPL) |

## Glossary: `source`, `source_tag`, `model_tag`, `output_tag`

Four related but distinct ideas show up across `train`, `bench`, and `chat`:

- **`output_tag`** (a `train` step): the checkpoint tag this step *writes to*. Defaults to the
  step's own `name`.
- **`source_tag`** (a `train` step with `kind: sft`): the checkpoint tag this step *reads its
  starting weights from* — normally an earlier `kind: base` step's `output_tag`.
- **`source`** (`train`/`bench`/`chat`): which checkpoint *namespace* to read from, `"base"` or
  `"sft"` — two separate directories, since a base (pretrained) checkpoint and an SFT (fine-tuned)
  checkpoint are never the same tag.
- **`model_tag`** (`bench`/`chat`): the checkpoint tag to load, inside whichever `source`
  namespace — the read-side counterpart to `output_tag`.

A pipeline has no dependency graph; each step names what it needs by tag instead. In
`jobs/smoke.json`, `"sft"` finds `"pre"`'s checkpoint via `"source_tag": "pre"`, and the `"core"`
bench step finds `"sft"`'s checkpoint via `"model_tag": "sft"` — that works only because a `train`
step's output tag defaults to its own step name.
