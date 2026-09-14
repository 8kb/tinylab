# Job-file reference

Every key a job file can carry, what it means, and its default when omitted. This is the complete
reference `tinylab.job.check_known_keys` validates against — an unrecognized key anywhere (a typo,
a key that's real for a different `kind`/`suite`) is a hard error with a did-you-mean suggestion, so
this list is exhaustive by construction: `tests/test_docs.py` fails if the code and this file
disagree.

A key starting with `_` (e.g. `"_comment"`, `"_note"`) is a freeform comment, ignored everywhere.

## Top level

| Key | Meaning |
|---|---|
| `defaults` | Deep-merged into every step and into `chat` (a step's own keys win; nested dicts merge, everything else is replaced wholesale). |
| `steps` | An ordered list of pipeline steps. Required. Each needs its own `name` (unique) and `op` (`prepare`/`train`/`bench`). |
| `chat` | The block `python -m tinylab chat` reads. Optional — omit it if this job file has nothing to chat with. |

## `model` block

Nested under a step's (or `defaults`') `"model"` key — shapes the architecture for `prepare`
(irrelevant, ignored), `train` (`kind: base` only builds one), and `bench`/`chat` (only relevant
indirectly, since they load an already-built checkpoint).

| Key | Meaning | Default |
|---|---|---|
| `preset` | Architecture preset name. Only `"gpt"` exists today. | `"gpt"` |
| `config` | A raw, already-materialized `modelcore.ModelConfig` tree (a dict with `"#type"` markers), used instead of `preset`+`depth` — e.g. one dumped by a prior `--dry-run`. | — |
| `depth` | The one dial: how many transformer blocks. Everything else (`n_embd`, `n_head`, …) derives from it. | `6` |
| `aspect_ratio` | `n_embd` grows with `depth * aspect_ratio`, rounded up to a multiple of `head_dim`. | `64` |
| `head_dim` | Per-head dimension. | `128` |
| `window_pattern` | Per-layer sliding-window tiling (`L`=full context, `S`=quarter context); tiled across layers, last layer always full. | `"SSSL"` |
| `arch_opts` | Extra keyword args passed straight through to the preset's `expand()`. | `{}` |
| `d_ref_scaling_params` | Overrides the muP scaling-law reference model's parameter count (normally re-derived automatically from a depth-12 reference model). `train`-only, `kind: base` only. | auto-derived |

## Common to every op (`prepare`/`train`/`bench`)

| Key | Meaning | Default |
|---|---|---|
| `device` | `"cuda"` / `"mps"` / `"cpu"` / `"auto"`. One value for a whole run — read off the first resolved step, so put it in `defaults`. | `"auto"` |
| `sequence_len` | Context length. Shared between `prepare` and `train`: a dataset is packed to a fixed `sequence_len`, and a `train` step raises if its own `sequence_len` doesn't match. Required by `prepare`/`train`; irrelevant to `bench`. | — |
| `model` | See the block above. Irrelevant to `prepare`/`bench`, harmless if present (defaults commonly set it once and let every step inherit it). | — |

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
`source_tag`'d base checkpoint). Required: `kind`, `sequence_len` (in `defaults`); `sft` also
requires `source_tag`.

| Key | Meaning | Default | Which `kind` |
|---|---|---|---|
| `kind` | `"base"` or `"sft"`. | required | both |
| `dataset` | Which prepared dataset to train on. | same auto-name as `prepare` | both |
| `output_tag` | Checkpoint tag this step writes to, under `<base_dir>/{base_checkpoints\|chatsft_checkpoints}/`. | the step's own `name` | both |
| `num_iterations` | Training horizon, in steps. `base`: `-1` (unset) defers to `target_flops`, then `target_param_data_ratio`. `sft`: unset derives one epoch over the dataset. | see left | both |
| `device_batch_size` | Micro-batch size per device, per forward/backward. | `4` | both |
| `total_batch_size` | Tokens per optimizer step, across grad-accum and all ranks; must be a multiple of `device_batch_size * sequence_len * world_size`. `base`: `-1` derives one from the muP scaling law. `sft`: derives from `device_batch_size` alone. | see left | both |
| `embedding_lr` | Embedding-table learning rate (scaled by the batch-size correction on `base`). | `0.3` | both |
| `unembedding_lr` | Unembedding (LM head) learning rate. | `0.008` (base) / `0.004` (sft) | both |
| `matrix_lr` | Muon (matrix-parameter) learning rate. | `0.02` | both |
| `scalar_lr` | Scalar-parameter learning rate. | `0.5` | both |
| `weight_decay` | AdamW weight decay. On `base` this is then rescaled by the training-plan math (see `docs/architecture.md`). | `0.28` (base) / `0.0` (sft) | both |
| `warmup_steps` | Linear LR warmup length, in steps. | `5` (base) / `0` (sft) | both |
| `warmdown_ratio` | Fraction of the run spent linearly decaying LR to `final_lr_frac`. | `0.65` (base) / `0.5` (sft) | both |
| `final_lr_frac` | LR multiplier at the end of warmdown. | `0.05` (base) / `0.0` (sft) | both |
| `eval_every` | Run a val-bpb pass every N steps (plus always at the final step). `0` disables eval entirely, including at the final step. | `50` | both |
| `eval_tokens` | Tokens to evaluate per val pass. | `device_batch_size * sequence_len * world_size * 4` | both |
| `target_flops` | Derive `num_iterations` from a total-FLOPs budget instead of setting it directly. | `-1.0` (unused) | base |
| `target_param_data_ratio` | Derive `num_iterations` from a tokens-per-parameter ratio (Chinchilla-style) instead. | `12` | base |
| `source` | Checkpoint namespace to read the starting weights from (`"base"` or `"sft"`). | `"base"` | sft |
| `source_tag` | Checkpoint tag to fine-tune from — normally an earlier `kind: base` step's `output_tag`. | required | sft |
| `source_step` | A specific step of that checkpoint, instead of its latest. | latest | sft |

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
