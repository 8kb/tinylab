# Job-file reference

Every key a job file can carry, what it means, and its default when omitted. This is the complete
reference `tinylab.job.check_known_keys` validates against — an unrecognized key anywhere (a typo,
a key that's real for a different `kind`/`suite`) is a hard error with a did-you-mean suggestion, so
this list is exhaustive by construction: `tests/test_docs.py` fails if the code and this file
disagree.

A key starting with `_` (e.g. `"_comment"`, `"_note"`) is a freeform comment, ignored everywhere. The
same convention holds inside a `model_config` file (a `modelcore.v2` config accepts `_` keys at every
level and writes them back unchanged) — the `_` is what tells a comment apart from a mistyped
parameter, which stays an error.

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
| `model_config` | Path to a materialized `modelcore.ModelConfig` tree (a dict with `"#type"` markers, `"format": "modelcore.v2"`) — dumped by nanochat's `scripts/model_info.py --dump-config`, or hand-written. A `modelcore.v1` file still loads (modelcore upgrades it). Relative paths resolve against the job file's own directory. tinylab does no preset/depth-dial derivation of its own — see AGENTS.md. Required for `train` with `kind: base`; optional for `kind: sft` (an adapter-override request on top of the loaded checkpoint's own config). Loading checks the tree's own `sequence_len`/`vocab_size` (baked in when it was dumped) match the step's `"sequence_len"` and the local tokenizer's vocab size — a mismatch is a hard error. Irrelevant to `prepare`/`bench`, harmless if present. | — |
| `world_size` | The GPU count a `train` step's `total_batch_size`/grad-accum arithmetic assumes — checked against the actual launch (e.g. `torchrun --nproc_per_node`) and a hard error on mismatch, since the job file fixes GPU count rather than tinylab inferring it. Irrelevant to `prepare`/`bench`. Required by `train`. | — |
| `tokenizer` | Which tokenizer this step uses: a **bare name** (`"bpe32k"` → `<base_dir>/tokenizers/bpe32k/`) or, if the value contains a path separator, a **path** (`"./toks/mine"`; relative paths resolve against the job file's own directory, like `model_config`). Read **per step**, not once for the whole run: a step without its own `"tokenizer"` falls back to the run's default (read off the first resolved step, same as `device` — put it in `defaults` for the common case of one tokenizer per run). A job training several differently-vocabbed models sets `"tokenizer"` on each `prepare`/`train`/`bench` step that needs a non-default one instead — see "Training two differently-vocabbed models" below. The default one is tinylab's bundled vocab, copied to `<base_dir>/tokenizers/default/` on first use; any other name must already exist (train it with a `tokenizer` step first). `bench`/`chat` pick up the checkpoint's own recorded tokenizer on their own when this is unset, and refuse to load it with one whose fingerprint differs. | run's default, else `"default"` |
| `log_dir` | Where `python -m tinylab job.json` (not `--dry-run`) writes this run's log files: `<base_dir>/<log_dir>/<job_name>.log` (the whole run) plus `<base_dir>/<log_dir>/<job_name>-<step_name>.log` per executed step. One value for a whole run, like `device` — put it in `defaults`. Same 1-4-`/`-joined-names format as a checkpoint tag (see the glossary below); no built-in fallback location, so a run with no `log_dir` set is a hard error before any step executes. | **required** |

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
| `output_tag` | Checkpoint tag this step writes to: `<base_dir>/checkpoints/<tag>/`. Arbitrary text, optionally with `/` folders (`gpt-d12-base`, `kvcache/d13-chat`) — see the glossary. An sft step's must differ from its own `source_tag`. | the step's own `name` | both |
| `num_iterations` | Training horizon, in steps. | required (base) / one epoch over the dataset (sft, unset) | both |
| `device_batch_size` | Micro-batch size per device, per forward/backward. | `4` | both |
| `total_batch_size` | Tokens per optimizer step, across grad-accum and all ranks; must be a multiple of `device_batch_size * sequence_len * world_size`. | required | both |
| `embedding_lr` | Embedding-table learning rate. | `0.3` | both |
| `unembedding_lr` | Unembedding (LM head) learning rate. | `0.008` | both |
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
| `fp8` | Enable FP8 training (`modelcore.ModelManager.enable_fp8`, always its own "tensorwise" recipe -- the only one implemented; needs an H100+ GPU). | `false` | both |
| `fp8_eval` | When `fp8` is on, measure val bpb directly in fp8 (`true`) instead of converting back to bf16 first (`false`). Irrelevant without `fp8`. | `true` | both |
| `doc_masking` | Restrict attention to within each packed row's own document (BOS-delimited), instead of allowing attention across document boundaries within a row. | `false` | both |
| `doc_masking_max_docs_per_row` | Override `build_doc_args`'s default per-row document budget (`DEFAULT_MAX_DOCS_PER_ROW=64`). | unset | both |
| `adapter_lr` | Learning rate for adapter (LoRA/DoRA A/B) params. Only meaningful when `model_config` carries `adapters`. | `modelcore.OptimizerHparams.adapter_lr` | both |
| `adapter_scalar_lr` | Learning rate for DoRA's per-channel magnitude param. | `modelcore.OptimizerHparams.adapter_scalar_lr` | both |
| `source_tag` | Checkpoint tag to fine-tune from — normally an earlier `kind: base` step's `output_tag`. | required | sft |
| `source_step` | A specific step of that checkpoint, instead of its latest. | latest | sft |
| `init_lr_frac` | Starting sft LR as a fraction of the (possibly warm-started) base LR — nanochat's `chat_sft.py --init-lr-frac`. | `0.8` | sft |
| `load_optimizer` | Warm-start the optimizer from `source_tag`'s own checkpoint (Muon/Adam momentum buffers only — LRs are reset to this step's own, then scaled by `init_lr_frac`). Forced off (and an explicit `true` is an error) when the step's model carries `adapters` — the pretrained optimizer's param-group layout doesn't match an adapter-augmented model. | `true` | sft |

## `tokenizer`

Trains a fresh BPE vocab and writes it to the tokenizer named by its `output` key — else the run's
`tokenizer`, else the default — overwriting whatever was there (no skip-if-exists guard). Most job
files never need this: tinylab ships a committed default vocab. If used, put it before any step that
loads *that same* tokenizer (each is loaded once and cached on first use; a step that already loaded
it makes this one raise). Training `"b"` after something loaded `"a"` is fine. No required keys; every
default below matches nanochat's own `scripts/tok_train.py`.

| Key | Meaning | Default |
|---|---|---|
| `shards` | Number of ClimbMix train shards to download and read (plus the fixed validation shard) — same corpus `prepare`'s `kind: base` uses. | `8` |
| `vocab_size` | Target vocab size, including the 9 special tokens (appended after training, never trained themselves). Must leave at least 256 ordinary tokens. | `32768` |
| `doc_cap` | Truncates any single document to this many characters before it reaches the trainer. | `10000` |
| `max_chars` | Stops reading once this many characters (post-`doc_cap`) have been seen. | `2000000000` |
| `output` | Which tokenizer to write: a bare name (`<base_dir>/tokenizers/<name>/`) or a path, same rule as the common `tokenizer` key. | the step's own `tokenizer`, else `"default"` |

## `bench`

Scores a checkpoint. `"suite": "core"` runs DCLM's CORE benchmark; `"suite": "chat"` runs the ARC/
MMLU/GSM8K/HumanEval chat-task suite plus the combined ChatCORE metric. Required: `suite`,
`model_tag`. The checkpoint's tokenizer is this step's own `tokenizer` if set, else the one the
checkpoint itself records.

| Key | Meaning | Default | Which `suite` |
|---|---|---|---|
| `suite` | `"core"` or `"chat"`. | required | both |
| `model_tag` | Checkpoint tag to load. | required | both |
| `model_step` | A specific step of that checkpoint, instead of its latest. | latest | both |
| `max_per_task` | Caps examples per CORE task. Must leave enough for that task's own few-shot count, or benchcore's few-shot sampling raises — see `docs/architecture.md`. | unset (every example) | core |
| `tasks` | Which chat tasks to run. | all of `ARC-Easy`, `ARC-Challenge`, `MMLU`, `GSM8K`, `HumanEval` | chat |
| `batch_size` | Categorical (ARC/MMLU) loop's problems-per-forward. Never reaches the generative loop. | `1` | chat |
| `generative_batch_size` | Generative (GSM8K/HumanEval) loop's problems-per-decode-batch. `1` is the one-problem-at-a-time loop; more needs `tinylab.engine.Engine.generate_batch_multi` (always present here) and, at `temperature > 0`, changes which tokens get sampled. A separate key from `batch_size` on purpose — see `docs/architecture.md`. | `1` | chat |
| `eval_workers` | Threads scoring a generative batch's completions (HumanEval sandboxes each in its own subprocess). | `1` | chat |
| `num_samples` | Samples per problem. | `1` | chat |
| `max_new_tokens` | Generation length cap. | `256` | chat |
| `temperature` | Sampling temperature. | `0.0` (greedy) | chat |
| `top_k` | Sampling top-k. | `50` | chat |
| `max_problems` | Caps problems per task (for a fast smoke run). Leave unset for a real run: uncapped GSM8K (~1319)/HumanEval (~164) is what `generative_batch_size` exists to make affordable. | unset (every problem) | chat |

## `chat`

Read by `python -m tinylab chat <job.json>`, not by the pipeline. Required: `model_tag`.

| Key | Meaning | Default |
|---|---|---|
| `model_tag` | Checkpoint tag to load. | required |
| `model_step` | A specific step of that checkpoint, instead of its latest. | latest |
| `temperature` | Sampling temperature. | `0.6` |
| `top_k` | Sampling top-k. | `50` |
| `max_tokens` | Generation length cap, per turn. | `256` |
| `prompt` | If present: send this one message, print the reply, and exit — instead of an interactive REPL. | unset (REPL) |

## Training two differently-vocabbed models

A job isn't limited to one tokenizer. `prepare`, `train`, and `bench` each resolve their own step's
`"tokenizer"` key independently (falling back to the run's default only when a step doesn't set
one), so a job that trains, say, a 32k-vocab model and an 8k-vocab model side by side names each
step's tokenizer explicitly rather than putting one in `defaults`:

```json
{
  "steps": [
    {"name": "tok32k", "op": "tokenizer", "vocab_size": 32768, "output": "exp1_32k"},
    {"name": "tok8k",  "op": "tokenizer", "vocab_size": 8192,  "output": "exp1_8k"},
    {"name": "base32k", "op": "prepare", "kind": "base", "tokenizer": "exp1_32k", "shards": 100},
    {"name": "base8k",  "op": "prepare", "kind": "base", "tokenizer": "exp1_8k",  "shards": 100},
    {"name": "pre32k", "op": "train", "kind": "base", "tokenizer": "exp1_32k", "model_config": "m32k.json"},
    {"name": "pre8k",  "op": "train", "kind": "base", "tokenizer": "exp1_8k",  "model_config": "m8k.json"}
  ]
}
```
(only the tokenizer-relevant keys are shown — a real `train` step still needs `sequence_len`,
`total_batch_size`, `num_iterations`, `eval_tokens`, `world_size`, etc., same as any other.)

Each `train` step's checkpoint records *its own* tokenizer (`meta_<step>.json`'s `model_config.
tokenizer.name`), not the run's default, so a later `bench`/`chat` step that omits `"tokenizer"`
still loads the right one automatically. `prepare`'s auto dataset name embeds the tokenizer's
fingerprint (`{climbmix|sft}_t<sequence_len>_<fingerprint>`), so two tokenizers never collide on
one dataset directory even with everything else (kind, sequence_len) identical.

## Glossary: `output_tag`, `source_tag`, `model_tag`, and what a tag is

There is one checkpoint namespace, `<base_dir>/checkpoints/<tag>/`, and **the tag is the whole
address**. It is 1 to 4 `/`-joined names: `gpt-d12-base`, `kvcache/d13-chat`,
`experiments/run3`. Each name is 1-16 characters from `[A-Za-z0-9_-]` — letters, digits, `_`, `-`,
nothing else (`tinylab.checkpoints.validate_name`/`validate_tag`). What *kind* of checkpoint it is
— pretrained, fine-tuned, whatever comes next — is yours to say in the tag; nothing in the path
encodes it. The same rule validates a job's `log_dir` (see below) and every step's own `name`.

Three job keys name a tag, and all three are complete addresses:

- **`output_tag`** (a `train` step): the tag this step *writes to*. Defaults to the step's own
  `name`.
- **`source_tag`** (a `train` step with `kind: sft`): the tag this step *reads its starting weights
  from* — normally an earlier `kind: base` step's `output_tag`. Must differ from that step's own
  `output_tag`.
- **`model_tag`** (`bench`/`chat`): the tag to load — the read-side counterpart to `output_tag`.

A pipeline has no dependency graph; each step names what it needs by tag instead. In
`jobs/smoke.json`, `"sft"` finds `"pre"`'s checkpoint via `"source_tag": "pre"`, and the `"core"`
bench step finds `"sft"`'s checkpoint via `"model_tag": "sft"` — that works only because a `train`
step's output tag defaults to its own step name. Give steps explicit tags when you want the name to
say more (`"output_tag": "gpt-d12-base"`).
