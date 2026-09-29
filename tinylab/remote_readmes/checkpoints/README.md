# checkpoints/

`checkpoints/<tag>/` where the tag is the tinylab checkpoint tag: `/`-joined names, the first being a
short experiment prefix (`exp01/m3/sft`). A step is three kinds of file:
`model_<step>.pt`, `optim_<step>_rank<r>.pt`, and `meta_<step>.json` (the completeness marker,
uploaded last; it also carries the model config). By default only the newest step is kept, and its
optimizer state with it, so the bucket always holds the latest resumable state; a job file's
`push_model` / `push_optim` change that.

Each tag folder's README says what produced it and from what (`inputs:` links to the tokenizer,
dataset and source checkpoint). Rules:
[`tinylab/docs/remote.md`](https://github.com/8kb/tinylab/blob/main/docs/remote.md).
