# tinylab-data

Central store for [tinylab](https://github.com/8kb/tinylab) artifacts: checkpoints, tokenizers,
prepared datasets, experiment logs and benchmark data. The layout mirrors tinylab's `base_dir`
(`~/.cache/tinylab/`) exactly: a path here is the path under `TINYLAB_BASE_DIR`.

| Folder | What it holds |
|---|---|
| [`tokenizers/`](tokenizers/README.md) | trained BPE vocabularies |
| [`prepared/`](prepared/README.md) | tokenized, packed datasets |
| [`checkpoints/`](checkpoints/README.md) | model weights, optimizer state, metadata |
| [`experiments/`](experiments/README.md) | one folder per experiment: card, job files, logs, results |
| [`eval_bundle/`](eval_bundle/README.md) | the CORE benchmark bundle |
| [`task_data/`](task_data/README.md) | chat-benchmark datasets (MMLU, GSM8K, ARC, HumanEval) |

Every self-contained thing (a tokenizer, a dataset, a checkpoint tag, an experiment) has its own
folder with its own `README.md`: a machine-written frontmatter (what produced it, from which inputs)
plus human sections. The full rules are in
[`tinylab/docs/remote.md`](https://github.com/8kb/tinylab/blob/main/docs/remote.md); the short version:

1. **Immutable.** A file of a complete upload is never overwritten with different content.
2. **Marker last.** A checkpoint step is complete only with its `meta_<step>.json`; a dataset only with
   its `manifest.json`. Both are uploaded last; a reader ignores anything without its marker.
3. **Deletion only by retention.** Only the producing step's own retention policy, or an explicit
   `tinylab remote rm`, deletes anything. The bucket is not versioned: a delete is forever.
4. **Public.** No secrets, tokens or environment dumps in logs or READMEs.
5. **Identity.** A tokenizer pulled next to a different local one with the same name is an error.

Use it with `python -m tinylab remote ls | pull | push | check`, or just put
`"remote": "hf://buckets/mendel-il/tinylab-data"` in a job file's `defaults`.
