# tokenizers/

One folder per trained tokenizer: `tokenizers/<name>/{README.md, tokenizer.pkl, token_bytes.pt}`.
The name is the value of a job file's `"tokenizer"` key (`tok8k`, `tok32k`). `tokenizer.pkl` is the
completeness marker and is uploaded last. A tokenizer's identity is its fingerprint (in its README);
a checkpoint and a dataset both record the fingerprint they were built with, so a same-named but
different tokenizer is refused, never silently loaded.

Rules: [`tinylab/docs/remote.md`](https://github.com/8kb/tinylab/blob/main/docs/remote.md).
