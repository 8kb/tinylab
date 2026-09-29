# task_data/

Benchmark datasets for chat evaluation (MMLU, GSM8K, ARC, HumanEval), laid out as
`task_data/<repo--slug>/<subset>/<split>/`. Only benchmark sets live here — not training corpora such
as SmolTalk — so a bench pod can pull exactly what it evaluates on.

Rules: [`tinylab/docs/remote.md`](https://github.com/8kb/tinylab/blob/main/docs/remote.md).
