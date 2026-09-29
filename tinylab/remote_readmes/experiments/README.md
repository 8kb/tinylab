# experiments/

One folder per experiment, named like its card in git (`01-ffn-width-vs-heads`):

- `README.md` — the experiment card (goal, design, results, links to the checkpoints, datasets and
  tokenizers it used; the source of truth is the git repo, this is a copy)
- `jobs/` — the job files as they were when launched
- `configs/` — the model configs those jobs pointed at
- `logs/` — run logs, uploaded periodically while a job runs
- `results/<job>.results.jsonl` — one result record per finished step

`experiments/scratch/` is for smoke tests and is cleaned out wholesale. Checkpoints, datasets and
tokenizers stay in their own shared folders because experiments reuse each other's.

Rules: [`tinylab/docs/remote.md`](https://github.com/8kb/tinylab/blob/main/docs/remote.md).
