# prepared/

One folder per prepared dataset: `prepared/<dataset>/{README.md, *.npy, manifest.json}` (a
[datacore](https://github.com/8kb/datacore) dataset). Shards are uploaded as they are written;
`manifest.json` goes last and is the completeness marker, so a dataset without one is still being
uploaded (or failed) and is invisible to readers. Default names encode the recipe:
`climbmix_t<sequence_len>_<tokenizer fingerprint>` / `sft_t...`. Raw upstream data (ClimbMix shards,
SmolTalk) is never stored here.

Rules: [`tinylab/docs/remote.md`](https://github.com/8kb/tinylab/blob/main/docs/remote.md).
