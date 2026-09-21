"""
Loading and validating the "model_config" job-file key -- a path to a materialized
modelcore.ModelConfig tree, never a depth dial. tinylab does no preset/depth-dial derivation of
its own: that logic (mup_dims, compute_window_sizes, gpt_lambda_schedule, and the PRESETS registry
itself) belongs to nanochat, the architecture playground -- its `scripts/model_info.py
--dump-config` is what produces the file this module loads. See AGENTS.md: "a config tree carries
only concrete, already-decided values, never a derivation rule" now applies to the whole job file,
not just modelcore's own tree, and a preset is exactly a derivation rule.
"""
import json

from modelcore import ModelConfig


def load_model_config(path: str, *, sequence_len: int, vocab_size: int) -> ModelConfig:
    """Hydrates a materialized ModelConfig tree from `path` (already absolutized and existence-
    checked by tinylab.job.resolve_steps, before anything runs) and checks it actually matches the
    step using it. A raw tree carries its own sequence_len/vocab_size, baked in at dump time --
    tinylab's old preset-dict branch silently ignored both, which meant a config dumped for one
    dataset/tokenizer could be trained against a mismatched one with no error at all."""
    with open(path, "r", encoding="utf-8") as f:
        config = ModelConfig.from_dict(json.load(f))
    assert config.sequence_len == sequence_len, (
        f'"model_config" {path!r} was dumped at sequence_len={config.sequence_len}, but this '
        f'step\'s own "sequence_len" is {sequence_len} -- re-dump the config at the right length '
        f'(nanochat: scripts/model_info.py --max-seq-len={sequence_len} --dump-config).'
    )
    assert config.vocab_size == vocab_size, (
        f'"model_config" {path!r} was dumped at vocab_size={config.vocab_size}, but the local '
        f'tokenizer\'s vocab size is {vocab_size} -- both must come from the same tokenizer.'
    )
    return config
