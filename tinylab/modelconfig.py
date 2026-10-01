"""
Loading and validating the "model_config" job-file key -- a path to a materialized
modelcore.ModelConfig tree, never a depth dial. tinylab does no preset/depth-dial derivation of
its own: that logic (mup_dims, compute_window_sizes, gpt_lambda_schedule, and the PRESETS registry
itself) lives in llmllab/tools/ -- its make_config.py is what produces the file this module loads.
See AGENTS.md: "a config tree carries only concrete, already-decided values, never a derivation
rule" applies to the whole job file, not just modelcore's own tree, and a preset is exactly a
derivation rule.
"""
import json

from modelcore import ModelConfig, OptimizerHparams


def load_model_config(path: str, *, sequence_len: int, vocab_size: int) -> ModelConfig:
    """Hydrates a materialized ModelConfig tree from `path` (already absolutized and existence-
    checked by tinylab.job.resolve_steps, before anything runs) and checks it actually matches the
    step using it. A raw tree carries its own sequence_len/vocab_size, baked in at dump time --
    tinylab's old preset-dict branch silently ignored both, which meant a config dumped for one
    dataset/tokenizer could be trained against a mismatched one with no error at all."""
    with open(path, "r", encoding="utf-8") as f:
        config = ModelConfig.from_dict(json.load(f))
    assert config.sequence_len == sequence_len, (
        f'"model_config" {path!r} was generated at sequence_len={config.sequence_len}, but this '
        f'step\'s own "sequence_len" is {sequence_len} -- re-dump the config at the right length '
        f'(llmllab/tools/make_config.py --max-seq-len {sequence_len}).'
    )
    assert config.vocab_size == vocab_size, (
        f'"model_config" {path!r} was generated at vocab_size={config.vocab_size}, but the local '
        f'tokenizer\'s vocab size is {vocab_size} -- both must come from the same tokenizer.'
    )
    return config


def load_override(cfg: dict, *, sequence_len: int, vocab_size: int) -> "ModelConfig | None":
    """An sft/rl step's optional "model_config" is a config-override request (e.g. attach LoRA/DoRA
    adapters to an already-trained base), not a from-scratch build: returns the loaded tree, or None
    when the step names none and the checkpoint's own stored config should be used unchanged."""
    if "model_config" not in cfg:
        return None
    return load_model_config(cfg["model_config"], sequence_len=sequence_len, vocab_size=vocab_size)


# Optimizer dials a step may set that modelcore has its own default for (adapter/conv/ssm groups
# exist only on models that have such parameters).
_OPTIONAL_LR_KEYS = ("adapter_lr", "adapter_scalar_lr", "conv_lr", "ssm_lr")


def optimizer_hparams(cfg: dict, defaults: dict) -> OptimizerHparams:
    """The step's OptimizerHparams: the five always-set dials come from cfg, else `defaults` (the
    calling op's own DEFAULTS); the optional ones only when the job file names them."""
    kwargs = {key: cfg.get(key, defaults[key])
              for key in ("unembedding_lr", "embedding_lr", "matrix_lr", "scalar_lr", "weight_decay")}
    kwargs.update({key: cfg[key] for key in _OPTIONAL_LR_KEYS if key in cfg})
    return OptimizerHparams(**kwargs)
