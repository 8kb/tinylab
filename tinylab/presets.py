"""
Architecture presets + depth-dial derivation, ported and merged from nanochat's
nanochat/architectures/presets.py and derive.py. expand(name, depth, **kwargs) turns a depth dial
into a concrete modelcore.ModelConfig tree -- modelcore's own invariant is that a config tree
carries only concrete already-decided values, never a derivation rule; these rules (mup_dims,
compute_window_sizes, gpt_lambda_schedule) are what turns "depth 6" into per-layer ints, and this
is the one place in tinylab that knows them. Only the "gpt" preset is kept -- see
docs/architecture.md for what else tinylab drops relative to nanochat.
"""
from modelcore import ComponentSpec, ModelConfig

# -----------------------------------------------------------------------------
# Derivation rules: how a depth dial becomes concrete per-layer values.


def mup_dims(depth: int, aspect_ratio: int, head_dim: int) -> tuple[int, int]:
    """muP-style depth dial: n_embd grows with depth * aspect_ratio, rounded up to a multiple of
    head_dim; n_head = n_embd // head_dim."""
    base_dim = depth * aspect_ratio
    n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = n_embd // head_dim
    return n_embd, n_head


def compute_window_sizes(pattern: str, n_layer: int, sequence_len: int) -> list[int]:
    """Per-layer sliding-window size (-1 = unlimited/full context). `pattern` is tiled across
    layers; L=long (full context), S=short (quarter context, rounded up to a 128-token tile). The
    final layer always gets full context."""
    pattern = pattern.upper()
    assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
    long_window = sequence_len
    short_window = -(-long_window // 4 // 128) * 128
    char_to_window = {"L": long_window, "S": short_window}
    windows = [char_to_window[pattern[i % len(pattern)]] for i in range(n_layer)]
    windows[-1] = long_window
    return windows


def has_value_embed(layer_idx: int, n_layer: int) -> bool:
    """GPT's value-embedding parity rule: alternating, with the last layer always included."""
    return layer_idx % 2 == (n_layer - 1) % 2


def gpt_lambda_schedule(layer_idx: int, n_layer: int) -> tuple[float, float]:
    """GPT's per-layer resid/x0-lambda init schedule: stronger residual & more input-embedding
    blending at early layers, decaying with depth. Returns (resid_lambda_init, x0_lambda_init)."""
    resid_lambda_init = 1.15 - (0.10 * layer_idx / max(n_layer - 1, 1))
    x0_lambda_init = 0.20 - (0.15 * layer_idx / max(n_layer - 1, 1))
    return resid_lambda_init, x0_lambda_init


# -----------------------------------------------------------------------------
# Preset assembly


def assemble_gpt(n_layer, n_head, n_kv_head, n_embd, head_dim, sequence_len, vocab_size, window_pattern,
                  reference=None) -> ModelConfig:
    """Builds the gpt-shaped tree from already-concrete dimensions."""
    windows = compute_window_sizes(window_pattern, n_layer, sequence_len)
    blocks = []
    for i in range(n_layer):
        resid_lambda_init, x0_lambda_init = gpt_lambda_schedule(i, n_layer)
        blocks.append(ComponentSpec("gpt_block", {
            "layer_idx": i, "n_head": n_head, "n_kv_head": n_kv_head, "window": windows[i],
            "has_value_embed": has_value_embed(i, n_layer),
            "resid_lambda_init": resid_lambda_init, "x0_lambda_init": x0_lambda_init,
        }))
    return ModelConfig(
        sequence_len=sequence_len, vocab_size=vocab_size, n_embd=n_embd, reference=reference,
        shared={"rope": ComponentSpec("rotary", {"head_dim": head_dim})},
        input=ComponentSpec("token_embedding", {"smear": True}),
        body=ComponentSpec("backout", {"backout_layer": n_layer // 2, "backout_lambda_init": 0.2, "blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )


def expand_gpt(depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768, window_pattern="SSSL") -> ModelConfig:
    n_embd, n_head = mup_dims(depth, aspect_ratio, head_dim)
    reference = {"preset": "gpt", "kwargs": dict(aspect_ratio=aspect_ratio, head_dim=head_dim,
                                                  max_seq_len=max_seq_len, vocab_size=vocab_size,
                                                  window_pattern=window_pattern)}
    return assemble_gpt(depth, n_head, n_head, n_embd, head_dim, max_seq_len, vocab_size, window_pattern,
                         reference=reference)


PRESETS = {"gpt": expand_gpt}

# The job-file "model" block's allowed keys -- validated by tinylab.job against this set, the same
# way every other step key is validated, so a typo (e.g. "dpeth") is a hard error instead of a
# silently-ignored no-op. "config" is the raw-materialized-tree escape hatch (see
# resolve_model_config); "d_ref_scaling_params" is read directly by tinylab.ops.train, not by any
# function in this module, but belongs to the same block.
MODEL_ACCEPTED_KEYS = {
    "preset", "config", "depth", "aspect_ratio", "head_dim", "window_pattern", "arch_opts",
    "d_ref_scaling_params",
}


def expand(name, depth, **kwargs) -> ModelConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Registered: {sorted(PRESETS)}")
    return PRESETS[name](depth, **kwargs)


def resolve_model_config(model_config, depth, *, aspect_ratio, head_dim, max_seq_len, vocab_size,
                          window_pattern=None, arch_opts=None) -> ModelConfig:
    """Resolve a "model" job-file block into the ModelConfig to actually build at `depth`. Two
    shapes come from the same JSON field: a preset name (str, e.g. "gpt") expands through
    PRESETS; a raw materialized tree (dict) hydrates directly via ModelConfig.from_dict (any
    nested dict carrying "#type" becomes a ComponentSpec automatically) -- e.g. one dumped by a
    prior run's --dry-run output."""
    if isinstance(model_config, dict):
        return ModelConfig.from_dict(model_config)
    kwargs = dict(aspect_ratio=aspect_ratio, head_dim=head_dim, max_seq_len=max_seq_len, vocab_size=vocab_size)
    if window_pattern is not None:
        kwargs["window_pattern"] = window_pattern
    return expand(model_config, depth, **kwargs, **(arch_opts or {}))


def resolve_reference_config(resolved_config: ModelConfig, ref_depth: int) -> ModelConfig:
    """The muP scaling-law reference model (tinylab.ops.train's d_ref) at ref_depth (12), for a
    config resolve_model_config already resolved to `resolved_config`. expand() always stamps a
    `reference` block on its output, so this works uniformly for any preset-derived config."""
    assert resolved_config.reference is not None, (
        f"config has no 'reference' block, so its muP scaling-law reference model can't be "
        f"re-derived automatically at depth {ref_depth}; pass \"d_ref_scaling_params\" instead"
    )
    return expand(resolved_config.reference["preset"], ref_depth, **resolved_config.reference["kwargs"])
