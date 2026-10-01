"""
`python -m tinylab info`: a read-only inspector. Reports parameters, FLOPs and KV-cache bytes for a
config file or a trained checkpoint tag, plus the training plan a given compute target implies --
without a GPU, without data, without loading weights. Ported from the stats half of our nanochat fork's
scripts/model_info.py (see llmllab/docs/history.md); the depth dial itself lives in
llmllab/tools/make_config.py.

    python -m tinylab info <config.json | checkpoint-tag> [--target-flops X | --target-param-data-ratio R]
        [--num-iterations N] [--total-batch-size B] [--weight-decay W] [--d-ref-scaling-params N]
        [--gpu NAME --num-gpus N --mfu F] [--kv-batch-size N] [--list-targets] [--json]

Every number is computed on the meta device (shapes only). The training plan
(modelcore.scaling.derive_training_plan) is a *report*: it is never written into a job file, so
the job file's "no derivation rules" invariant (see AGENTS.md) holds -- read the numbers off, paste
them into the step's keys.

A config file always gets a plan (default --target-param-data-ratio 12); a checkpoint tag only when
a plan flag is given, since a trained model's own horizon is already in its meta. For a tag only
meta_<step>.json is read: val bpb, wall time, whether the tokenizer's fingerprint matches, and the
adapter inventory. A tag's latest step is used.
"""
import argparse
import contextlib
import io
import json
import os
import sys

from tinylab import checkpoints
from tinylab.ops import train
from tinylab.runtime import get_base_dir

DEFAULT_TARGET_PARAM_DATA_RATIO = 12.0
DEFAULT_WEIGHT_DECAY = train.DEFAULTS["base"]["weight_decay"]


def _meta_model(config):
    """A meta-device Model (shapes only, no weights): adapter targets/inventory are structural."""
    import torch
    from modelcore import Model
    with torch.device("meta"):
        return Model(config)


def list_linear_targets(model) -> list:
    """Every module FQN that resolves to a modelcore Linear -- every legal AdapterSpec.target."""
    from modelcore.components.linear import Linear
    return [name for name, module in model.named_modules() if isinstance(module, Linear)]


def adapter_inventory(model) -> list:
    from modelcore import find_adapters
    rows = []
    for fqn, module in find_adapters(model):
        for name, delta in module.deltas.items():
            rows.append({"target": fqn, "name": name, "type": type(delta).__name__,
                         "enabled": bool(module.enabled[name]),
                         "params": sum(p.numel() for p in delta.parameters())})
    return rows


def read_checkpoint_meta(tag, base_dir=None):
    """(meta dict, step) of `tag`'s latest step -- meta_<step>.json only, no weights."""
    checkpoint_dir = checkpoints.resolve_checkpoint_dir(tag, base_dir)
    step = checkpoints.find_last_step(checkpoint_dir)
    with open(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"), "r", encoding="utf-8") as f:
        return json.load(f), step


def _load_config(source):
    """(ModelConfig, meta, step): `source` is a config JSON path, or a checkpoint tag."""
    from modelcore import ModelConfig
    if source.endswith(".json") or os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as f:
            return ModelConfig.from_dict(json.load(f)), None, None
    meta, step = read_checkpoint_meta(source)
    return ModelConfig.from_dict(meta["model_config"]), meta, step


def mixer_types(config) -> list:
    """The distinct token-mixer types of a config's blocks, in order of first appearance
    (["attention"], or ["short_conv", "attention"] for a hybrid) -- what the tree really contains,
    unlike `reference.preset`, which only names the preset a hybrid was started from."""
    found = []

    def walk(spec):
        if spec.type == "block":
            mixer = spec.params["mixer"].type
            if mixer not in found:
                found.append(mixer)
        for value in spec.params.values():
            for child in (value if isinstance(value, list) else [value]):
                if hasattr(child, "type") and hasattr(child, "params"):
                    walk(child)

    walk(config.body)
    return found


def stats_block(manager, config, kv_batch_size=1) -> dict:
    stats = manager.stats(config)
    reference = config.reference or {}
    return {
        "mixers": mixer_types(config),
        "preset": reference.get("preset"),
        "depth": stats.n_layer,
        "shape": {**stats.shape_summary, "num_kv_slots": stats.kv_cache_spec["num_kv_slots"]},
        "params": {**stats.params_by_role, "total": stats.num_params, "matmul": stats.num_matmul_params,
                   "scaling": stats.num_scaling_params},
        "flops": {"per_token": stats.flops_per_token,
                  "prefill_at_seqlen": stats.prefill_flops(config.sequence_len),
                  "decode_at_seqlen": stats.decode_flops(config.sequence_len)},
        "kv_cache": {"bytes_per_token": stats.kv_bytes_per_token(),
                     "total_mb_at_seqlen": stats.kv_bytes_per_token() * config.sequence_len * kv_batch_size / 1e6,
                     "kv_batch_size": kv_batch_size},
    }


def training_plan_block(row, args) -> dict:
    """The plan modelcore.scaling.derive_training_plan derives for this model's own numbers. d_ref
    is --d-ref-scaling-params, else the model's own scaling params (right when this *is* the d12
    muP reference; llmllab/tools/make_config.py prints the reference's number for any other depth)."""
    from modelcore import derive_training_plan
    scaling = row["params"]["scaling"]
    d_ref = args.d_ref_scaling_params if args.d_ref_scaling_params is not None else scaling
    plan = derive_training_plan(
        num_scaling_params=scaling, d_ref_scaling_params=d_ref, num_flops_per_token=row["flops"]["per_token"],
        target_param_data_ratio=args.target_param_data_ratio, target_flops=args.target_flops,
        num_iterations=args.num_iterations, total_batch_size=args.total_batch_size, weight_decay=args.weight_decay)
    # gpu_hours is the billed resource (independent of num_gpus); wall_clock_hours is how long it
    # takes to finish (divided by num_gpus). Don't conflate the two when pricing a run.
    gpu_hours = wall_clock_hours = None
    if args.gpu is not None:
        from modelcore import peak_flops
        gpu_hours = plan.total_flops / (peak_flops(args.gpu) * args.mfu) / 3600
        wall_clock_hours = gpu_hours / args.num_gpus
    return {
        "target_tokens": plan.target_tokens, "total_batch_size": plan.total_batch_size,
        "auto_batch_size": plan.auto_batch_size, "num_iterations": plan.num_iterations,
        "horizon_source": plan.horizon_source, "total_tokens": plan.total_tokens, "total_flops": plan.total_flops,
        "batch_lr_scale": plan.batch_lr_scale, "weight_decay_scaled": plan.weight_decay_scaled,
        "d_ref_scaling_params": d_ref,
        "d_ref_source": "--d-ref-scaling-params" if args.d_ref_scaling_params is not None else "this model itself",
        "gpu_hours": gpu_hours, "wall_clock_hours": wall_clock_hours,
    }


def _fingerprint_status(meta, config):
    """"match" / "MISMATCH" / "unknown": the checkpoint's recorded tokenizer fingerprint against
    the tokenizer its own config names, if that one is available locally."""
    recorded = meta.get("tokenizer_fingerprint")
    if recorded is None:
        return "unknown"
    from tinylab.tokenizer import get_tokenizer
    name = (config.tokenizer or {}).get("name") if isinstance(config.tokenizer, dict) else None
    try:
        local = get_tokenizer(tokenizer=name).fingerprint()
    except FileNotFoundError:
        return "unknown"
    return "match" if recorded == local else "MISMATCH"


def trained_block(tag, meta, step, config, model) -> dict:
    total_batch_size = meta.get("total_batch_size")
    return {
        "tag": tag, "step": step,
        "tokens_trained": step * total_batch_size if total_batch_size is not None else None,
        "val_bpb": meta.get("val_bpb"), "core_metric": meta.get("core_metric"),
        "train_time_sec": meta.get("total_training_time"),
        "tokenizer_fingerprint_status": _fingerprint_status(meta, config),
        "adapters": adapter_inventory(model),
    }


def inspect(source, args, manager) -> dict:
    config, meta, step = _load_config(source)
    row = stats_block(manager, config, args.kv_batch_size)
    row["source"] = source
    explicit = (args.target_param_data_ratio is not None or args.target_flops > 0 or args.num_iterations > 0
                or args.total_batch_size > 0)
    if args.target_param_data_ratio is None:
        args.target_param_data_ratio = DEFAULT_TARGET_PARAM_DATA_RATIO
    if meta is None or explicit:
        row["training_plan"] = training_plan_block(row, args)
    if meta is not None:
        row["trained"] = trained_block(source, meta, step, config, _meta_model(config))
    return row


def print_human(row):
    shape, params, flops, kv = row["shape"], row["params"], row["flops"], row["kv_cache"]
    label = row["source"] if "trained" not in row else row["trained"]["tag"]
    print(f"\n{'=' * 80}\n{label}\n{'=' * 80}")
    print(f"  n_layer={shape['n_layer']} n_embd={shape['n_embd']} n_head={shape['n_head']} n_kv_head={shape['n_kv_head']} "
          f"sequence_len={shape['sequence_len']} window={shape['window']!r} num_kv_slots={shape['num_kv_slots']}")
    print("  Params by role:")
    for key, value in params.items():
        print(f"    {key:16s}: {value:,}")
    print(f"  FLOPs/token: {flops['per_token']:.3e}  |  prefill@seqlen: {flops['prefill_at_seqlen']:.3e}  |  decode@seqlen: {flops['decode_at_seqlen']:.3e}")
    print(f"  KV cache: {kv['bytes_per_token']:,} bytes/token  |  {kv['total_mb_at_seqlen']:.2f} MB @ seqlen x batch={kv['kv_batch_size']}")
    if "training_plan" in row:
        plan = row["training_plan"]
        print(f"  Training plan: {plan['total_tokens']:,} tokens ({plan['horizon_source']}) over {plan['num_iterations']:,} iters "
              f"@ batch={plan['total_batch_size']:,}{' (auto)' if plan['auto_batch_size'] else ''}  |  {plan['total_flops']:.3e} FLOPs")
        print(f"  Scaling corrections: batch LR scale {plan['batch_lr_scale']:.4f}, weight decay {plan['weight_decay_scaled']:.4f} "
              f"(d_ref scaling params {plan['d_ref_scaling_params']:,}: {plan['d_ref_source']})")
        if plan["gpu_hours"] is not None:
            print(f"  Estimated GPU-hours: {plan['gpu_hours']:.2f}  (~{plan['wall_clock_hours'] * 60:.0f} min wall-clock)")
    if "trained" in row:
        t = row["trained"]
        tokens = f"{t['tokens_trained']:,}" if t["tokens_trained"] is not None else "?"
        bpb = f"{t['val_bpb']:.6f}" if t["val_bpb"] is not None else "?"
        core = f"{t['core_metric']:.4f}" if t["core_metric"] is not None else "not evaluated"
        wall = f"{t['train_time_sec'] / 60:.2f}m" if t["train_time_sec"] is not None else "?"
        print(f"  Trained: step {t['step']:,}  |  {tokens} tokens  |  val bpb: {bpb}  |  CORE: {core}  |  wall time: {wall}")
        print(f"  Tokenizer fingerprint: {t['tokenizer_fingerprint_status']}")
        for a in t["adapters"]:
            print(f"  Adapter {a['name']!r} on {a['target']} ({a['type']}, {'enabled' if a['enabled'] else 'disabled'}, {a['params']:,} params)")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m tinylab info", description=__doc__.split("\n\n")[0])
    p.add_argument("source", help="a model config JSON file, or a checkpoint tag")
    p.add_argument("--target-param-data-ratio", type=float, default=None,
                   help=f"tokens per scaling param (default {DEFAULT_TARGET_PARAM_DATA_RATIO:g})")
    p.add_argument("--target-flops", type=float, default=-1.0)
    p.add_argument("--num-iterations", type=int, default=-1)
    p.add_argument("--total-batch-size", type=int, default=-1)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--d-ref-scaling-params", type=int, default=None,
                   help="scaling params of the d12 muP reference (default: this model's own)")
    p.add_argument("--gpu", default=None, help="GPU name for a GPU-hours estimate, e.g. 'NVIDIA H100'")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--mfu", type=float, default=0.4, help="assumed MFU for the GPU-hours estimate")
    p.add_argument("--kv-batch-size", type=int, default=1, help="batch size for the reported total KV-cache MB")
    p.add_argument("--list-targets", action="store_true", help="print every adaptable Linear FQN, then exit")
    p.add_argument("--json", action="store_true")
    return p


def main(argv) -> int:
    args = build_parser().parse_args(argv)
    from modelcore import ModelManager
    manager = ModelManager()
    # Model construction logs diagnostics ("Padding vocab_size...") to stdout, which would pollute --json.
    sink = io.StringIO() if args.json else None
    try:
        with contextlib.redirect_stdout(sink) if sink is not None else contextlib.nullcontext():
            if args.list_targets:
                config, _meta, _step = _load_config(args.source)
                targets = list_linear_targets(_meta_model(config))
                adapted = {a["target"] for a in adapter_inventory(_meta_model(config))}
                row = None
            else:
                row = inspect(args.source, args, manager)
    except FileNotFoundError as e:
        print(f"error: {args.source!r} is neither a config file nor a checkpoint tag under {get_base_dir()}: {e}", file=sys.stderr)
        return 1
    if args.list_targets:
        if args.json:
            print(json.dumps([{"target": t, "has_adapter": t in adapted} for t in targets], indent=2))
        else:
            print(f"Adaptable Linear targets for {args.source!r}:")
            for t in targets:
                print(f"  {t}" + ("  (has adapter)" if t in adapted else ""))
        return 0
    if args.json:
        print(json.dumps(row, indent=2))
    else:
        print_human(row)
    return 0
