"""
Job loading and dispatch: the whole of tinylab's "config, not flags" contract lives here. A job
file is one JSON document -- "defaults" deep-merged under each step (and under the sibling "chat"
block that tinylab.chat reads), then an ordered "steps" list dispatched to tinylab.ops.OPS.

No dependency graph: the pipeline *is* the order, and each step's inputs are named artifacts on
disk (a prepared dataset, a checkpoint tag), so `--only` can re-run one step standalone once those
artifacts already exist.
"""
import difflib
import json

from tinylab.ops import COMMON_KEYS, OPS
from tinylab.presets import MODEL_ACCEPTED_KEYS


class JobError(ValueError):
    """A malformed job file: an unknown op, an unknown key, a missing required field. Distinct
    from a runtime failure inside an op (a bad job file is a mistake to fix before spending any
    time/network/GPU; ops raise plain exceptions for everything else)."""


def _deep_merge(base: dict, override: dict) -> dict:
    """override's keys win; nested dicts merge recursively, everything else (including lists) is
    replaced wholesale."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _raise_unknown(d: dict, accepted: set, *, where: str):
    unknown = {k for k in d if not k.startswith("_")} - accepted
    if not unknown:
        return
    for key in sorted(unknown):
        suggestion = difflib.get_close_matches(key, accepted, n=1)
        hint = f" -- did you mean {suggestion[0]!r}?" if suggestion else ""
        raise JobError(f"{where}: unknown key {key!r}{hint} (accepted: {sorted(accepted)})")


def check_known_keys(cfg: dict, accepted: set, *, where: str):
    """Validates cfg's top-level keys against `accepted`, plus -- whenever cfg carries a "model"
    block -- that block's own keys against presets.MODEL_ACCEPTED_KEYS. Public: both
    resolve_steps (every op) and tinylab.chat (the "chat" block) call this, so a typo'd key is a
    hard error in either place, not just one of them."""
    _raise_unknown({k: v for k, v in cfg.items() if k not in ("name", "op")}, accepted, where=where)
    model = cfg.get("model")
    if isinstance(model, dict):
        _raise_unknown(model, MODEL_ACCEPTED_KEYS, where=f'{where}, "model"')


def load(job_path: str) -> dict:
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    if "steps" not in job:
        raise JobError(f"{job_path}: missing required top-level key 'steps'")
    # A leading underscore marks a freeform comment key (JSON has no comment syntax) -- ignored
    # everywhere, never validated.
    unknown_top = {k for k in job if not k.startswith("_")} - {"defaults", "steps", "chat"}
    if unknown_top:
        raise JobError(f"{job_path}: unknown top-level key(s) {sorted(unknown_top)} (accepted: defaults, steps, chat)")
    return job


def resolve_steps(job: dict, only: str | None = None) -> list[dict]:
    """Deep-merges defaults into every step, validates op + keys, and returns the resolved list
    (filtered to `only` if given). Raises JobError before anything is run."""
    defaults = job.get("defaults", {})
    steps = job["steps"]
    names = [s.get("name") for s in steps]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise JobError(f"duplicate step name(s): {dupes}")

    resolved = []
    for i, raw_step in enumerate(steps):
        if "name" not in raw_step:
            raise JobError(f"steps[{i}]: missing required key 'name'")
        if "op" not in raw_step:
            raise JobError(f"steps[{i}] ({raw_step['name']!r}): missing required key 'op'")
        op = raw_step["op"]
        if op not in OPS:
            suggestion = difflib.get_close_matches(op, OPS, n=1)
            hint = f" -- did you mean {suggestion[0]!r}?" if suggestion else ""
            raise JobError(f"steps[{i}] ({raw_step['name']!r}): unknown op {op!r}{hint} (available: {sorted(OPS)})")
        step = _deep_merge(defaults, raw_step)
        accepted = OPS[op].accepted_keys(step) | COMMON_KEYS
        check_known_keys(step, accepted, where=f"steps[{i}] ({step['name']!r}, op={op!r})")
        resolved.append(step)

    if only is not None:
        resolved = [s for s in resolved if s["name"] == only]
        if not resolved:
            raise JobError(f"--only {only!r}: no step with that name (have: {[s['name'] for s in steps]})")
    return resolved


def resolve_chat(job: dict) -> dict:
    """Deep-merges defaults under the "chat" block. Returns {} if the job file has none -- callers
    should treat that as "nothing to chat with configured", not a default source/tag guess."""
    return _deep_merge(job.get("defaults", {}), job.get("chat", {}))


def run_file(job_path: str, *, only: str | None = None, dry_run: bool = False) -> list[dict]:
    """Loads, validates, and (unless dry_run) runs every resolved step in order. Returns each
    step's result dict (or, for --dry-run, its fully resolved config instead)."""
    job = load(job_path)
    steps = resolve_steps(job, only=only)

    if dry_run:
        for step in steps:
            print(json.dumps(step, indent=2))
        return steps

    from tinylab.ops import Context
    from tinylab.runtime import compute_cleanup
    # "device" is a COMMON_KEYS default, so every resolved step already carries it -- read it off
    # the first one rather than re-reading job["defaults"] directly, so a step-level override (an
    # unusual but valid case) is honored the same way the rest of a step's config is. One Context
    # is shared for the whole run: tinylab doesn't support switching devices mid-job.
    device_type = steps[0].get("device", "auto") if steps else "auto"
    ctx = Context(device_type=device_type)
    results = []
    try:
        for step in steps:
            print(f"=== [{step['name']}] op={step['op']} ===")
            result = OPS[step["op"]].run(step, ctx)
            print(json.dumps(result))
            results.append(result)
    finally:
        compute_cleanup()
    return results
