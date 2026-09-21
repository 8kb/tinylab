"""
Job loading and dispatch: the whole of tinylab's "config, not flags" contract lives here. A job
file is one JSON document -- "defaults" deep-merged under each step (and under the sibling "chat"
block that tinylab.chat reads), then an ordered "steps" list dispatched to tinylab.ops.OPS.

No dependency graph: the pipeline *is* the order, and each step's inputs are named artifacts on
disk (a prepared dataset, a checkpoint tag), so `--only` can re-run one step standalone once those
artifacts already exist.

`--resume` (run_file's `resume` param) is the one exception to "config, not flags" -- whether a
particular invocation is a fresh start or a continuation is a fact about the invocation, not the
pipeline, so it lives on the CLI rather than as a job-file key. See run_file's own docstring for
the job state file mechanism this drives.
"""
import difflib
import hashlib
import json
import os
import time

from tinylab.checkpoints import validate_tag
from tinylab.ops import COMMON_KEYS, OPS


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
    """Validates cfg's top-level keys against `accepted`. Public: both resolve_steps (every op)
    and tinylab.chat (the "chat" block) call this, so a typo'd key is a hard error in either
    place, not just one of them."""
    _raise_unknown({k: v for k, v in cfg.items() if k not in ("name", "op")}, accepted, where=where)


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


def _resolve_model_config_path(step: dict, job_dir: str, *, where: str):
    """Mutates step["model_config"] in place from a (possibly relative) path to an absolute one,
    resolved against the job file's own directory, and checks it exists -- before anything runs,
    so a missing/typo'd path is a JobError up front rather than a FileNotFoundError mid-pipeline.
    A step with no "model_config" key is untouched (harmless -- every op besides train kind=base
    ignores it entirely; train itself requires it at run time, not here, since "model_config" is
    a COMMON_KEYS default shared even by steps that never use it)."""
    if "model_config" not in step:
        return
    path = step["model_config"]
    abs_path = path if os.path.isabs(path) else os.path.normpath(os.path.join(job_dir, path))
    if not os.path.isfile(abs_path):
        raise JobError(f'{where}, "model_config": {path!r} does not exist (resolved to {abs_path!r})')
    step["model_config"] = abs_path


def _resolve_tokenizer_specs(step: dict, job_dir: str, *, where: str):
    """Mutates step's "tokenizer" (and a tokenizer op's "output") in place: a value containing a
    path separator is a path, made absolute against the job file's own directory -- the same rule
    "model_config" follows -- while a bare name is left as it is. No existence check: a "tokenizer"
    step may be about to create it."""
    for key in ("tokenizer", "output"):
        if key not in step:
            continue
        value = step[key]
        if not isinstance(value, str) or not value.strip():
            raise JobError(f"{where}, {key!r}: must be a non-empty tokenizer name or path, got {value!r}")
        if "/" in value or os.sep in value:
            expanded = os.path.expanduser(value)
            step[key] = expanded if os.path.isabs(expanded) else os.path.normpath(os.path.join(job_dir, expanded))


def _check_checkpoint_tags(step: dict, *, where: str):
    """A checkpoint tag is arbitrary text with optional "/" folders (see tinylab.checkpoints.
    validate_tag) -- checked here so a bad one is a JobError up front, not a ValueError after a
    prepare step has already spent an hour. A train step with no explicit "output_tag" uses its own
    "name" as the tag, so that gets the same check."""
    for key in ("output_tag", "source_tag", "model_tag"):
        if key in step:
            try:
                validate_tag(step[key])
            except ValueError as e:
                raise JobError(f"{where}, {key!r}: {e}") from None
    if step.get("op") == "train" and "output_tag" not in step:
        try:
            validate_tag(step["name"])
        except ValueError as e:
            raise JobError(f"{where}: this step has no \"output_tag\", so its \"name\" is its checkpoint tag -- {e}") from None


def resolve_steps(job: dict, only: str | None = None, *, job_dir: str = "") -> list[dict]:
    """Deep-merges defaults into every step, validates op + keys, and returns the resolved list
    (filtered to `only` if given). Raises JobError before anything is run. job_dir: the job file's
    own directory, against which a relative "model_config" or "tokenizer" path resolves (default "" == cwd, for
    callers that don't have a real job file path, e.g. most of this repo's own tests)."""
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
        where = f"steps[{i}] ({step['name']!r}, op={op!r})"
        accepted = OPS[op].accepted_keys(step) | COMMON_KEYS
        check_known_keys(step, accepted, where=where)
        _resolve_model_config_path(step, job_dir, where=where)
        _resolve_tokenizer_specs(step, job_dir, where=where)
        _check_checkpoint_tags(step, where=where)
        resolved.append(step)

    if only is not None:
        resolved = [s for s in resolved if s["name"] == only]
        if not resolved:
            raise JobError(f"--only {only!r}: no step with that name (have: {[s['name'] for s in steps]})")
    return resolved


def resolve_chat(job: dict, *, job_dir: str = "") -> dict:
    """Deep-merges defaults under the "chat" block. Returns {} if the job file has none -- callers
    should treat that as "nothing to chat with configured", not a default tag guess. A path-form
    "tokenizer" is made absolute against job_dir and a bad checkpoint tag is a JobError, the same
    as for a step."""
    cfg = _deep_merge(job.get("defaults", {}), job.get("chat", {}))
    if cfg:
        _resolve_tokenizer_specs(cfg, job_dir, where='"chat"')
        _check_checkpoint_tags(cfg, where='"chat"')
    return cfg


def _state_path(job_path: str) -> str:
    """One job state file per job file, under <base_dir>/job_state/ -- not next to the job file
    itself, matching every other piece of tinylab's runtime state (checkpoints, prepared datasets,
    the tokenizer all live under TINYLAB_BASE_DIR, never beside a source-controlled job file).
    Named <stem>-<8 hex chars of sha256(abspath)> so two job files with the same basename in
    different directories don't collide, while staying legible in a directory listing."""
    from tinylab.runtime import get_base_dir
    abs_path = os.path.abspath(job_path)
    stem = os.path.splitext(os.path.basename(job_path))[0]
    digest = hashlib.sha256(abs_path.encode()).hexdigest()[:8]
    state_dir = os.path.join(get_base_dir(), "job_state")
    os.makedirs(state_dir, exist_ok=True)
    return os.path.join(state_dir, f"{stem}-{digest}.json")


def _state_exists(path: str) -> bool:
    """True if any trace of a previous attempt exists -- the current generation, the previous one
    (".old"), or a transient write in flight (".tmp") when something died mid-write. Any of the
    three means a previous run happened and never cleaned up after itself."""
    return any(os.path.exists(path + suffix) for suffix in ("", ".old", ".tmp"))


def _remove_state(path: str) -> None:
    for suffix in ("", ".old", ".tmp"):
        candidate = path + suffix
        if os.path.exists(candidate):
            os.remove(candidate)


def _write_state(path: str, content: dict) -> dict:
    """Atomic, generation-rotated write: write the new content to a temp file (flushed + fsync'ed,
    so it's durable on disk before anything else happens), drop the previous ".old" generation,
    demote the current file to ".old" (atomic rename), then promote the temp file to current
    (atomic rename). A crash at any point in that sequence leaves either the untouched previous
    file or the untouched, already-durable ".old" readable -- never a torn write, regardless of
    how reliable the filesystem's rename semantics actually are (matters on network/overlay
    storage, which a GPU pod's mount can be). "counter" increments on every write -- not needed to
    pick between current/.old (the rotation order alone guarantees which is newer), just a cheap,
    always-available sanity signal for a human inspecting the file by hand. Returns content with
    "counter" filled in, so the caller's own in-memory copy stays in sync with what's now on disk."""
    old_path, tmp_path = path + ".old", path + ".tmp"
    content = dict(content, counter=content.get("counter", 0) + 1)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(old_path):
        os.remove(old_path)
    if os.path.exists(path):
        os.replace(path, old_path)
    os.replace(tmp_path, path)
    return content


def _read_state(path: str) -> dict | None:
    """The current generation if it exists and parses; the previous one if not (a crash mid-write
    left `path` missing or truncated -- prints a warning, since resuming from ".old" means
    resuming from whatever was confirmed one generation ago, possibly one checkpoint behind the
    truest state); None if neither is usable (treated as "nothing to resume from" -- --resume then
    just runs every step fresh, which is safe, if wasteful, rather than raising over a corrupted
    state file -- also warned about, for the same reason)."""
    for i, candidate in enumerate((path, path + ".old")):
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    content = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if i == 1:
                print(f"warning: {path} is missing or unreadable -- resuming from the previous generation ({candidate}) instead, which may be one checkpoint behind.")
            return content
    if os.path.exists(path) or os.path.exists(path + ".old"):
        print(f"warning: neither {path} nor its previous generation could be read -- resuming as a fresh start (no steps will be skipped).")
    return None


def _pid_alive(pid) -> bool:
    """Best-effort liveness check for a job state file's recorded "pid" -- os.kill(pid, 0) sends
    no signal, just asks the OS whether the process exists. Inherently racy (the process could
    exit the instant after this check returns True) and POSIX-specific (this repo's own targets:
    laptop dev, RunPod GPU pods -- see AGENTS.md), not a real mutex; good enough to catch the
    common case (a second run launched by mistake while the first is still going), not a guarantee."""
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # exists but not signalable by us (e.g. owned by another user) -- still alive
    return True


def run_file(job_path: str, *, only: str | None = None, dry_run: bool = False, resume: bool = False) -> list[dict]:
    """Loads, validates, and (unless dry_run) runs every resolved step in order. Returns each
    step's result dict (or, for --dry-run, its fully resolved config instead) -- for a resumed run
    this holds only the steps actually executed *this* invocation, not the ones skipped because
    they'd already completed, so len(results) can be shorter than len(steps).

    Resume/skip is a whole-pipeline concern, active only for a real (not --dry-run), full-pipeline
    (not --only) run: a job state file under <base_dir>/job_state/ (see _write_state/_read_state
    for its crash-safety) records, per step, "done" (skip unconditionally on a later --resume) or
    -- for a `train` step that took at least one periodic checkpoint but didn't finish -- "in
    progress" plus exactly which checkpoint step to resume it from, so a resumed train step never
    has to guess by scanning its checkpoint directory (which could contain a *newer* checkpoint
    file that started writing but never finished -- see tinylab.ops.train).

    A state file that already exists is handled two ways, checked in this order:
    - If its recorded "pid" is still alive (see _pid_alive), this is refused unconditionally --
      even with `resume=True` -- since two processes racing on the same checkpoint files is real
      corruption, not a resumable situation. Delete the state file yourself if that pid has since
      been reused by something unrelated.
    - Otherwise, without `resume=True` it's a hard, immediate error (a previous run died without
      cleanup); pass resume=True to continue it, or delete the state file to force a fresh start.

    resume=True with no existing state file is a no-op (an ordinary fresh run) -- safe to pass
    unconditionally from an unattended restart script. The state file is deleted only when every
    step finishes without raising; a step that fails leaves it in place, exactly as intended."""
    job = load(job_path)
    job_dir = os.path.dirname(os.path.abspath(job_path))
    steps = resolve_steps(job, only=only, job_dir=job_dir)

    if dry_run:
        for step in steps:
            print(json.dumps(step, indent=2))
        return steps

    from tinylab.ops import Context
    from tinylab.runtime import compute_cleanup

    tracking = only is None
    state_path = _state_path(job_path) if tracking else None
    steps_state = {}  # step name -> {"status": "done"} | {"status": "in_progress", "checkpoint_step": N}
    state_content = None
    if tracking:
        if _state_exists(state_path):
            existing = _read_state(state_path)
            # A still-alive recorded pid is refused unconditionally -- even with --resume, which
            # is for continuing after a crash, not running a second instance alongside a live one.
            # Two processes racing on the same checkpoint files is real corruption, not just
            # wasted compute, so this check comes before the resume/no-resume branch below, not
            # folded into it.
            if existing is not None and _pid_alive(existing.get("pid")):
                raise JobError(
                    f"{state_path} records pid={existing['pid']}, which still appears to be "
                    f"running -- refusing to start a second, concurrent run of the same job (this "
                    f"applies even with --resume). If that process has genuinely exited and its "
                    f"pid has since been reused by something unrelated, delete the job state file "
                    f"to force a fresh start."
                )
            if not resume:
                found = ", ".join(p for p in (state_path, state_path + ".old", state_path + ".tmp") if os.path.exists(p))
                raise JobError(
                    f"{found} already exists -- a previous run of this job may still be in "
                    f"progress or crashed without cleanup. Pass --resume to continue it, or "
                    f"delete it to force a fresh start."
                )
            steps_state = existing.get("steps", {}) if existing is not None else {}
        state_content = _write_state(state_path, {
            "job_path": os.path.abspath(job_path), "pid": os.getpid(), "started_at": time.time(),
            "steps": steps_state,
        })

        def _record_progress(name: str, checkpoint_step: int) -> None:
            nonlocal state_content
            steps_state[name] = {"status": "in_progress", "checkpoint_step": checkpoint_step}
            state_content = _write_state(state_path, dict(state_content, steps=steps_state))
    else:
        _record_progress = None

    resume_hints = {name: s["checkpoint_step"] for name, s in steps_state.items()
                     if s.get("status") == "in_progress" and "checkpoint_step" in s}

    # "device" is a COMMON_KEYS default, so every resolved step already carries it -- read it off
    # the first one rather than re-reading job["defaults"] directly, so a step-level override (an
    # unusual but valid case) is honored the same way the rest of a step's config is. One Context
    # is shared for the whole run: tinylab doesn't support switching devices mid-job.
    device_type = steps[0].get("device", "auto") if steps else "auto"
    # Likewise "tokenizer": one per run, read off the first step (path-form values were already
    # made absolute in resolve_steps).
    tokenizer_spec = steps[0].get("tokenizer") if steps else None
    ctx = Context(device_type=device_type, resume=resume, tokenizer_spec=tokenizer_spec,
                  _resume_checkpoint_steps=resume_hints, _on_checkpoint=_record_progress)
    results = []
    try:
        for step in steps:
            if steps_state.get(step["name"], {}).get("status") == "done":
                print(f"=== [{step['name']}] already completed, skipping (--resume) ===")
                continue
            print(f"=== [{step['name']}] op={step['op']} ===")
            result = OPS[step["op"]].run(step, ctx)
            print(json.dumps(result))
            results.append(result)
            steps_state[step["name"]] = {"status": "done"}
            if tracking:
                state_content = _write_state(state_path, dict(state_content, steps=steps_state))
    finally:
        compute_cleanup()
    if tracking:
        _remove_state(state_path)  # only reached on a clean run -- a raised exception leaves it in place
    return results
