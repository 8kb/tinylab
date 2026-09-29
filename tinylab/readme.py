"""
The README.md every bucket entity carries (see docs/remote.md). Code owns the YAML frontmatter and
appends lines to "## History"; every other section is human-owned and never rewritten by code.

The frontmatter is a strict YAML subset with no dependency: one `key: <json value>` line per key
(a JSON value is a valid YAML flow value), so it round-trips through `json` alone.
"""
import json
from datetime import datetime, timezone

SECTIONS = ("Motivation", "History", "Related", "Notes")
_FENCE = "---"
_PLACEHOLDER = "_TODO: say why this exists._"

# Frontmatter key order when rendering; anything else follows alphabetically.
_KEY_ORDER = ("kind", "id", "created", "producer", "inputs", "retention", "steps", "metrics")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def split(text: str) -> tuple[dict, str]:
    """(frontmatter, body). A text with no frontmatter block gives ({}, text)."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != _FENCE:
        return {}, text
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == _FENCE)
    except StopIteration:
        return {}, text
    meta = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        try:
            meta[key.strip()] = json.loads(value.strip())
        except json.JSONDecodeError:
            meta[key.strip()] = value.strip()
    return meta, "\n".join(lines[end + 1:])


def _render_meta(meta: dict) -> str:
    keys = [k for k in _KEY_ORDER if k in meta] + sorted(k for k in meta if k not in _KEY_ORDER)
    return "\n".join([_FENCE] + [f"{k}: {json.dumps(meta[k], sort_keys=True)}" for k in keys] + [_FENCE])


def join(meta: dict, body: str) -> str:
    return _render_meta(meta) + "\n" + body.lstrip("\n")


def _related_lines(inputs: dict, id_: str) -> list[str]:
    """Relative links from this entity's folder to each input entity's folder -- clickable in the
    bucket's file browser."""
    depth = len(id_.split("/"))
    up = "../" * depth
    return [f"- {role}: [{value}]({up}{value}/README.md)" for role, value in inputs.items()
            if isinstance(value, str) and "/" in value.split("@")[0]]


def create(kind: str, id_: str, *, producer: dict, inputs: dict | None = None, motivation: str = "",
           retention: dict | None = None, steps: list | None = None, metrics: dict | None = None,
           extra: dict | None = None, history: str = "created") -> str:
    """A new entity README. `motivation` seeds the Motivation section (from a step's "_comment");
    empty leaves a placeholder that `missing_motivation` reports."""
    inputs = inputs or {}
    meta = {"kind": kind, "id": id_, "created": now(), "producer": producer, "inputs": inputs}
    if retention is not None:
        meta["retention"] = retention
    if steps is not None:
        meta["steps"] = steps
    if metrics is not None:
        meta["metrics"] = metrics
    meta.update(extra or {})
    title = id_.split("/", 1)[-1] if kind == "checkpoint" else id_.rsplit("/", 1)[-1]
    body = "\n".join([
        f"# {title}", "",
        "## Motivation", "", motivation.strip() or _PLACEHOLDER, "",
        "## History", "", f"- {meta['created']} {history}", "",
        "## Related", "", *(_related_lines(inputs, id_) or ["_none_"]), "",
        "## Notes", "", "",
    ])
    return join(meta, body)


def _section_bounds(lines: list[str], name: str):
    """(start, end) of the lines *after* the `## name` heading up to the next `## ` heading, or
    None if the section doesn't exist."""
    for i, line in enumerate(lines):
        if line.strip() == f"## {name}":
            end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
            return i + 1, end
    return None


def update(text: str, *, meta_updates: dict | None = None, history: str | None = None) -> str:
    """Merges `meta_updates` into the frontmatter (top-level keys replaced) and appends one line
    to "## History"; every other byte of the body is untouched."""
    meta, body = split(text)
    meta.update(meta_updates or {})
    if history:
        lines = body.split("\n")
        bounds = _section_bounds(lines, "History")
        entry = f"- {now()} {history}"
        if bounds is None:
            lines += ["", "## History", "", entry, ""]
        else:
            _, end = bounds
            insert_at = end
            while insert_at > bounds[0] and not lines[insert_at - 1].strip():
                insert_at -= 1
            lines.insert(insert_at, entry)
        body = "\n".join(lines)
    return join(meta, body)


def section(text: str, name: str) -> str | None:
    _, body = split(text)
    lines = body.split("\n")
    bounds = _section_bounds(lines, name)
    if bounds is None:
        return None
    return "\n".join(lines[bounds[0]:bounds[1]]).strip()


def missing_motivation(text: str) -> bool:
    """True if Motivation is absent, empty, or still the generated placeholder."""
    content = section(text, "Motivation")
    return not content or content == _PLACEHOLDER


def same_producer(a: dict, b: dict) -> bool:
    """Two producer records name the same producing step. version/git/hardware are deliberately
    not compared: a resumed run may legitimately be a later commit on different hardware."""
    keys = ("repo", "experiment", "job", "step")
    return all(a.get(k) == b.get(k) for k in keys)
