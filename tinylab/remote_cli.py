"""
`python -m tinylab remote ...` -- manual access to the bucket (see docs/remote.md):

    remote ls [prefix]                       entities in the bucket: kind, steps, size, README
    remote pull <entity> [--step N] [--optim]
    remote push <entity> [--steps 1,2] [--optim] [--force]   also how to redo a failed background push
    remote check                             token/write access, READMEs to write, incomplete steps
    remote rm <entity>[@step] [--yes]        the only manual delete; asks first
    remote push-docs                         uploads tinylab/remote_readmes/ (the type READMEs) to the bucket

An <entity> is a bucket path: tokenizers/<name>, prepared/<dataset>, checkpoints/<tag>,
experiments/<experiment>, eval_bundle, task_data/<...>. The remote is `--remote URL`, else
$TINYLAB_REMOTE, else DEFAULT_REMOTE.
"""
from __future__ import annotations

import argparse
import os
import sys

from tinylab import readme
from tinylab import remote as R
from tinylab.runtime import get_base_dir

DEFAULT_REMOTE = "hf://buckets/mendel-il/tinylab-data"
README_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_readmes")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def _split_step(entity: str) -> tuple[str, int | None]:
    if "@" in entity:
        base, _, step = entity.rpartition("@")
        return base.strip("/"), int(step)
    return entity.strip("/"), None


def _local(entity: str) -> str:
    return os.path.join(get_base_dir(), *entity.split("/"))


# ---- ls --------------------------------------------------------------------------------------

def cmd_ls(remote: R.Remote, args) -> int:
    listing = remote.list(args.prefix or "")
    found = R.entities(listing)
    if not found:
        print(f"{remote.url}: nothing under {args.prefix or '/'}")
        return 0
    for id_ in sorted(found):
        e = found[id_]
        extra = ""
        if e["kind"] == "checkpoint":
            steps = R.complete_steps(listing, id_)
            partial = sorted(set(R.step_files(listing, id_)) - set(steps))
            extra = f" steps={steps}" + (f" INCOMPLETE={partial}" if partial else "")
        elif e["kind"] == "dataset" and f"{id_}/manifest.json" not in listing:
            extra = " INCOMPLETE (no manifest)"
        readme_flag = "" if f"{id_}/README.md" in listing else "  [no README]"
        print(f"{e['kind']:<11} {id_}  {_human(e['bytes'])} in {e['files']} file(s){extra}{readme_flag}")
    return 0


# ---- pull ------------------------------------------------------------------------------------

def cmd_pull(remote: R.Remote, args) -> int:
    entity, at_step = _split_step(args.entity)
    step = args.step if args.step is not None else at_step
    top = entity.split("/")[0]
    if top == R.CHECKPOINTS:
        got = R.pull_checkpoint(remote, entity[len(R.CHECKPOINTS) + 1:], step=step, optim=args.optim)
        print(f"pulled {entity} step {got} -> {_local(entity)}")
    elif top == R.TOKENIZERS:
        fetched = R.pull_tokenizer(remote, entity.split("/")[1])
        print(f"pulled {entity}: {len(fetched)} file(s)")
    elif top == R.PREPARED:
        if f"{entity}/manifest.json" not in remote.list(entity):
            print(f"error: {entity} has no manifest.json in the bucket -- incomplete, not pulling", file=sys.stderr)
            return 1
        print(f"pulled {entity}: {len(R.pull_prefix(remote, entity))} file(s)")
    else:
        print(f"pulled {entity}: {len(R.pull_prefix(remote, entity))} file(s)")
    return 0


# ---- push ------------------------------------------------------------------------------------

def _existing_producer(remote: R.Remote, entity: str) -> dict:
    text = remote.get_bytes(f"{entity}/README.md")
    if text:
        producer = readme.split(text.decode("utf-8"))[0].get("producer")
        if producer:
            return producer  # re-pushing to an existing entity is the same producer, by definition
    return R.make_producer("manual", "cli", "push", R.describe_hardware())


def cmd_push(remote: R.Remote, args) -> int:
    entity, _ = _split_step(args.entity)
    top = entity.split("/")[0]
    src = _local(entity)
    if not os.path.isdir(src):
        print(f"error: {src} does not exist locally", file=sys.stderr)
        return 1
    producer = _existing_producer(remote, entity)
    if top == R.CHECKPOINTS:
        tag = entity[len(R.CHECKPOINTS) + 1:]
        local_steps = sorted(R.step_files({os.path.join(entity, n): 0 for n in os.listdir(src)}, entity))
        steps = [int(s) for s in args.steps.split(",")] if args.steps else local_steps[-1:]
        if not steps:
            print(f"error: no checkpoint files in {src}", file=sys.stderr)
            return 1
        for step in steps:
            R.push_checkpoint_step(remote, src, tag, step, push_model="all", push_optim="all" if args.optim else "none",
                                   ranks=range(int(os.environ.get("WORLD_SIZE", 1))), producer=producer, force=args.force)
            print(f"pushed {entity} step {step}")
        return 0
    marker = {R.TOKENIZERS: "tokenizer.pkl", R.PREPARED: "manifest.json"}.get(top)
    if top in (R.TOKENIZERS, R.PREPARED):
        existing = R.gate_producer(remote, entity, producer, force=args.force)
        pushed = R.push_dir(remote, src, entity, marker_name=marker, force=args.force)
        R.write_readme(remote, entity, existing, kind="tokenizer" if top == R.TOKENIZERS else "dataset", producer=producer,
                       history="pushed (cli)")
    else:
        pushed = R.push_dir(remote, src, entity, marker_name=None, force=args.force)
    print(f"pushed {entity}: {len(pushed)} file(s) uploaded")
    return 0


# ---- check -----------------------------------------------------------------------------------

def cmd_check(remote: R.Remote, args) -> int:
    try:
        who = remote.check_write()
        print(f"ok: can write to {remote.url} as {who}")
    except R.RemoteError as e:
        print(f"WARNING: cannot write to {remote.url}: {e}")
    listing = remote.list("")
    found = R.entities(listing)
    problems = 0

    prefixes = set()
    for id_, e in found.items():
        if e["kind"] == "experiment":
            text = remote.get_bytes(f"{id_}/README.md")
            if text:
                prefix = readme.split(text.decode("utf-8"))[0].get("tag_prefix")
                if prefix:
                    prefixes.add(prefix)

    need_motivation = []
    for id_, e in sorted(found.items()):
        if e["kind"] == "bench-data":
            continue
        text = remote.get_bytes(f"{id_}/README.md")
        if text is None:
            print(f"missing README: {id_}")
            problems += 1
        elif readme.missing_motivation(text.decode("utf-8")):
            need_motivation.append(id_)
        if e["kind"] == "checkpoint":
            steps = R.step_files(listing, id_)
            for s, f in sorted(steps.items()):
                if not (f["meta"] and f["model"]):
                    print(f"incomplete step: {id_}@{s} (has {[k for k in ('model', 'meta') if f[k]] + (['optim'] if f['optim'] else [])}; "
                          f"needs model + meta)")
                    problems += 1
            first = id_.split("/")[1]
            if prefixes and first not in prefixes and first != "scratch":
                print(f"foreign tag prefix: {id_} (known experiment prefixes: {sorted(prefixes)})")
                problems += 1
        elif e["kind"] == "dataset" and f"{id_}/manifest.json" not in listing:
            print(f"incomplete dataset (no manifest): {id_}")
            problems += 1
    if need_motivation:
        print("README needs a Motivation:")
        for id_ in need_motivation:
            print(f"  {id_}")
    print(f"{len(found)} entities, {problems} problem(s), {len(need_motivation)} README(s) to write")
    return 0


# ---- rm --------------------------------------------------------------------------------------

def cmd_rm(remote: R.Remote, args) -> int:
    entity, step = _split_step(args.entity)
    listing = remote.list(entity)
    if not listing:
        print(f"error: nothing at {entity}", file=sys.stderr)
        return 1
    if step is not None:
        if not entity.startswith(R.CHECKPOINTS + "/"):
            print("error: @step only applies to checkpoints/<tag>", file=sys.stderr)
            return 1
        f = R.step_files(listing, entity).get(step)
        if not f:
            print(f"error: {entity} has no step {step}", file=sys.stderr)
            return 1
        doomed = [p for p in (f["meta"], f["model"]) if p] + list(f["optim"].values())  # marker first
    else:
        doomed = list(listing)
    total = sum(listing[p] for p in doomed)
    print(f"about to PERMANENTLY delete {len(doomed)} file(s), {_human(total)}, from {remote.url} (the bucket is not versioned):")
    for p in doomed[:10]:
        print(f"  {p}")
    if len(doomed) > 10:
        print(f"  ... and {len(doomed) - 10} more")
    if not args.yes and input("type 'delete' to confirm: ").strip() != "delete":
        print("aborted")
        return 1
    remote.delete(doomed)
    text = remote.get_bytes(f"{entity}/README.md")
    if step is not None and text:
        remaining = R.complete_steps(remote.list(entity), entity)
        remote.put_bytes(f"{entity}/README.md", readme.update(
            text.decode("utf-8"), meta_updates={"steps": remaining}, history=f"removed step {step} (cli)").encode("utf-8"))
    print(f"deleted {len(doomed)} file(s)")
    return 0


# ---- push-docs -------------------------------------------------------------------------------

def cmd_push_docs(remote: R.Remote, args) -> int:
    pairs = []
    for root, _, names in os.walk(README_DIR):
        for n in sorted(names):
            path = os.path.join(root, n)
            pairs.append((path, os.path.relpath(path, README_DIR).replace(os.sep, "/")))
    if not pairs:
        print(f"error: no files under {README_DIR}", file=sys.stderr)
        return 1
    remote.upload(pairs)  # docs are mutable by design: the source of truth is git
    print("pushed docs: " + ", ".join(r for _, r in pairs))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tinylab remote", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--remote", default=os.environ.get("TINYLAB_REMOTE", DEFAULT_REMOTE))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ls"); s.add_argument("prefix", nargs="?", default="")
    s = sub.add_parser("pull"); s.add_argument("entity"); s.add_argument("--step", type=int); s.add_argument("--optim", action="store_true")
    s = sub.add_parser("push"); s.add_argument("entity"); s.add_argument("--steps"); s.add_argument("--optim", action="store_true")
    s.add_argument("--force", action="store_true", help="overwrite / take over an existing entity (CLI only, never a job-file key)")
    sub.add_parser("check")
    s = sub.add_parser("rm"); s.add_argument("entity"); s.add_argument("--yes", action="store_true")
    sub.add_parser("push-docs")
    return p


COMMANDS = {"ls": cmd_ls, "pull": cmd_pull, "push": cmd_push, "check": cmd_check, "rm": cmd_rm, "push-docs": cmd_push_docs}


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.cmd](R.open_remote(args.remote), args)
    except (R.RemoteError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
