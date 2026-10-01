"""
The central store: an HF bucket (`hf://buckets/<owner>/<name>`) mirroring `base_dir` 1:1 -- a path
in the bucket is the path relative to TINYLAB_BASE_DIR, with no mapping. See docs/remote.md for the
layout and the rules this module enforces (marker-last uploads, immutability, retention-only
deletion, no `sync --delete`).

Three pieces:
- `Remote`: a thin bucket wrapper (`HFRemote` for real, `MemoryRemote` for tests/dry runs) -- list,
  download, upload, delete, all with retry + backoff.
- `Uploader`: one background thread with a FIFO queue of push closures. A marker task (a
  checkpoint step's meta, a dataset's manifest) is skipped if an earlier task of the same entity
  failed, so a marker never claims completeness the files behind it don't have. Errors never raise
  mid-training: they surface on `flush()`.
- `Prefetcher`: background shard downloads, in manifest order, with a lookahead window.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import queue
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from tinylab import readme
from tinylab.runtime import get_base_dir, print0

HF_PREFIX = "hf://buckets/"
# The top-level folders of the bucket (they mirror base_dir's own layout); an entity id is
# "<folder>/<name>".
CHECKPOINTS, TOKENIZERS, PREPARED, TASK_DATA = "checkpoints", "tokenizers", "prepared", "task_data"
RETRIES = 4
BACKOFF_SECONDS = 2.0

# Never synchronized, in either direction (rel paths, relative to base_dir).
NEVER_SYNC_GLOBS = ("base_data_*", "base_data_*/*", "job_state/*", "locks/*", "*.tmp", "*.tmp.npy", "*.part", "*.lock",
                    "eval_bundle.zip", "task_data/*smoltalk*", "task_data/*smoltalk*/*", "*/__pycache__/*", ".DS_Store")

_MODEL_RE = re.compile(r"^model_(\d+)\.pt$")
_META_RE = re.compile(r"^meta_(\d+)\.json$")
_OPTIM_RE = re.compile(r"^optim_(\d+)_rank(\d+)\.pt$")


class RemoteError(RuntimeError):
    pass


class ImmutableError(RemoteError):
    """A push would overwrite a complete, already-uploaded artifact with different content."""


class ProducerConflict(RemoteError):
    """The entity already exists in the bucket, produced by a different step."""


class UploadError(RemoteError):
    """Raised by Uploader.flush(): one or more background pushes failed (after retries)."""


def is_syncable(rel: str) -> bool:
    rel = rel.replace(os.sep, "/")
    base = os.path.basename(rel)
    return not any(fnmatch.fnmatch(rel, g) or ("/" not in g and fnmatch.fnmatch(base, g)) for g in NEVER_SYNC_GLOBS)


# ---------------------------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------------------------

def _retriable(e: Exception) -> bool:
    if any("NotFound" in cls.__name__ for cls in type(e).__mro__):  # a missing file/bucket won't appear by waiting
        return False
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return not isinstance(e, (PermissionError, FileNotFoundError, KeyboardInterrupt))


def _retry(fn, *, what: str):
    for attempt in range(RETRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- classified by _retriable
            if attempt == RETRIES - 1 or not _retriable(e):
                raise
            delay = BACKOFF_SECONDS * (2 ** attempt)
            print0(f"remote: {what} failed ({e!r}); retrying in {delay:.0f}s")
            _sleep(delay)


_sleep = time.sleep  # patched by tests


class Remote:
    """Interface. Paths are '/'-separated, relative to the bucket root."""
    url: str

    def list(self, prefix: str = "") -> dict[str, int]:
        """Every file at or under `prefix` (a folder), recursively: {path: size}."""
        raise NotImplementedError

    def _fetch(self, pairs: list[tuple[str, str]]) -> None:
        raise NotImplementedError

    def upload(self, pairs: list[tuple[str, str]]) -> None:
        """[(local_path, remote_path)] in one batch."""
        raise NotImplementedError

    def put_bytes(self, remote_path: str, data: bytes) -> None:
        raise NotImplementedError

    def get_bytes(self, remote_path: str) -> bytes | None:
        raise NotImplementedError

    def delete(self, paths: list[str]) -> None:
        raise NotImplementedError

    def check_write(self) -> str:
        """Verifies this process can write to the bucket; returns who it is. Raises RemoteError."""
        raise NotImplementedError

    def download(self, pairs: list[tuple[str, str]]) -> None:
        """[(remote_path, local_path)]. Each file lands atomically (`.part` then rename), so a
        local file that exists is always complete."""
        if not pairs:
            return
        staged = []
        for remote_path, local_path in pairs:
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            staged.append((remote_path, local_path + ".part"))
        self._fetch(staged)
        for _, part in staged:
            os.replace(part, part[:-len(".part")])


class HFRemote(Remote):
    def __init__(self, bucket_id: str, token: str | None = None):
        from huggingface_hub import HfApi
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()  # per-file bars from every background push would drown the job log
        self.bucket_id = bucket_id
        self.url = HF_PREFIX + bucket_id
        # token=None: HfApi falls back to HF_TOKEN / the saved login; a public bucket reads fine without one.
        self._api = HfApi(token=token)

    def list(self, prefix: str = "") -> dict[str, int]:
        from huggingface_hub import BucketFile  # top-level export: stable across 1.x and 2.x
        prefix = prefix.strip("/")

        def go():
            out = {}
            for item in self._api.list_bucket_tree(self.bucket_id, prefix=prefix or None, recursive=True):
                if isinstance(item, BucketFile) and (not prefix or item.path == prefix or item.path.startswith(prefix + "/")):
                    out[item.path] = item.size
            return out
        return _retry(go, what=f"list {prefix or '/'}")

    def _fetch(self, pairs):
        _retry(lambda: self._api.download_bucket_files(self.bucket_id, pairs, raise_on_missing_files=True),
               what=f"download {len(pairs)} file(s)")

    def upload(self, pairs):
        if pairs:
            _retry(lambda: self._api.batch_bucket_files(self.bucket_id, add=[(str(l), r) for l, r in pairs]),
                   what=f"upload {len(pairs)} file(s)")

    def put_bytes(self, remote_path, data):
        _retry(lambda: self._api.batch_bucket_files(self.bucket_id, add=[(data, remote_path)]), what=f"put {remote_path}")

    def get_bytes(self, remote_path):
        import tempfile
        from huggingface_hub.errors import EntryNotFoundError
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "f")
            try:
                _retry(lambda: self._api.download_bucket_files(self.bucket_id, [(remote_path, target)], raise_on_missing_files=True),
                       what=f"get {remote_path}")
            except EntryNotFoundError:
                return None
            if not os.path.exists(target):
                return None
            with open(target, "rb") as f:
                return f.read()

    def delete(self, paths):
        if paths:
            _retry(lambda: self._api.batch_bucket_files(self.bucket_id, delete=list(paths)), what=f"delete {len(paths)} file(s)")

    def check_write(self):
        try:
            who = self._api.whoami()
        except Exception as e:  # noqa: BLE001
            raise RemoteError(f"no usable HF token (whoami failed: {e!r}) -- set HF_TOKEN or run `hf auth login`") from None
        probe = f".write-probe/{uuid.uuid4().hex}"
        try:
            self.put_bytes(probe, b"")
            self.delete([probe])
        except Exception as e:  # noqa: BLE001
            raise RemoteError(f"token for {who.get('name')!r} cannot write to {self.url}: {e!r}") from None
        return who.get("name", "?")


class MemoryRemote(Remote):
    """In-memory bucket for tests and offline dry runs. `memory://<name>` URLs share one instance."""
    _registry: dict[str, "MemoryRemote"] = {}

    def __init__(self, name: str = "test", *, writable: bool = True):
        self.url = f"memory://{name}"
        self.files: dict[str, bytes] = {}
        self.writable = writable
        self.fail_next: list[Exception] = []  # test hook: raised by the next mutating calls, one each
        self.log: list[tuple] = []            # ("upload"|"delete"|"put", path) in call order
        self._lock = threading.Lock()

    @classmethod
    def named(cls, name: str) -> "MemoryRemote":
        return cls._registry.setdefault(name, cls(name))

    def _maybe_fail(self):
        if self.fail_next:
            raise self.fail_next.pop(0)

    def _need_write(self):
        if not self.writable:
            raise PermissionError("read-only remote")
        self._maybe_fail()

    def list(self, prefix=""):
        prefix = prefix.strip("/")
        with self._lock:
            return {p: len(b) for p, b in sorted(self.files.items())
                    if not prefix or p == prefix or p.startswith(prefix + "/")}

    def _fetch(self, pairs):
        for remote_path, local_path in pairs:
            with self._lock:
                if remote_path not in self.files:
                    raise FileNotFoundError(remote_path)
                data = self.files[remote_path]
            with open(local_path, "wb") as f:
                f.write(data)

    def upload(self, pairs):
        self._need_write()
        for local_path, remote_path in pairs:
            with open(local_path, "rb") as f:
                data = f.read()
            with self._lock:
                self.files[remote_path] = data
                self.log.append(("upload", remote_path))

    def put_bytes(self, remote_path, data):
        self._need_write()
        with self._lock:
            self.files[remote_path] = data
            self.log.append(("put", remote_path))

    def get_bytes(self, remote_path):
        with self._lock:
            return self.files.get(remote_path)

    def delete(self, paths):
        self._need_write()
        for p in paths:
            with self._lock:
                self.files.pop(p, None)
                self.log.append(("delete", p))

    def check_write(self):
        if not self.writable:
            raise RemoteError(f"{self.url} is read-only")
        return "memory-user"


def open_remote(url: str) -> Remote:
    """`hf://buckets/<owner>/<name>` -> HFRemote; `memory://<name>` -> a shared MemoryRemote."""
    if url.startswith(HF_PREFIX):
        bucket_id = url[len(HF_PREFIX):].strip("/")
        if bucket_id.count("/") != 1:
            raise RemoteError(f"bad bucket url {url!r}: expected hf://buckets/<owner>/<name>")
        return HFRemote(bucket_id)
    if url.startswith("memory://"):
        return MemoryRemote.named(url[len("memory://"):])
    raise RemoteError(f"unsupported remote {url!r} (expected hf://buckets/<owner>/<name>)")


# ---------------------------------------------------------------------------------------------
# Layout knowledge: entities, steps, markers
# ---------------------------------------------------------------------------------------------

def step_files(listing: dict[str, int], entity_dir: str) -> dict[int, dict]:
    """{step: {"model": path|None, "meta": path|None, "optim": {rank: path}}} for one checkpoint
    folder's direct files."""
    steps: dict[int, dict] = {}
    for path in listing:
        if os.path.dirname(path) != entity_dir:
            continue
        name = os.path.basename(path)
        if m := _MODEL_RE.match(name):
            steps.setdefault(int(m[1]), {"model": None, "meta": None, "optim": {}})["model"] = path
        elif m := _META_RE.match(name):
            steps.setdefault(int(m[1]), {"model": None, "meta": None, "optim": {}})["meta"] = path
        elif m := _OPTIM_RE.match(name):
            steps.setdefault(int(m[1]), {"model": None, "meta": None, "optim": {}})["optim"][int(m[2])] = path
    return steps


def complete_steps(listing: dict[str, int], entity_dir: str) -> list[int]:
    """Steps with a marker (`meta_<step>.json`) *and* a model -- the only ones a pull will see."""
    return sorted(s for s, f in step_files(listing, entity_dir).items() if f["meta"] and f["model"])


def entities(listing: dict[str, int]) -> dict[str, dict]:
    """{entity_id: {"kind", "files", "bytes"}} for every leaf entity in a listing. A README.md is
    not an entity on its own; it's an attribute of the folder it sits in."""
    found: dict[str, dict] = {}

    def add(id_, kind, path, size):
        e = found.setdefault(id_, {"kind": kind, "files": 0, "bytes": 0})
        e["files"] += 1
        e["bytes"] += size

    for path, size in listing.items():
        parts = path.split("/")
        top = parts[0]
        if top == TOKENIZERS and len(parts) >= 3:
            add("/".join(parts[:2]), "tokenizer", path, size)
        elif top == PREPARED and len(parts) >= 3:
            add("/".join(parts[:2]), "dataset", path, size)
        elif top == CHECKPOINTS and len(parts) >= 3:
            name = parts[-1]
            if _MODEL_RE.match(name) or _META_RE.match(name) or _OPTIM_RE.match(name):
                add("/".join(parts[:-1]), "checkpoint", path, size)
        elif top == "experiments" and len(parts) >= 3:
            add("/".join(parts[:2]), "experiment", path, size)
        elif top == "eval_bundle" and len(parts) >= 2:
            add("eval_bundle", "bench-data", path, size)
        elif top == TASK_DATA and len(parts) >= 3:
            add("/".join(parts[:-1]), "bench-data", path, size)
    return found


def wants(policy, step: int) -> bool:
    """Does `policy` ("last" | "all" | "none" | [steps]) push this step? ("last" pushes every save,
    then prunes the older ones -- the bucket always holds the newest resumable state.)"""
    if policy in ("last", "all"):
        return True
    if policy == "none" or policy is None:
        return False
    return step in policy


def validate_policy(policy, *, key: str):
    if policy in ("last", "all", "none"):
        return
    if isinstance(policy, list) and all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in policy):
        return
    raise ValueError(f'{key} must be "last", "all", "none", or a list of steps, got {policy!r}')


# ---------------------------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def push_group(remote: Remote, listing: dict[str, int], files: list[tuple[str, str]], marker: tuple[str, str] | None,
               *, force: bool = False) -> list[str]:
    """Uploads `files` in one batch, then `marker` (always last). `listing` is the remote state of
    the entity's folder, used for the immutability rule: if the marker already exists remotely the
    group is *complete*, and each file must then match in size (skipped) or the push refuses; if
    the marker is absent the group is a partial upload, freely overwritten. Updates `listing` in
    place. Returns the remote paths actually uploaded."""
    everything = list(files) + ([marker] if marker else [])
    complete = marker is not None and marker[1] in listing
    to_upload = []
    for local, remote_path in everything:
        size = os.path.getsize(local)
        if complete and remote_path in listing:
            if listing[remote_path] != size and not force:
                raise ImmutableError(f"{remote_path} already exists in {remote.url} as part of a complete upload "
                                     f"(remote {listing[remote_path]} B, local {size} B) -- refusing to overwrite "
                                     f"(only `remote push --force` may)")
            if listing[remote_path] == size and not force:
                continue
        to_upload.append((local, remote_path, size))
    plain = [(l, r) for l, r, _ in to_upload if not marker or r != marker[1]]
    remote.upload(plain)
    for _, r, s in to_upload:
        if not marker or r != marker[1]:
            listing[r] = s
    if marker and any(r == marker[1] for _, r, _ in to_upload):
        remote.upload([(marker[0], marker[1])])
        listing[marker[1]] = os.path.getsize(marker[0])
    return [r for _, r, _ in to_upload]


def gate_producer(remote: Remote, entity_id: str, producer: dict, *, force: bool = False) -> str | None:
    """The entity's existing README text, or None. Raises ProducerConflict if it was produced by a
    different step (rule 2) -- checked *before* any file is pushed."""
    text = remote.get_bytes(f"{entity_id}/README.md")
    if text is None:
        return None
    text = text.decode("utf-8")
    existing = readme.split(text)[0].get("producer") or {}
    if existing and not readme.same_producer(existing, producer) and not force:
        raise ProducerConflict(f"{entity_id} already exists in the bucket, produced by {existing} -- this push is "
                               f"from {producer}. Pick another id, or `remote push --force` if you mean to take it over.")
    return text


def write_readme(remote: Remote, entity_id: str, existing: str | None, *, kind: str, producer: dict,
                 inputs: dict | None = None, motivation: str = "", retention: dict | None = None,
                 steps: list | None = None, metrics: dict | None = None, extra: dict | None = None,
                 history: str) -> None:
    if existing is None:
        text = readme.create(kind, entity_id, producer=producer, inputs=inputs, motivation=motivation,
                             retention=retention, steps=steps, metrics=metrics, extra=extra, history=history)
    else:
        updates = {}
        for key, value in (("steps", steps), ("metrics", metrics), ("retention", retention)):
            if value is not None:
                updates[key] = value
        updates.update(extra or {})
        text = readme.update(existing, meta_updates=updates, history=history)
    remote.put_bytes(f"{entity_id}/README.md", text.encode("utf-8"))


def _checkpoint_names(step: int, ranks) -> tuple[str, str, list[str]]:
    return f"model_{step:06d}.pt", f"meta_{step:06d}.json", [f"optim_{step:06d}_rank{r}.pt" for r in ranks]


def push_checkpoint_step(remote: Remote, local_dir: str, tag: str, step: int, *, push_model, push_optim, ranks,
                         producer: dict, inputs: dict | None = None, motivation: str = "", metrics: dict | None = None,
                         force: bool = False) -> None:
    """One checkpoint step: [model, optim_rank*] first, meta last; then retention (`"last"` deletes
    the older steps' remote copies -- meta before model, so an older step never looks complete
    without its weights); then the entity README. Runs on the Uploader thread."""
    entity_id = f"{CHECKPOINTS}/{tag}"
    model_n, meta_n, optim_ns = _checkpoint_names(step, ranks)
    existing = gate_producer(remote, entity_id, producer, force=force)
    listing = remote.list(entity_id)
    files = []
    if wants(push_optim, step):
        files += [(os.path.join(local_dir, n), f"{entity_id}/{n}") for n in optim_ns if os.path.exists(os.path.join(local_dir, n))]
    marker = None
    if wants(push_model, step):
        files.append((os.path.join(local_dir, model_n), f"{entity_id}/{model_n}"))
        marker = (os.path.join(local_dir, meta_n), f"{entity_id}/{meta_n}")
    if not files and not marker:
        return
    # marker is None for an optimizer-only push: no completeness marker to gate on, so the group is
    # never "complete".
    pushed = push_group(remote, listing, files, marker, force=force)

    doomed = []
    for s, f in sorted(step_files(listing, entity_id).items()):
        if s >= step:
            continue
        if push_model == "last":
            doomed += [p for p in (f["meta"], f["model"]) if p]
        if push_optim == "last":
            doomed += list(f["optim"].values())
    if doomed:
        remote.delete(doomed)
        for p in doomed:
            listing.pop(p, None)

    kept = complete_steps(listing, entity_id)
    write_readme(remote, entity_id, existing, kind="checkpoint", producer=producer, inputs=inputs, motivation=motivation,
                 retention={"model": push_model, "optim": push_optim}, steps=kept, metrics=metrics,
                 history=f"pushed step {step}" + (f"; pruned {sorted({int(_step_of(p)) for p in doomed})}" if doomed else ""))


def _step_of(path: str) -> int:
    return int(re.search(r"_(\d+)", os.path.basename(path))[1])


def push_dir(remote: Remote, local_dir: str, remote_dir: str, *, marker_name: str | None, force: bool = False,
             listing: dict | None = None) -> list[str]:
    """Every syncable file under local_dir (README.md excluded -- it's remote-managed) -> remote_dir,
    `marker_name` last. Used by tokenizer/dataset/bench-data pushes."""
    files = []
    for root, _, names in os.walk(local_dir):
        for n in sorted(names):
            path = os.path.join(root, n)
            rel = os.path.relpath(path, local_dir).replace(os.sep, "/")
            if rel == "README.md" or not is_syncable(rel):
                continue
            files.append((path, f"{remote_dir}/{rel}"))
    marker = next(((l, r) for l, r in files if os.path.relpath(l, local_dir) == marker_name), None) if marker_name else None
    plain = [(l, r) for l, r in files if (l, r) != marker]
    return push_group(remote, remote.list(remote_dir) if listing is None else listing, plain, marker, force=force)


# ---------------------------------------------------------------------------------------------
# Background uploader
# ---------------------------------------------------------------------------------------------

class Uploader:
    """FIFO queue of push closures `fn(remote)` on one non-daemon background thread. The thread
    exits by itself after a few idle seconds (and is restarted lazily on the next submit), so a
    finished process is never held hostage by an idle worker, yet a process that ends with pushes
    still queued waits for them."""
    IDLE_SECONDS = 3.0

    def __init__(self, remote: Remote):
        self.remote = remote
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._errors: list[str] = []
        self._failed: set[str] = set()

    def submit(self, entity: str, fn, *, marker: bool = False, label: str = "") -> None:
        with self._lock:
            self._q.put((entity, fn, marker, label or entity))
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="tinylab-uploader", daemon=False)
                self._thread.start()

    def _run(self):
        while True:
            try:
                item = self._q.get(timeout=self.IDLE_SECONDS)
            except queue.Empty:
                with self._lock:
                    if self._q.empty():
                        self._thread = None
                        return
                continue
            entity, fn, marker, label = item
            try:
                if marker and entity in self._failed:
                    raise RemoteError("skipped: an earlier upload of this entity failed, so its completeness marker is withheld")
                fn(self.remote)
            except Exception as e:  # noqa: BLE001 -- collected, raised by flush()
                self._failed.add(entity)
                self._errors.append(f"{label}: {e!r}")
                print0(f"remote: upload failed -- {label}: {e!r} (training continues; `remote push` can redo it)")
            finally:
                self._q.task_done()

    def pending(self) -> int:
        return self._q.unfinished_tasks

    def flush(self) -> None:
        self._q.join()
        with self._lock:
            errors, self._errors = self._errors, []
            self._failed.clear()
        if errors:
            raise UploadError("; ".join(errors))


# ---------------------------------------------------------------------------------------------
# Experiment folder sync (logs, results, snapshots) -- mutable files, re-uploaded when they change
# ---------------------------------------------------------------------------------------------

class ExperimentSync:
    """Mirrors `<base>/experiments/<exp>/` to the bucket. Files there are logs/results (appended
    to) or write-once snapshots, so unlike artifacts they may be overwritten. Runs on a timer thread
    every `interval` seconds and on demand (`sync()`); failures are remembered and raised by `close()`."""

    def __init__(self, remote: Remote, experiment: str, *, interval: float = 180.0):
        self.remote, self.experiment, self.interval = remote, experiment, interval
        self.local = os.path.join(get_base_dir(), "experiments", experiment)
        self._seen: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: str | None = None
        self._thread = threading.Thread(target=self._loop, name="tinylab-expsync", daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.sync()
            except Exception as e:  # noqa: BLE001
                self._error = repr(e)

    def sync(self) -> int:
        """Uploads every changed syncable file. Returns how many."""
        with self._lock:
            pairs, stamps = [], {}
            if os.path.isdir(self.local):
                for root, _, names in os.walk(self.local):
                    for n in names:
                        path = os.path.join(root, n)
                        rel = os.path.relpath(path, self.local).replace(os.sep, "/")
                        if not is_syncable(rel):
                            continue
                        st = os.stat(path)
                        stamp = (st.st_size, st.st_mtime)
                        if self._seen.get(path) != stamp:
                            pairs.append((path, f"experiments/{self.experiment}/{rel}"))
                            stamps[path] = stamp
            if pairs:
                self.remote.upload(pairs)
                self._seen.update(stamps)
            return len(pairs)

    def close(self):
        self._stop.set()
        self.sync()
        if self._error:
            error, self._error = self._error, None
            raise UploadError(f"experiment sync: {error}")


# ---------------------------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------------------------

def pull_paths(remote: Remote, paths: list[str], base_dir: str | None = None, *, skip_existing: bool = True) -> list[str]:
    """Downloads each bucket path into the same relative path under base_dir; returns those fetched."""
    base_dir = base_dir or get_base_dir()
    pairs = [(p, os.path.join(base_dir, *p.split("/"))) for p in paths]
    if skip_existing:
        pairs = [(r, l) for r, l in pairs if not os.path.exists(l)]
    remote.download(pairs)
    return [r for r, _ in pairs]


def pull_prefix(remote: Remote, prefix: str, base_dir: str | None = None) -> list[str]:
    """Everything under prefix that isn't local yet (READMEs are remote-managed, not pulled)."""
    listing = remote.list(prefix)
    paths = [p for p in listing if os.path.basename(p) != "README.md" and is_syncable(p)]
    return pull_paths(remote, paths, base_dir)


def pull_checkpoint(remote: Remote, tag: str, base_dir: str | None = None, *, step: int | None = None,
                    optim: bool = False, ranks=None) -> int:
    """model + meta of `step` (default: the newest step with a marker); `optim_*` only if asked.
    Returns the step. Raises FileNotFoundError if the bucket has no complete matching step."""
    entity_id = f"{CHECKPOINTS}/{tag}"
    listing = remote.list(entity_id)
    steps = complete_steps(listing, entity_id)
    if step is None:
        if not steps:
            raise FileNotFoundError(f"{remote.url}: no complete checkpoint under {entity_id} (need model + meta)")
        step = steps[-1]
    elif step not in steps:
        raise FileNotFoundError(f"{remote.url}: {entity_id} has no complete step {step} (complete: {steps})")
    f = step_files(listing, entity_id)[step]
    paths = [f["model"], f["meta"]]
    if optim:
        chosen = f["optim"] if ranks is None else {r: p for r, p in f["optim"].items() if r in ranks}
        paths += list(chosen.values())
    pull_paths(remote, paths, base_dir)
    return step


def pull_tokenizer(remote: Remote, name: str, base_dir: str | None = None) -> list[str]:
    """The whole tokenizers/<name>/ folder. If a local copy already exists its tokenizer.pkl must
    be byte-identical (rule 6) -- a mismatch is a hard error, never an overwrite."""
    base_dir = base_dir or get_base_dir()
    local_pkl = os.path.join(base_dir, TOKENIZERS, name, "tokenizer.pkl")
    listing = remote.list(f"{TOKENIZERS}/{name}")
    if f"{TOKENIZERS}/{name}/tokenizer.pkl" not in listing:
        raise FileNotFoundError(f"{remote.url}: no tokenizer {name!r} ({TOKENIZERS}/{name}/tokenizer.pkl missing)")
    if os.path.exists(local_pkl):
        remote_bytes = remote.get_bytes(f"{TOKENIZERS}/{name}/tokenizer.pkl")
        if hashlib.sha256(remote_bytes).hexdigest() != sha256_file(local_pkl):
            raise RemoteError(f"tokenizer {name!r}: the local tokenizer.pkl differs from the bucket's -- refusing to "
                              f"overwrite it (token ids would change meaning). Rename one of them.")
    return pull_prefix(remote, f"{TOKENIZERS}/{name}", base_dir)


# ---------------------------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------------------------

class Prefetcher:
    """Downloads files of one remote folder into a local folder in a fixed order. `ensure(name)`
    blocks until `name` is local and starts the next `lookahead` names in the background, wrapping
    at the end (the dataloader cycles)."""

    def __init__(self, remote: Remote, remote_dir: str, local_dir: str, names: list[str], *, lookahead: int = 2, workers: int = 2):
        self.remote, self.remote_dir, self.local_dir = remote, remote_dir.rstrip("/"), local_dir
        self.names = list(names)
        self._pos = {n: i for i, n in enumerate(self.names)}
        self.lookahead = lookahead
        self._futures: dict[str, object] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tinylab-prefetch")

    def _local(self, name):
        return os.path.join(self.local_dir, name)

    def _fetch(self, name):
        if not os.path.exists(self._local(name)):
            self.remote.download([(f"{self.remote_dir}/{name}", self._local(name))])

    def _schedule(self, name):
        with self._lock:
            if name not in self._futures and not os.path.exists(self._local(name)):
                self._futures[name] = self._pool.submit(self._fetch, name)
            return self._futures.get(name)

    def ensure(self, name: str) -> None:
        future = self._schedule(name)
        i = self._pos.get(name)
        if i is not None:
            for k in range(1, self.lookahead + 1):
                self._schedule(self.names[(i + k) % len(self.names)])
        if future is not None:
            try:
                future.result()
            except BaseException:
                with self._lock:
                    self._futures.pop(name, None)  # let a later ensure() retry
                raise

    def fetch_all(self, names: list[str]) -> None:
        for future in [self._schedule(n) for n in names]:
            if future is not None:
                future.result()

    def close(self):
        self._pool.shutdown(wait=True)


# ---------------------------------------------------------------------------------------------
# Run-level helpers
# ---------------------------------------------------------------------------------------------

def make_producer(experiment: str, job: str, step: str, hardware: str | None = None) -> dict:
    """The README frontmatter `producer` record."""
    import subprocess
    try:
        from importlib.metadata import version
        ver = "v" + version("tinylab")
    except Exception:  # noqa: BLE001
        ver = "unknown"
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        sha = None
    producer = {"repo": "tinylab", "version": ver, "git": sha, "experiment": experiment, "job": job, "step": step}
    if hardware:
        producer["hardware"] = hardware
    return producer


def describe_hardware(world_size: int = 1) -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).replace("NVIDIA ", "").replace(" ", "-")
            return f"{world_size}x{name}"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"
