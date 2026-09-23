"""
Runtime plumbing: base dir, device/DDP init, minimal logging. Device/DDP/seed bring-up
(is_ddp_requested/is_ddp_initialized/get_dist_info/autodetect_device_type/compute_init/
compute_cleanup) now lives in modelcore.runtime -- this file's own copy was ported from nanochat's
nanochat/common.py, and nanochat's copy has since moved to the same place, since modelcore has no
host dependencies at all. Re-exported here so every existing `from tinylab.runtime import ...` call
site keeps working unchanged.
"""
import os
from contextlib import contextmanager
from datetime import datetime

from modelcore.runtime import DEFAULT_RUNTIME
from modelcore.runtime import (  # noqa: F401 -- re-exported below for existing call sites
    autodetect_device_type, compute_cleanup, get_dist_info, is_ddp_initialized, is_ddp_requested,
)
from modelcore.runtime import compute_init as _compute_init

# The dtype used for compute (matmuls, activations); master weights stay fp32. Detection lives in
# modelcore.runtime (modelcore has no tinylab dependencies at all) -- this is a convenience
# re-export of the process-wide default runtime's value. Override with MODELCORE_DTYPE.
COMPUTE_DTYPE = DEFAULT_RUNTIME.compute_dtype
COMPUTE_DTYPE_REASON = DEFAULT_RUNTIME.compute_dtype_reason


def get_base_dir():
    """Not a module-level constant -- computed on every call so TINYLAB_BASE_DIR can be set (e.g.
    by a test) without needing to import this module in a particular order. tinylab keeps its own
    cache directory, separate from nanochat's ~/.cache/nanochat/, even though the ported tokenizer
    produces identical token ids -- see AGENTS.md."""
    if os.environ.get("TINYLAB_BASE_DIR"):
        base_dir = os.environ["TINYLAB_BASE_DIR"]
    else:
        base_dir = os.path.join(os.path.expanduser("~"), ".cache", "tinylab")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


# The file print0 currently tees to (rank 0 only -- see log_to_file below), plus the
# f"[{step_name}] " prefix (if any) it strips before writing a line there. A plain module-level
# pair, not a stack object: nesting is exactly one level deep in practice (job.run_file's own
# general log, briefly swapped to a step's own file and back), and log_to_file's own restore-on-
# exit is what makes even that safe to nest.
_log_file = None
_log_strip_prefix = None


@contextmanager
def log_to_file(path, *, strip_prefix=None):
    """Points print0's file-tee at `path` (opened in append mode) for the duration of this block,
    restoring whatever was pointed at before on exit -- nestable (job.run_file's own general log
    stays open for the whole run, briefly swapped to a step's own file for that step's op.run()
    call). Rank-0 only: every other rank already prints nothing via print0, so it opens/writes
    nothing here either -- no risk of N ranks racing on the same file."""
    global _log_file, _log_strip_prefix
    if int(os.environ.get("RANK", 0)) != 0:
        yield
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a", encoding="utf-8")
    prev_file, prev_prefix = _log_file, _log_strip_prefix
    _log_file, _log_strip_prefix = f, strip_prefix
    try:
        yield
    finally:
        _log_file, _log_strip_prefix = prev_file, prev_prefix
        f.close()


def print0(s="", **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(s, **kwargs)
        if _log_file is not None:
            text = str(s)
            if _log_strip_prefix and text.startswith(_log_strip_prefix):
                text = text[len(_log_strip_prefix):]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_kwargs = {k: v for k, v in kwargs.items() if k != "file"}
            print(f"[{timestamp}] {text}", file=_log_file, **file_kwargs)
            _log_file.flush()


def compute_init(device_type="cuda"):
    """Basic device/seed/DDP initialization, shared by every op. device_type: cuda|cpu|mps, or
    "auto"/"" to autodetect. Mechanism lives in modelcore.runtime.compute_init (see this module's
    own docstring); this wrapper only routes the autodetection message through print0 (rank-0-only
    printing), matching this repo's original behavior."""
    return _compute_init(device_type, log=print0)
