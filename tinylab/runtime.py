"""
Runtime plumbing: base dir, device/DDP init, minimal logging. Device/DDP/seed bring-up
(is_ddp_requested/is_ddp_initialized/get_dist_info/autodetect_device_type/compute_init/
compute_cleanup) now lives in modelcore.runtime -- this file's own copy was ported from nanochat's
nanochat/common.py, and nanochat's copy has since moved to the same place, since modelcore has no
host dependencies at all. Re-exported here so every existing `from tinylab.runtime import ...` call
site keeps working unchanged.
"""
import os
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


def print0(s="", **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(s, **kwargs)


def compute_init(device_type="cuda"):
    """Basic device/seed/DDP initialization, shared by every op. device_type: cuda|cpu|mps, or
    "auto"/"" to autodetect. Mechanism lives in modelcore.runtime.compute_init (see this module's
    own docstring); this wrapper only routes the autodetection message through print0 (rank-0-only
    printing), matching this repo's original behavior."""
    return _compute_init(device_type, log=print0)
