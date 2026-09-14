"""
Runtime plumbing: base dir, device/DDP init, minimal logging. Ported from nanochat's
nanochat/common.py (llmllab/nanochat) and trimmed to what tinylab's job runner needs -- no wandb
stand-in, no GPU peak-FLOPs/bandwidth tables, no colored logging or banner. See that file for the
untrimmed version if a future op needs MFU reporting back.
"""
import os
import torch
import torch.distributed as dist
from modelcore.runtime import DEFAULT_RUNTIME

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


def is_ddp_requested() -> bool:
    """True if launched by torchrun (env present), even before init."""
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_ddp_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    if is_ddp_requested():
        return True, int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 0, 1


def autodetect_device_type():
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type


def compute_init(device_type="cuda"):
    """Basic device/seed/DDP initialization, shared by every op. device_type: cuda|cpu|mps, or
    "auto" to autodetect."""
    if device_type == "auto" or device_type == "":
        device_type = autodetect_device_type()
    assert device_type in ("cuda", "mps", "cpu"), f"Invalid device type: {device_type}"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "device_type='cuda' but CUDA is not available"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "device_type='mps' but MPS is not available"

    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")

    is_ddp_req, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_req and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    return is_ddp_req, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup():
    if is_ddp_initialized():
        dist.destroy_process_group()
