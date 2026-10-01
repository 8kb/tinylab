"""Every train step drops the previous step's compiled code before compiling its own: dynamo's
recompile limit is per code object, not per model, so without the reset the Nth step of a job
silently runs eager (measured: 4x slower)."""
from conftest import TINY_GPT_CONFIG, TINY_SEQUENCE_LEN

from tinylab.ops import Context
from tinylab.ops.train import run as train_run


def test_each_train_step_resets_dynamo_before_it_compiles(tiny_dataset, monkeypatch):
    import torch
    events = []
    monkeypatch.setattr(torch._dynamo, "reset", lambda: events.append("reset"))
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: events.append("compile") or model)
    cfg = {"name": "pre", "op": "train", "kind": "base", "dataset": tiny_dataset, "sequence_len": TINY_SEQUENCE_LEN,
           "model_config": TINY_GPT_CONFIG, "device_batch_size": 2, "total_batch_size": 64, "world_size": 1,
           "num_iterations": 1, "eval_every": 0, "eval_tokens": 64}
    ctx = Context(device_type="cpu")
    train_run(dict(cfg, name="a", output_tag="a"), ctx)
    train_run(dict(cfg, name="b", output_tag="b"), ctx)
    assert events == ["reset", "compile", "reset", "compile"]
