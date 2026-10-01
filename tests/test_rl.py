"""The `rl` op on the tiny fixture model with a 4-problem stand-in for GSM8K (the real one needs a
network download): two real REINFORCE steps, the checkpoint they write, and --resume continuing it.

Marked slow: runs real (tiny) generation and training loops. Deselect with `-m "not slow"`."""
import json
import os

import pytest

from tinylab.ops import Context, rl

from conftest import TINY_SEQUENCE_LEN

pytestmark = pytest.mark.slow


class _FakeGSM8K:
    """Four 'what is n+n' problems; a completion is right if it contains the answer."""

    def __init__(self):
        self.problems = [(f"What is {n}+{n}?", str(2 * n)) for n in (1, 2, 3, 4)]

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        question, answer = self.problems[idx]
        return {"messages": [{"role": "user", "content": question},
                             {"role": "assistant", "content": [{"type": "text", "text": answer}]}]}

    def evaluate(self, conversation, response):
        return int(conversation["messages"][-1]["content"][-1]["text"] in response)

    def reward(self, conversation, response):
        return float(self.evaluate(conversation, response))


@pytest.fixture(autouse=True)
def fake_gsm8k(monkeypatch):
    monkeypatch.setattr(rl, "_gsm8k_tasks", lambda ctx: (_FakeGSM8K(), _FakeGSM8K()))


def _cfg(source_tag, **overrides):
    cfg = {"name": "rlstep", "op": "rl", "source_tag": source_tag, "output_tag": "rlout", "sequence_len": TINY_SEQUENCE_LEN,
           "world_size": 1, "examples_per_step": 2, "num_samples": 2, "device_batch_size": 2, "max_new_tokens": 4,
           "eval_every": 1, "eval_examples": 2, "save_every": 1}
    cfg.update(overrides)
    return cfg


def _meta(base_dir, step):
    with open(os.path.join(base_dir, "checkpoints", "rlout", f"meta_{step:06d}.json")) as f:
        return json.load(f)


def test_two_steps_write_an_rl_checkpoint(base_dir, tiny_checkpoint):
    out = rl.run(_cfg(tiny_checkpoint), Context(device_type="cpu"))
    assert out["step"] == 2  # 4 problems / 2 per step, one epoch
    assert len(out["pass_at_k"]) == 2 and all(0.0 <= v <= 1.0 for v in out["pass_at_k"])
    meta = _meta(base_dir, 2)
    assert meta["kind"] == "rl" and meta["step"] == 2
    assert meta["base_model_tag"] == tiny_checkpoint
    assert meta["model_config"]["template"] == "nanochat"
    assert os.path.exists(os.path.join(base_dir, "checkpoints", "rlout", "optim_000002_rank0.pt"))


def test_resume_continues_from_the_checkpoint(base_dir, tiny_checkpoint):
    rl.run(_cfg(tiny_checkpoint, num_epochs=1), Context(device_type="cpu"))
    # A longer horizon on the same output_tag: resume picks up at step 2 and runs 2 more.
    out = rl.run(_cfg(tiny_checkpoint, num_epochs=2), Context(device_type="cpu", resume=True))
    assert out["step"] == 4
    assert _meta(base_dir, 4)["step"] == 4


def test_output_tag_must_differ_from_source_tag(base_dir, tiny_checkpoint):
    with pytest.raises(AssertionError, match="output_tag == source_tag"):
        rl.run(_cfg(tiny_checkpoint, output_tag=tiny_checkpoint), Context(device_type="cpu"))


def test_num_samples_must_be_a_multiple_of_device_batch_size(base_dir, tiny_checkpoint):
    with pytest.raises(AssertionError, match="multiple of device_batch_size"):
        rl.run(_cfg(tiny_checkpoint, num_samples=3), Context(device_type="cpu"))


def test_num_iterations_caps_the_run(base_dir, tiny_checkpoint):
    out = rl.run(_cfg(tiny_checkpoint, num_iterations=1), Context(device_type="cpu"))
    assert out["step"] == 1
    assert _meta(base_dir, 1)["step"] == 1
