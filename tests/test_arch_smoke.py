"""Train -> load -> generate smoke over the v3 architectures (gated attention, canon, and the
short_conv / mamba2 / mamba3 hybrids). Recurrent state is owned by modelcore's Decoder, so Engine
needs nothing architecture-specific: generation working here is that claim's test.

Marked slow. Fixtures were built once from modelcore/tests/conftest.py's builders and dumped with
ModelConfig.to_dict().
"""
import os

import pytest
import torch

from tinylab.checkpoints import load_model
from tinylab.engine import Engine
from tinylab.ops import Context
from tinylab.ops.train import run as train_run
from tinylab.tokenizer import get_tokenizer

from test_smoke import _prepare_fake_dataset

pytestmark = pytest.mark.slow

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
_ARCHS = ["gated_tiny", "canon_tiny", "hybrid_conv_tiny", "hybrid_mamba2_tiny", "hybrid_mamba3_tiny"]


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    # Several differently-shaped models in one process hit dynamo's recompile limit inside
    # MuonAdamW.step -- a test-isolation artifact, not a bug.
    torch._dynamo.reset()
    yield


@pytest.mark.parametrize("arch", _ARCHS)
def test_train_load_generate(arch, base_dir):
    tokenizer = get_tokenizer(base_dir)
    _prepare_fake_dataset(base_dir, tokenizer, 32)
    cfg = {
        "name": "pre", "op": "train", "kind": "base", "dataset": "smoke", "sequence_len": 32,
        "model_config": os.path.join(_FIXTURES, f"{arch}.json"),
        "device_batch_size": 2, "total_batch_size": 64, "num_iterations": 3, "world_size": 1,
        "eval_every": 3, "eval_tokens": 64,
    }
    result = train_run(cfg, Context(device_type="cpu"))
    assert result["val_bpb"] is not None and result["val_bpb"] > 0

    model, loaded_tokenizer, meta = load_model("pre", torch.device("cpu"), phase="eval")
    assert meta["model_config"]["format"] == "modelcore.v3"
    model.eval()
    engine = Engine(model, loaded_tokenizer)
    prompt = loaded_tokenizer.encode("The quick brown fox", prepend="<|bos|>")
    results, _ = engine.generate_batch(prompt, num_samples=2, max_tokens=8, temperature=0.0)
    assert len(results) == 2 and all(len(r) > len(prompt) for r in results)
    # Greedy decoding with recurrent state (KV cache / SSM / conv) must be deterministic across rows.
    assert results[0] == results[1]
