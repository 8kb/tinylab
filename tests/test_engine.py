"""Tests for tinylab.engine: use_calculator's allow/deny surface (pure functions, no model or
tokenizer needed), plus Engine.generate_batch_multi's wiring against a real (untrained, but
"awoken") tiny model and the bundled default tokenizer -- fast and fully offline, unlike
test_smoke.py's real training loop, since what's being proven here is the Engine adapter's own
plumbing (special-token resolution, ToolSpec, collect_batch_multi assembly), not numerics that
modelcore's own test suite already covers exhaustively at the kernel level."""
import os

import torch

from modelcore import ModelManager
from tinylab.engine import Engine, use_calculator
from tinylab.modelconfig import load_model_config
from tinylab.tokenizer import get_tokenizer

_TINY_GPT_CONFIG = os.path.join(os.path.dirname(__file__), "fixtures", "gpt_tiny.json")


def _awake_engine(base_dir, seed=0):
    """A freshly built model has every attention c_proj/MLP down-projection/smear lambda_
    zero-initialized (see modelcore/AGENTS.md), so its logits are a function of the token embedding
    alone until woken -- same helper as modelcore/tests/test_generate.py's _build_awake."""
    tokenizer = get_tokenizer(base_dir=base_dir)
    config = load_model_config(_TINY_GPT_CONFIG, sequence_len=32, vocab_size=tokenizer.get_vocab_size())
    manager = ModelManager()
    model = manager.create_model(config, device=torch.device("cpu"), seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() and p.abs().sum() == 0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    model.eval()
    return Engine(model, tokenizer, manager=manager)


def test_allows_pure_arithmetic():
    assert use_calculator("2 + 3 * 4") == 14


def test_disallows_power_operator():
    assert use_calculator("2 ** 10") is None


def test_allows_string_count():
    assert use_calculator("'mississippi'.count('s')") == 4


def test_disallows_dunder_access():
    assert use_calculator("().__class__.__bases__[0].__subclasses__()") is None


def test_disallows_import():
    assert use_calculator("__import__('os').listdir('.')") is None


def test_disallows_arbitrary_method_call_without_count():
    # allowed_chars would let this through, but the ".count(" gate should still block it
    assert use_calculator("'x'.upper()") is None


def test_strips_commas_from_numbers():
    assert use_calculator("1,000 + 1") == 1001


def test_unparseable_expression_returns_none_not_an_exception():
    assert use_calculator("1 +") is None


def test_generate_batch_multi_matches_generate_batch_alone_at_temperature_zero(base_dir):
    engine = _awake_engine(base_dir)
    prompts = [
        engine.tokenizer.encode("The quick brown fox"),
        engine.tokenizer.encode("A small tinylab model learns to predict the next token"),
        engine.tokenizer.encode("Hi"),
    ]
    batched_results, batched_masks = engine.generate_batch_multi(prompts, num_samples=2, max_tokens=6, temperature=0.0)
    for p, prompt in enumerate(prompts):
        alone_results, alone_masks = engine.generate_batch(prompt, num_samples=2, max_tokens=6, temperature=0.0)
        assert batched_results[p] == alone_results, f"prompt {p} (len {len(prompt)}) differs batched vs alone"
        assert batched_masks[p] == alone_masks


def test_generate_batch_multi_of_one_prompt_equals_generate_batch(base_dir):
    engine = _awake_engine(base_dir)
    prompt = engine.tokenizer.encode("The quick brown fox")
    multi_results, multi_masks = engine.generate_batch_multi([prompt], num_samples=3, max_tokens=5, temperature=0.0)
    alone_results, alone_masks = engine.generate_batch(prompt, num_samples=3, max_tokens=5, temperature=0.0)
    assert multi_results == [alone_results]
    assert multi_masks == [alone_masks]
