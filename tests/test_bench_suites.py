"""The bench op's non-score suites -- bpb, sample, tokenizer, infer -- on the tiny fixture model
and the synthetic "smoke" dataset (see conftest.py). `infer` is CUDA-only: here it must refuse
cleanly; its numbers are only checkable on a GPU pod."""
import math

import pytest

from tinylab.ops import Context, bench


def _ctx():
    return Context(device_type="cpu")


def test_accepted_keys_are_suite_aware():
    assert "split_tokens" in bench.accepted_keys({"suite": "bpb"})
    assert "split_tokens" not in bench.accepted_keys({"suite": "sample"})
    assert "max_per_task" not in bench.accepted_keys({"suite": "bpb"})
    assert "model_tag" in bench.accepted_keys({"suite": "infer"})
    # The tokenizer suite measures the step's own tokenizer: no checkpoint to name.
    assert "model_tag" not in bench.accepted_keys({"suite": "tokenizer"})
    assert {"suite", "baselines"} == bench.accepted_keys({"suite": "tokenizer"})


def test_unknown_suite_is_an_error():
    with pytest.raises(AssertionError, match="suite must be one of"):
        bench.run({"name": "b", "suite": "nope"}, _ctx())


@pytest.mark.slow
def test_bpb_reports_train_and_val(tiny_checkpoint, tiny_dataset):
    out = bench.run({"name": "b", "suite": "bpb", "model_tag": tiny_checkpoint, "dataset": tiny_dataset,
                     "split_tokens": 128, "device_batch_size": 2}, _ctx())
    assert set(out["results"]) == {"train", "val"}
    assert all(math.isfinite(v) and v > 0 for v in out["results"].values())


@pytest.mark.slow
def test_sample_returns_a_greedy_and_a_sampled_text_per_prompt(tiny_checkpoint):
    out = bench.run({"name": "b", "suite": "sample", "model_tag": tiny_checkpoint,
                     "prompts": ["The quick", "Hello"], "max_new_tokens": 4}, _ctx())
    assert [s["prompt"] for s in out["samples"]] == ["The quick", "Hello"]
    assert all(isinstance(s["greedy"], str) and isinstance(s["sampled"], str) for s in out["samples"])


@pytest.mark.slow
def test_sample_greedy_is_deterministic(tiny_checkpoint):
    cfg = {"name": "b", "suite": "sample", "model_tag": tiny_checkpoint, "prompts": ["The quick"], "max_new_tokens": 6}
    a, b = bench.run(dict(cfg), _ctx()), bench.run(dict(cfg), _ctx())
    assert a["samples"][0]["greedy"] == b["samples"][0]["greedy"]


def test_tokenizer_suite_measures_compression_without_a_model(base_dir):
    out = bench.run({"name": "t", "suite": "tokenizer"}, _ctx())
    assert out["vocab_sizes"]["ours"] > 256
    assert set(out["results"]) == {"ours"}
    assert {"news", "korean", "code", "math", "science"} <= set(out["results"]["ours"])
    english = out["results"]["ours"]["news"]
    assert english["bytes"] / english["tokens"] == pytest.approx(english["ratio"])
    assert english["ratio"] > 1.0


def test_infer_refuses_off_cuda(base_dir):
    with pytest.raises(AssertionError, match="needs a CUDA GPU"):
        bench.run({"name": "i", "suite": "infer", "model_tag": "whatever"}, _ctx())
