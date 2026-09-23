"""Tests for tinylab.ops.train's pure-math piece: _lr_schedule (the LR-multiplier and Muon-
momentum schedules). tinylab derives no training horizon/batch-size/LR-scale of its own any more
-- that scaling-law math (previously tested here via the now-removed derive_training_plan/B_REF
re-export) lives only in modelcore.scaling, for nanochat's own use; every number it used to
produce is now a required tinylab job-file key instead. See AGENTS.md."""
import pytest

from tinylab.ops.train import _lr_schedule, accepted_keys


def test_init_lr_frac_and_load_optimizer_are_sft_only():
    """Neither exists in nanochat's scripts/base_train.py either (chat_sft.py-only args) -- on a
    kind="base" step they must be a startup error, not a silently-ignored key."""
    assert {"init_lr_frac", "load_optimizer"} <= accepted_keys({"kind": "sft"})
    assert not {"init_lr_frac", "load_optimizer"} & accepted_keys({"kind": "base"})


def test_lr_multiplier_ramps_up_during_warmup_and_holds_at_one():
    get_lr, _get_momentum = _lr_schedule(num_iterations=100, warmup_steps=10, warmdown_ratio=0.5, final_lr_frac=0.1, momentum_warmup_steps=400)
    assert get_lr(0) < get_lr(5) < get_lr(9)
    assert get_lr(9) <= 1.0
    assert get_lr(40) == 1.0  # past warmup, before warmdown starts


def test_lr_multiplier_decays_to_final_frac_at_the_last_step():
    get_lr, _get_momentum = _lr_schedule(num_iterations=100, warmup_steps=10, warmdown_ratio=0.5, final_lr_frac=0.1, momentum_warmup_steps=400)
    assert get_lr(100) == 0.1


def test_muon_momentum_warmdown_is_reachable_on_a_short_run():
    """Regression test: modelcore.optim.schedules.muon_momentum checks `it < momentum_warmup_steps`
    first, so a run whose horizon is shorter than momentum_warmup_steps must pass an explicit,
    smaller value here -- the old `min(400, num_iterations // 3)` heuristic that used to guarantee
    this is gone; a job file now sets "muon_momentum_warmup_steps" itself."""
    num_iterations = 30
    momentum_warmup_steps = 10
    _get_lr, get_momentum = _lr_schedule(num_iterations=num_iterations, warmup_steps=5, warmdown_ratio=0.65, final_lr_frac=0.05, momentum_warmup_steps=momentum_warmup_steps)
    warmdown_start = num_iterations - round(0.65 * num_iterations)
    # momentum must actually move away from 0.97 (its warmup ceiling / warmdown start value) by
    # the last step -- if the warmdown branch were unreachable, this would stay at 0.97.
    assert get_momentum(num_iterations) < 0.97
    assert get_momentum(warmdown_start) == 0.97


def test_muon_momentum_warms_up_from_085_to_097():
    num_iterations = 1000
    momentum_warmup_steps = 333
    get_lr, get_momentum = _lr_schedule(num_iterations=num_iterations, warmup_steps=10, warmdown_ratio=0.1, final_lr_frac=0.1, momentum_warmup_steps=momentum_warmup_steps)
    assert get_momentum(0) == pytest.approx(0.85)
    assert get_momentum(momentum_warmup_steps) == pytest.approx(0.97)
