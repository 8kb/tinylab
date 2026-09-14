"""Tests for tinylab.ops.train's pure-math pieces: derive_training_plan (the scaling-law horizon/
LR/weight-decay derivation -- the most consequential function in the repo, previously untested)
and _lr_schedule (the LR-multiplier and Muon-momentum schedules, including a regression test for
the now-fixed short-run momentum warmdown)."""
import pytest

from tinylab.ops.train import B_REF, derive_training_plan, _lr_schedule


def _plan(**overrides):
    kwargs = dict(
        num_scaling_params=1_000_000, d_ref_scaling_params=1_000_000, num_flops_per_token=6_000_000,
        target_param_data_ratio=12, target_flops=-1.0, num_iterations=-1, total_batch_size=B_REF,
        weight_decay=0.1,
    )
    kwargs.update(overrides)
    return derive_training_plan(**kwargs)


def test_user_num_iterations_takes_precedence():
    plan = _plan(num_iterations=50)
    assert plan.horizon_source == "user"
    assert plan.num_iterations == 50


def test_target_flops_derives_iterations_when_no_explicit_count():
    plan = _plan(num_iterations=-1, target_flops=6_000_000 * B_REF * 10)
    assert plan.horizon_source == "target_flops"
    assert plan.num_iterations == 10


def test_target_param_data_ratio_is_the_fallback_horizon():
    plan = _plan(num_iterations=-1, target_flops=-1.0, target_param_data_ratio=12)
    assert plan.horizon_source == "target_param_data_ratio"
    assert plan.num_iterations == (12 * 1_000_000) // B_REF


def test_no_horizon_specified_raises():
    try:
        _plan(num_iterations=-1, target_flops=-1.0, target_param_data_ratio=-1)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "No training horizon" in str(e)


def test_batch_lr_scale_is_one_at_reference_batch_size():
    plan = _plan(total_batch_size=B_REF, num_iterations=10)
    assert plan.batch_lr_scale == 1.0


def test_batch_lr_scale_scales_with_sqrt_of_batch_ratio():
    plan = _plan(total_batch_size=B_REF // 4, num_iterations=10)
    assert plan.batch_lr_scale == (0.25) ** 0.5


def test_auto_batch_size_picks_a_power_of_two():
    plan = _plan(total_batch_size=-1, num_iterations=10)
    assert plan.auto_batch_size is True
    assert plan.total_batch_size > 0
    assert plan.total_batch_size & (plan.total_batch_size - 1) == 0  # power of two


def test_total_tokens_and_flops_are_consistent_with_the_chosen_horizon():
    plan = _plan(num_iterations=7, total_batch_size=1024, num_flops_per_token=100)
    assert plan.total_tokens == 1024 * 7
    assert plan.total_flops == 100 * 1024 * 7


# -----------------------------------------------------------------------------

def test_lr_multiplier_ramps_up_during_warmup_and_holds_at_one():
    get_lr, _get_momentum = _lr_schedule(num_iterations=100, warmup_steps=10, warmdown_ratio=0.5, final_lr_frac=0.1)
    assert get_lr(0) < get_lr(5) < get_lr(9)
    assert get_lr(9) <= 1.0
    assert get_lr(40) == 1.0  # past warmup, before warmdown starts


def test_lr_multiplier_decays_to_final_frac_at_the_last_step():
    get_lr, _get_momentum = _lr_schedule(num_iterations=100, warmup_steps=10, warmdown_ratio=0.5, final_lr_frac=0.1)
    assert get_lr(100) == 0.1


def test_muon_momentum_warmdown_is_reachable_on_a_short_run():
    """Regression test: get_muon_momentum used to check `it < 400` first, so any run under 400
    iterations (every job shipped in this repo) never reached its own warmdown branch at all."""
    num_iterations = 30
    _get_lr, get_momentum = _lr_schedule(num_iterations=num_iterations, warmup_steps=5, warmdown_ratio=0.65, final_lr_frac=0.05)
    warmdown_start = num_iterations - round(0.65 * num_iterations)
    # momentum must actually move away from 0.97 (its warmup ceiling / warmdown start value) by
    # the last step -- if the warmdown branch were unreachable, this would stay at 0.97.
    assert get_momentum(num_iterations) < 0.97
    assert get_momentum(warmdown_start) == 0.97


def test_muon_momentum_warms_up_from_085_to_097():
    # momentum_warmup_iters = max(1, min(400, num_iterations // 3)) -- 1000 // 3 = 333 here, not
    # the upstream 400 ceiling (that only applies once num_iterations >= 1200).
    num_iterations = 1000
    momentum_warmup_iters = max(1, min(400, num_iterations // 3))
    get_lr, get_momentum = _lr_schedule(num_iterations=num_iterations, warmup_steps=10, warmdown_ratio=0.1, final_lr_frac=0.1)
    assert get_momentum(0) == pytest.approx(0.85)
    assert get_momentum(momentum_warmup_iters) == pytest.approx(0.97)
