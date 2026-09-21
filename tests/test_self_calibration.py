from circe import FiredEpisode, SelfCalibratingMonitor, doubly_robust_salvage_loss
from circe.self_calibration import (
    dr_counterfactual,
    empirical_bernstein_radius,
    reprieve_schedule,
)


def _fire(**kw):
    base = dict(episode_id="e", round=1, m_hat=0.5, reprieved=False, propensity=0.5,
                outcome_cont=None, outcome_halt=0)
    base.update(kw)
    return FiredEpisode(**base)


def test_dr_counterfactual_unreprieved_uses_model():
    ep = _fire(reprieved=False, m_hat=0.7)
    assert dr_counterfactual(ep) == 0.7


def test_dr_counterfactual_reprieved_applies_inverse_propensity():
    ep = _fire(reprieved=True, propensity=0.4, m_hat=0.5, outcome_cont=1)
    assert dr_counterfactual(ep) == 0.5 + (1 - 0.5) / 0.4


def test_dr_salvage_loss_formula():
    ep1 = _fire(reprieved=True, propensity=0.5, m_hat=0.5, outcome_cont=1)
    ep2 = _fire(reprieved=False, m_hat=0.5)
    # (0.5 + 1 + 0.5 - 0 - 0) / 2 = 1.0
    assert doubly_robust_salvage_loss([ep1, ep2], n_total=2) == pytest.approx(1.0)


def test_reprieve_schedule_in_unit_interval_and_decreasing():
    prev = 2.0
    for t in range(1, 20):
        eps = reprieve_schedule(t, n_configs=500, mean_reprieve_cost=1.0)
        assert 0.0 < eps <= 1.0
        assert eps <= prev + 1e-12
        prev = eps


def test_empirical_bernstein_radius_shrinks_with_data():
    import math
    radius_small_n = empirical_bernstein_radius([1.0, 0.0, 1.0, 0.0], 0.05, bound=1.0)
    data = [0.5] * 1000
    radius_large_n = empirical_bernstein_radius(data, 0.05, bound=1.0)
    assert radius_large_n < radius_small_n


def test_monitor_detects_drift_on_many_false_kills():
    mon = SelfCalibratingMonitor(alpha=0.05, delta=0.05)
    mon.begin_round(100)
    for i in range(100):
        # full reprieve (eps=1) so every counterfactual is observed; all would
        # have succeeded -> salvage loss ~1.0 >> alpha
        mon.record_fire(_fire(episode_id=f"e{i}", reprieved=True, propensity=1.0,
                              m_hat=0.5, outcome_cont=1))
    est = mon.estimate()
    assert est["salvage_loss"] == pytest.approx(1.0)
    assert mon.drift_detected()


def test_monitor_no_drift_when_all_kills_correct():
    mon = SelfCalibratingMonitor(alpha=0.05, delta=0.05)
    mon.begin_round(100)
    for i in range(100):
        mon.record_fire(_fire(episode_id=f"e{i}", reprieved=True, propensity=1.0,
                              m_hat=0.5, outcome_cont=0))
    # salvage loss ~0, lower bound stays below alpha
    assert not mon.drift_detected()


import pytest