import math

import pytest

from circe import BettingEProcess


def test_wealth_starts_at_one_and_stays_nonnegative():
    proc = BettingEProcess(q0=0.1, alpha=0.05)
    assert proc.wealth == 1.0
    for stall in [0, 1, 0, 1, 1, 1]:
        _, wealth, _ = proc.update(stall)
        assert wealth >= 0.0
        assert math.isfinite(wealth)


def test_gamma_stays_within_paper_bound():
    proc = BettingEProcess(q0=0.2, alpha=0.05)
    gamma_max = (1.0 - 1e-9) / (1.0 - 0.2)
    for _ in range(20):
        gamma, _, _ = proc.update(1)
        assert 0.0 <= gamma <= gamma_max


def test_certificate_fires_after_sustained_stalls():
    proc = BettingEProcess(q0=0.1, alpha=0.05)
    fired = False
    for _ in range(40):
        _, _, certificate = proc.update(1)
        if certificate:
            fired = True
            break
    assert fired


def test_no_certificate_when_stall_rate_below_q0():
    # All zeros (no stalls) never raises a certificate.
    proc = BettingEProcess(q0=0.5, alpha=0.01)
    for _ in range(100):
        _, wealth, certificate = proc.update(0)
        assert not certificate
        assert wealth <= 1.0 + 1e-9


def test_reset_clears_history_and_wealth():
    proc = BettingEProcess(q0=0.1, alpha=0.05)
    for _ in range(10):
        proc.update(1)
    assert proc.wealth > 1.0
    proc.reset()
    assert proc.wealth == 1.0
    assert proc.history == []


def test_high_q0_mixed_stalls_do_not_go_nonpositive():
    # For q0 > 0.5 the naive gamma bound 1/(1-q0) can drive the S=0 factor
    # 1 - gamma*q0 negative. The clipped bound must keep every factor positive.
    proc = BettingEProcess(q0=0.8, alpha=0.05)
    for stall in [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]:
        _, wealth, _ = proc.update(stall)
        assert wealth > 0.0
        assert math.isfinite(wealth)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        BettingEProcess(q0=0.0, alpha=0.05)
    with pytest.raises(ValueError):
        BettingEProcess(q0=0.1, alpha=1.0)
    proc = BettingEProcess(q0=0.1, alpha=0.05)
    with pytest.raises(ValueError):
        proc.update(2)
