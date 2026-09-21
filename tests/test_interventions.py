import pytest

from circe import InterventionModels, optimistic_index, threshold_based_models
from circe.interventions import COMPACT, CONTINUE, ESCALATE, REWIND

CIRCE_FULL = {
    "compact_context_threshold": 0.98,
    "rewind_error_streak": 3.0,
    "escalate_value_threshold": 0.5,
    "max_rewinds": 2,
}


def test_optimistic_index_formula():
    assert optimistic_index(0.3, 0.05, 0.1, beta=1.0) == pytest.approx(3.5)
    # larger uncertainty raises the index (UCB exploration)
    assert optimistic_index(0.3, 0.1, 0.1, beta=1.0) > optimistic_index(0.3, 0.05, 0.1, beta=1.0)


def test_choose_prefers_highest_index():
    models = threshold_based_models(CIRCE_FULL)
    # occupancy well under threshold (compact gain 0), no errors, low value -> escalate
    action = models.choose({"f4_context_occupancy": 0.9, "f2_error_streak": 0.0, "value": 0.1})
    assert action == ESCALATE


def test_choose_returns_continue_when_no_intervention_needed():
    models = threshold_based_models(CIRCE_FULL)
    action = models.choose({"f4_context_occupancy": 0.5, "f2_error_streak": 0.0, "value": 0.8})
    assert action == CONTINUE


def test_rewind_wins_on_error_streak():
    models = threshold_based_models(CIRCE_FULL)
    action = models.choose({"f4_context_occupancy": 0.5, "f2_error_streak": 10.0, "value": 0.8})
    assert action == REWIND


def test_compact_wins_on_full_context():
    models = threshold_based_models(CIRCE_FULL)
    action = models.choose({"f4_context_occupancy": 0.995, "f2_error_streak": 0.0, "value": 0.8})
    assert action == COMPACT