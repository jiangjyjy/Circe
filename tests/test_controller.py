import pytest

from circe import CirceController, ControllerConfig, StepTelemetry


def _controller(q0=0.1, alpha=0.05, patience=1, mode="halt"):
    config = ControllerConfig(q0=q0, alpha=alpha, patience=patience, mode=mode)
    return CirceController.from_config(config, task_id="test-episode")


def _stall_step(i):
    return StepTelemetry(tool_name="bash", action_signature=f"a{i}", novel_bytes=0, observation_text="noop")


def _productive_step(i):
    return StepTelemetry(tool_name="bash", action_signature=f"a{i}", novel_bytes=500, observation_text=f"out{i}")


def test_controller_halts_on_sustained_stall():
    ctl = _controller()
    ctl.start(task_text="some task", task_id="ep")
    decisions = []
    # productive prefix, then a long zero-novelty tail
    for i in range(3):
        decisions.append(ctl.step(_productive_step(i)).action)
    for i in range(20):
        decisions.append(ctl.step(_stall_step(i)).action)
    assert "halt" in decisions


def test_controller_never_halts_productive_trajectory():
    ctl = _controller()
    ctl.start(task_text="some task", task_id="ep")
    for i in range(30):
        assert ctl.step(_productive_step(i)).action == "continue"


def test_reset_starts_a_fresh_episode():
    ctl = _controller()
    ctl.start(task_text="task", task_id="ep")
    for _ in range(20):
        ctl.step(_stall_step(0))
    assert ctl.snapshot()["wealth"] > 1.0
    ctl.reset("task", "ep")
    snap = ctl.snapshot()
    assert snap["wealth"] == 1.0
    assert snap["step"] == 0


def test_full_mode_requires_model():
    import pytest

    with pytest.raises(ValueError):
        ControllerConfig(mode="full")
        CirceController.from_config(ControllerConfig(mode="full"), task_id="x")


def test_reprieve_runs_to_natural_termination():
    # With reprieve_rate=1.0 the first certificate is always converted into a
    # reprieve, and the episode must then never halt again (so its full
    # counterfactual outcome can be observed).
    config = ControllerConfig(q0=0.1, alpha=0.05, patience=1, reprieve_rate=1.0)
    ctl = CirceController.from_config(config, task_id="reprieve-ep")
    ctl.start(task_text="t", task_id="reprieve-ep")
    actions = [ctl.step(_stall_step(i)).action for i in range(30)]
    assert "halt" not in actions
    assert ctl.snapshot()["reprieved"] is True


class _Estimator:
    classes_ = [0, 1]

    def predict_proba(self, X):
        import numpy as np
        return np.full((len(X), 2), 0.5)


class _Model:
    estimator = _Estimator()


def _error_step(i):
    return StepTelemetry(tool_name="bash", action_signature=f"a{i}", novel_bytes=100,
                         tool_status="error", error_signature="E", tests_passed=0)


def test_rewind_restores_buffer_and_sets_summary():
    config = ControllerConfig(mode="full", q0=0.1, alpha=0.05, patience=1)
    ctl = CirceController.from_config(config, model=_Model(), task_id="rw")
    ctl.start(task_text="t", task_id="rw")
    d = ctl.step(_error_step(0))
    assert d.action == "rewind"
    assert d.summary != ""
    assert d.k >= 1
    # after rewind the internal buffer and step counter were rolled back
    snap = ctl.snapshot()
    assert snap["step"] <= 1
    assert snap["stall_history"] == []


def test_halt_sets_artifact_reference():
    config = ControllerConfig(q0=0.1, alpha=0.05, patience=1, mode="halt")
    ctl = CirceController.from_config(config, task_id="h")
    ctl.start(task_text="t", task_id="h")
    # a productive step with tests progress becomes the best checkpoint
    ctl.step(StepTelemetry(tool_name="bash", action_signature="p", novel_bytes=500, tests_passed=5))
    d = None
    for i in range(30):
        d = ctl.step(_stall_step(i))
        if d.action == "halt":
            break
    assert d is not None and d.action == "halt"
    assert d.artifact == "checkpoint:1"


def test_rewind_clears_rolled_back_checkpoint():
    config = ControllerConfig(mode="full", q0=0.1, alpha=0.05, patience=1)
    ctl = CirceController.from_config(config, model=_Model(), task_id="rw")
    ctl.start(task_text="t", task_id="rw")
    # step 1: an error AND tests progress -> the best checkpoint lands on step 1,
    # which rewind then rolls back
    d = ctl.step(StepTelemetry(tool_name="bash", action_signature="a0", novel_bytes=100,
                               tool_status="error", error_signature="E", tests_passed=5))
    assert d.action == "rewind"
    # the best checkpoint was inside the rolled-back tail, so it must be cleared
    assert ctl.snapshot()["best_checkpoint_step"] == 0


def test_segment_alpha_splits_budget_in_full_mode_only():
    halt = ControllerConfig(mode="halt", q0=0.1, alpha=0.05)
    assert CirceController.from_config(halt, task_id="h").snapshot()["segment_alpha"] == pytest.approx(0.05)
    full = ControllerConfig(mode="full", q0=0.1, alpha=0.05)
    ctl = CirceController.from_config(full, model=_Model(), task_id="f")
    # 1 + max_rewinds(2) + max_compacts(2) = 5 segments
    assert ctl.snapshot()["segment_alpha"] == pytest.approx(0.05 / 5)


def test_config_merges_nested_defaults_for_old_configs():
    old = {
        "alpha": 0.05, "delta": 0.05, "q0": 0.1, "patience": 1, "reprieve_rate": 0.05, "mode": "full",
        "stall": {"max_novel_bytes_per_token": 1.0, "max_test_pass_delta": 0, "max_observation_novelty": 0.7},
        "eprocess": {"prior_successes": 1.0, "prior_failures": 1.0},
        "circe_full": {"compact_context_threshold": 0.98, "rewind_error_streak": 3,
                       "escalate_value_threshold": 0.5, "max_rewinds": 2},
    }
    cfg = ControllerConfig.from_dict(old)
    assert cfg.circe_full["max_compacts"] == 2.0
    assert cfg.circe_full["rewind_depth"] == 1.0
