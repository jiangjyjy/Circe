from circe import StepRecord, features_from_steps, is_stall

STALL = {"max_novel_bytes_per_token": 1.0, "max_test_pass_delta": 0.0, "max_observation_novelty": 0.7}


def _steps(n, novel_bytes, observation=""):
    return [
        StepRecord(
            step=i + 1,
            novel_bytes=novel_bytes,
            observation_text=observation,
            tests_passed=0,
            tests_failed=0,
            tool_name="bash",
            action_signature=f"cmd-{i}",
        )
        for i in range(n)
    ]


def test_zero_novelty_is_stall():
    # four steps of zero novelty, flat tests, identical observations -> stall
    steps = _steps(8, 0, observation="same output")
    features = features_from_steps(steps, "task text", len(steps))
    assert is_stall(features, STALL) == 1


def test_new_workspace_bytes_not_stall():
    steps = _steps(8, 100, observation="")
    features = features_from_steps(steps, "task text", len(steps))
    assert is_stall(features, STALL) == 0


def test_test_progress_not_stall():
    steps = _steps(8, 0, observation="same output")
    # tests_passed climbs over the window -> test gain > 0 -> not a stall
    for i, s in enumerate(steps[4:], start=4):
        s.tests_passed = i
    features = features_from_steps(steps, "task text", len(steps))
    assert is_stall(features, STALL) == 0


def test_feature_manifest_length():
    from circe import FEATURE_NAMES

    assert len(FEATURE_NAMES) == 34
    features = features_from_steps(_steps(8, 0), "task text", 8)
    assert tuple(features) == FEATURE_NAMES