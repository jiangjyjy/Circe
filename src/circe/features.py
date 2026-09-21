from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np

from .schema import StepRecord, Trajectory


FEATURE_NAMES = (
    # F1: control-flow recurrence (8)
    "f1_tool_repeat_rate_w4",
    "f1_action_signature_repeat_rate_w8",
    "f1_command_bigram_repeat_rate_w8",
    "f1_same_tool_streak",
    "f1_cycle_multiplicity_w8",
    "f1_state_revisit_rate_w8",
    "f1_noop_rate_w8",
    "f1_action_entropy_w8",
    # F2: environment feedback (8)
    "f2_tool_error_rate_w4",
    "f2_error_streak",
    "f2_error_identity_rate_w8",
    "f2_timeout_rate_w8",
    "f2_test_pass_delta_w4",
    "f2_test_fail_delta_w4",
    "f2_compile_failure_streak",
    "f2_observation_novelty_w4",
    # F3: workspace productivity (8)
    "f3_files_touched_w4",
    "f3_unique_files_touched_w8",
    "f3_bytes_added_w4",
    "f3_bytes_deleted_w4",
    "f3_novel_bytes_per_output_token_w4",
    "f3_revert_ratio_w8",
    "f3_workspace_tree_edit_distance_w8",
    "f3_productive_tool_rate_w8",
    # F4: context pressure (5)
    "f4_context_occupancy",
    "f4_fresh_input_growth_w4",
    "f4_output_token_growth_w4",
    "f4_cached_prefix_ratio",
    "f4_compaction_count",
    # F5: lexical self-report (5)
    "f5_hedging_score",
    "f5_goal_restatement_score",
    "f5_apology_retry_score",
    "f5_imminent_completion_score",
    "f5_refusal_score",
)

assert len(FEATURE_NAMES) == 34

# Multiplicative weight applied to the F5 lexical self-report family (§1.3).
F5_WEIGHT = 0.5


_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_HEDGING = re.compile(r"\b(maybe|perhaps|might|possibly|uncertain|seems?)\b|可能|也许|不确定", re.I)
_APOLOGY_RETRY = re.compile(r"\b(sorry|apolog|retry|try again|another attempt)\b|抱歉|重试|再试", re.I)
_IMMINENT = re.compile(r"\b(almost done|nearly complete|final step|about to finish)\b|快完成|最后一步", re.I)
_REFUSAL = re.compile(r"\b(cannot|can't|unable|give up|impossible|refuse)\b|无法|不能|放弃", re.I)


def _window(items: list[StepRecord], size: int) -> list[StepRecord]:
    return items[-size:]


def _duplicate_rate(values: Iterable[str]) -> float:
    sequence = [value for value in values if value]
    if not sequence:
        return 0.0
    return 1.0 - len(set(sequence)) / len(sequence)


def _streak(values: list[str], predicate) -> int:
    total = 0
    for value in reversed(values):
        if predicate(value):
            total += 1
        else:
            break
    return total


def _normalized_entropy(values: list[str]) -> float:
    if len(values) <= 1:
        return 0.0
    counts = Counter(values)
    probabilities = [count / len(values) for count in counts.values()]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    maximum = math.log(len(counts))
    return entropy / maximum if maximum > 0 else 0.0


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text)}


def _jaccard(left: str, right: str) -> float:
    a, b = _token_set(left), _token_set(right)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _cycle_multiplicity(values: list[str]) -> float:
    if len(values) < 2:
        return 0.0
    best = 0
    for cycle_len in range(1, min(4, len(values) // 2 + 1)):
        tail = values[-cycle_len:]
        repeats = 1
        cursor = len(values) - 2 * cycle_len
        while cursor >= 0 and values[cursor : cursor + cycle_len] == tail:
            repeats += 1
            cursor -= cycle_len
        best = max(best, repeats)
    return min(1.0, max(0, best - 1) / 3.0)


def _bigram_repeat_rate(values: list[str]) -> float:
    if len(values) < 3:
        return 0.0
    bigrams = list(zip(values, values[1:]))
    return _duplicate_rate(["|".join(item) for item in bigrams])


def _growth(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    baseline = max(1, values[0])
    return (values[-1] - values[0]) / baseline


def _tree_distance(left: dict[str, str], right: dict[str, str]) -> float:
    """Normalized tree-edit-style distance between two workspace snapshots.

    Each snapshot maps ``file-path -> content-hash``. The distance is the
    fraction of paths whose content changed or that were added/removed, i.e.
    ``1 - |left ∩ right| / |left ∪ right|`` over ``(path, hash)`` pairs.
    Returns 0.0 when neither snapshot is populated (no workspace telemetry).
    """
    if not left and not right:
        return 0.0
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    changed = sum(1 for key in keys if left.get(key) != right.get(key))
    return changed / len(keys)


def features_from_steps(steps: list[StepRecord], task_text: str, through_step: int) -> dict[str, float]:
    """Compute the 34-feature vector for the prefix ``steps[:through_step]``.

    Accepts a raw step list (rather than a full ``Trajectory``) so the online
    controller can feed its rolling buffer without fabricating a trajectory.
    """
    if not 1 <= through_step <= len(steps):
        raise ValueError("through_step outside steps")
    prefix = steps[:through_step]
    w4, w8 = _window(prefix, 4), _window(prefix, 8)
    current = prefix[-1]

    tools4 = [step.tool_name for step in w4]
    signatures8 = [step.action_signature for step in w8]
    tool_streak = _streak([step.tool_name for step in prefix], lambda value: value == current.tool_name)
    error_streak = _streak([step.tool_status for step in prefix], lambda value: value != "ok")
    compile_streak = _streak([step.compile_ok for step in prefix], lambda value: not value)

    observations = [step.observation_text for step in w4]
    if len(observations) <= 1:
        observation_novelty = 1.0
    else:
        observation_novelty = float(np.mean([1.0 - _jaccard(a, b) for a, b in zip(observations, observations[1:])]))

    previous4 = prefix[max(0, through_step - 5)] if through_step > 1 else prefix[0]
    test_pass_delta = current.tests_passed - previous4.tests_passed
    test_fail_delta = current.tests_failed - previous4.tests_failed
    files = [name for step in w8 for name in step.files_touched]
    bytes_written = sum(step.bytes_added for step in w8)
    reverted = sum(step.reverted_bytes for step in w8)
    fresh_tokens = [step.prompt_tokens - step.cached_tokens for step in w4]
    output_tokens = [step.output_tokens for step in w4]
    text = current.assistant_text

    values = {
        "f1_tool_repeat_rate_w4": _duplicate_rate(tools4),
        "f1_action_signature_repeat_rate_w8": _duplicate_rate(signatures8),
        "f1_command_bigram_repeat_rate_w8": _bigram_repeat_rate(signatures8),
        "f1_same_tool_streak": float(tool_streak),
        "f1_cycle_multiplicity_w8": _cycle_multiplicity(signatures8),
        "f1_state_revisit_rate_w8": _duplicate_rate([step.state_hash for step in w8]),
        "f1_noop_rate_w8": sum(step.noop for step in w8) / len(w8),
        "f1_action_entropy_w8": _normalized_entropy([step.tool_name for step in w8]),
        "f2_tool_error_rate_w4": sum(step.tool_status != "ok" for step in w4) / len(w4),
        "f2_error_streak": float(error_streak),
        "f2_error_identity_rate_w8": _duplicate_rate([step.error_signature for step in w8 if step.error_signature]),
        "f2_timeout_rate_w8": sum(step.timed_out for step in w8) / len(w8),
        "f2_test_pass_delta_w4": float(test_pass_delta),
        "f2_test_fail_delta_w4": float(test_fail_delta),
        "f2_compile_failure_streak": float(compile_streak),
        "f2_observation_novelty_w4": observation_novelty,
        "f3_files_touched_w4": float(sum(len(step.files_touched) for step in w4)),
        "f3_unique_files_touched_w8": float(len(set(files))),
        "f3_bytes_added_w4": float(sum(step.bytes_added for step in w4)),
        "f3_bytes_deleted_w4": float(sum(step.bytes_deleted for step in w4)),
        "f3_novel_bytes_per_output_token_w4": float(sum(step.novel_bytes for step in w4)) / max(1.0, float(sum(step.output_tokens for step in w4))),
        "f3_revert_ratio_w8": reverted / max(1, bytes_written),
        "f3_workspace_tree_edit_distance_w8": (
            float(np.mean(
                [_tree_distance(prev.workspace_tree, curr.workspace_tree)
                 for prev, curr in zip(w8, w8[1:])]
            ))
            if len(w8) >= 2 else 0.0
        ),
        "f3_productive_tool_rate_w8": sum((step.novel_bytes > 0 or step.tests_passed > 0) and not step.noop for step in w8) / len(w8),
        "f4_context_occupancy": current.prompt_tokens / max(1, current.context_limit),
        "f4_fresh_input_growth_w4": _growth(fresh_tokens),
        "f4_output_token_growth_w4": _growth(output_tokens),
        "f4_cached_prefix_ratio": current.cached_tokens / max(1, current.prompt_tokens),
        "f4_compaction_count": float(current.compaction_count),
        "f5_hedging_score": float(len(_HEDGING.findall(text))),
        "f5_goal_restatement_score": _jaccard(text, task_text),
        "f5_apology_retry_score": float(len(_APOLOGY_RETRY.findall(text))),
        "f5_imminent_completion_score": float(len(_IMMINENT.findall(text))),
        "f5_refusal_score": float(len(_REFUSAL.findall(text))),
    }
    # Deliberately down-weight the F5 lexical self-report family: it is the one
    # channel an adversary or a sycophantic model can poison, and the paper
    # (§1.3) down-weights it and reports an ablation without it.
    for name in (
        "f5_hedging_score",
        "f5_goal_restatement_score",
        "f5_apology_retry_score",
        "f5_imminent_completion_score",
        "f5_refusal_score",
    ):
        values[name] *= F5_WEIGHT
    if tuple(values) != FEATURE_NAMES:
        raise AssertionError("feature manifest/order drift")
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite telemetry value")
    return values


def extract_feature_dict(trajectory: Trajectory, through_step: int) -> dict[str, float]:
    return features_from_steps(trajectory.steps, trajectory.task_text, through_step)


def extract_feature_vector(trajectory: Trajectory, through_step: int) -> np.ndarray:
    values = extract_feature_dict(trajectory, through_step)
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=float)


def is_stall(features: dict[str, float], thresholds: dict[str, float]) -> int:
    no_workspace = features["f3_novel_bytes_per_output_token_w4"] <= float(
        thresholds["max_novel_bytes_per_token"]
    )
    no_test_gain = features["f2_test_pass_delta_w4"] <= float(thresholds["max_test_pass_delta"])
    low_observation_novelty = features["f2_observation_novelty_w4"] <= float(
        thresholds["max_observation_novelty"]
    )
    return int(no_workspace and no_test_gain and low_observation_novelty)
