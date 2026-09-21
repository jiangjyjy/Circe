from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from ._io import atomic_write_json
from .features import FEATURE_NAMES, extract_feature_dict, extract_feature_vector, is_stall
from .schema import Trajectory


@dataclass
class SuccessModel:
    """Fitted outcome model V_hat used by the full controller (and B9-style baselines)."""

    estimator: object

    def predict_success(self, trajectory: Trajectory, through_step: int) -> float:
        vector = extract_feature_vector(trajectory, through_step).reshape(1, -1)
        probabilities = self.estimator.predict_proba(vector)  # type: ignore[attr-defined]
        classes = list(self.estimator.classes_)  # type: ignore[attr-defined]
        if 1 not in classes:
            return 0.0
        return float(probabilities[0, classes.index(1)])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump({"feature_names": FEATURE_NAMES, "estimator": self.estimator}, handle)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SuccessModel":
        with path.open("rb") as handle:
            value = pickle.load(handle)
        if tuple(value["feature_names"]) != FEATURE_NAMES:
            raise ValueError("model feature manifest does not match runtime")
        return cls(estimator=value["estimator"])


def train_success_model(trajectories: Iterable[Trajectory], seed: int) -> SuccessModel:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for trajectory in trajectories:
        for step in range(1, len(trajectory.steps) + 1):
            rows.append(extract_feature_vector(trajectory, step))
            labels.append(trajectory.final_outcome)
    if not rows:
        raise ValueError("no training rows")
    x = np.vstack(rows)
    y = np.asarray(labels, dtype=int)
    if len(set(labels)) == 1:
        estimator = DummyClassifier(strategy="constant", constant=labels[0])
    else:
        estimator = HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=120,
            learning_rate=0.06,
            l2_regularization=0.1,
            random_state=seed,
        )
    estimator.fit(x, y)
    return SuccessModel(estimator=estimator)


def calibrate_q0(
    trajectories: Iterable[Trajectory],
    stall_thresholds: dict[str, float],
    delta: float,
) -> dict[str, float | int]:
    """Calibrate q0, the upper bound on the stall rate of salvageable trajectories.

    Uses a step-level Clopper-Pearson upper bound on the Bernoulli stall rate of
    *successful* trajectories (each step is one trial), floored at the observed
    mean. A task-level empirical-Bernstein bound is also recorded for audit.
    """
    rates: list[float] = []
    total_steps = 0
    total_stalls = 0
    for trajectory in trajectories:
        if trajectory.final_outcome != 1:
            continue
        indicators = [
            is_stall(extract_feature_dict(trajectory, step), stall_thresholds)
            for step in range(1, len(trajectory.steps) + 1)
        ]
        rates.append(sum(indicators) / len(indicators))
        total_steps += len(indicators)
        total_stalls += sum(indicators)
    if not rates:
        raise ValueError("q0 calibration requires at least one successful trajectory")
    n = len(rates)
    mean = float(np.mean(rates))
    variance = float(np.var(rates, ddof=1)) if n > 1 else 0.25
    log_term = math.log(3.0 / delta)
    radius = math.sqrt(2.0 * variance * log_term / max(1, n)) + 3.0 * log_term / max(1, n)
    raw_upper = mean + radius

    from scipy.stats import beta as _beta

    if total_steps > 0:
        cp_upper = float(
            _beta.ppf(1.0 - delta, total_stalls + 1, max(1, total_steps - total_stalls))
        )
        q0_upper = min(1.0, max(cp_upper, mean))
    else:
        q0_upper = min(1.0, max(1e-4, mean * 3.0))
    return {
        "successful_tasks": n,
        "steps": total_steps,
        "stalls": total_stalls,
        "mean_task_stall_rate": mean,
        "q0_upper": q0_upper,
        "q0_method": "step_level_clopper_pearson",
        "q0_task_bernstein_upper": min(1.0, raw_upper),
        "delta": delta,
    }


def calibrate_q1(
    trajectories: Iterable[Trajectory],
    stall_thresholds: dict[str, float],
) -> dict[str, float | int]:
    """Point estimate of the doomed stall rate q1 from failed trajectories."""
    rates: list[float] = []
    total_steps = 0
    total_stalls = 0
    for trajectory in trajectories:
        if trajectory.final_outcome != 0:
            continue
        indicators = [
            is_stall(extract_feature_dict(trajectory, step), stall_thresholds)
            for step in range(1, len(trajectory.steps) + 1)
        ]
        if not indicators:
            continue
        rates.append(sum(indicators) / len(indicators))
        total_steps += len(indicators)
        total_stalls += sum(indicators)
    if not rates:
        raise ValueError("q1 calibration requires at least one failed trajectory")
    n = len(rates)
    mean = float(np.mean(rates))
    return {
        "doomed_tasks": n,
        "steps": total_steps,
        "stalls": total_stalls,
        "mean_task_stall_rate": mean,
        "q1": mean,
    }


def bernoulli_kl(q1: float, q0: float) -> float:
    """Bernoulli KL divergence kl(q1 || q0), used for the predicted detection delay."""
    if not 0 < q0 < 1:
        raise ValueError(f"q0 must be in (0,1) for KL, got {q0}")
    if not 0 <= q1 <= 1:
        raise ValueError(f"q1 must be in [0,1], got {q1}")
    if q1 == q0:
        return 0.0
    q1c = min(max(q1, 1e-12), 1 - 1e-12)
    q0c = min(max(q0, 1e-12), 1 - 1e-12)
    return q1c * math.log(q1c / q0c) + (1 - q1c) * math.log((1 - q1c) / (1 - q0c))


def save_calibration(path: Path, calibration: dict[str, object]) -> None:
    atomic_write_json(path, calibration)
