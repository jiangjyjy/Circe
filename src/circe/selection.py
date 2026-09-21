from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.stats import beta

from .calibration import SuccessModel
from .policies import build_policy, evaluate_halt_only
from .schema import PolicyResult, Trajectory


def clopper_pearson_upper(events: int, total: int, delta: float) -> float:
    if total <= 0:
        return 1.0
    if events >= total:
        return 1.0
    return float(beta.ppf(1.0 - delta, events + 1, total - events))


def summarize_selection(results: Iterable[PolicyResult], delta: float) -> dict[str, float | int]:
    """Summarize a set of policy results for LTT: salvage-loss upper bound,
    pass@1 delta, and cost saving."""
    rows = [item for item in results if item.status == "ok"]
    if not rows:
        return {
            "n": 0,
            "gross_events": 0,
            "risk_ucl": 1.0,
            "delta_pass_pp": float("nan"),
            "saving_pct": float("nan"),
        }
    gross_events = sum(item.y_b1 == 1 and item.y_policy == 0 for item in rows)
    delta_pass = 100.0 * np.mean([item.y_policy - item.y_b1 for item in rows])
    cost_b1 = sum(item.cost_b1 for item in rows)
    cost_policy = sum(item.cost_policy for item in rows)
    saving = 100.0 * (1.0 - cost_policy / cost_b1) if cost_b1 > 0 else 0.0
    return {
        "n": len(rows),
        "gross_events": int(gross_events),
        "risk_ucl": clopper_pearson_upper(int(gross_events), len(rows), delta),
        "delta_pass_pp": float(delta_pass),
        "saving_pct": float(saving),
    }


def ltt_select(
    candidates: dict[str, list[dict[str, Any]]],
    order_trajectories: list[Trajectory],
    ltt_trajectories: list[Trajectory],
    q0: float,
    model: SuccessModel | None,
    protocol: dict[str, Any],
    *,
    alpha: float,
    delta: float,
    margin_pp: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Fixed-sequence LTT selection (Theorem 7).

    Candidates are ordered by predicted cost saving on the ``order`` split, then
    tested in sequence on ``ltt_trajectories``; the first candidate whose
    salvage-loss upper bound is within ``alpha`` and whose pass@1 delta is within
    ``margin_pp`` is selected. Falls back to a disabled (never-halt) policy when
    no candidate is valid.
    """
    selected: dict[str, dict[str, Any]] = {}
    for policy_id, param_list in candidates.items():
        scored: list[dict[str, Any]] = []
        for params in param_list:
            policy = build_policy(policy_id, dict(params), protocol)
            order_summary = summarize_selection(
                [evaluate_halt_only(t, policy, model, q0) for t in order_trajectories],
                delta,
            )
            scored.append({"params": dict(params), "order": order_summary})
        scored.sort(key=lambda item: float(item["order"]["saving_pct"]), reverse=True)

        choice: dict[str, Any] | None = None
        evaluated: list[dict[str, Any]] = []
        for item in scored:
            policy = build_policy(policy_id, item["params"], protocol)
            ltt_summary = summarize_selection(
                [evaluate_halt_only(t, policy, model, q0) for t in ltt_trajectories],
                delta,
            )
            record = {**item, "ltt_cal": ltt_summary}
            evaluated.append(record)
            # Fixed-sequence testing (Theorem 7): hypotheses are tested in the
            # fixed saving-descending order and the procedure stops at the first
            # candidate that FAILS to reject its null H_λ: R(λ) > α. Selecting
            # only the last rejected candidate preserves the level; skipping a
            # failure to pick a later candidate would break multiple-testing
            # control.
            if (
                float(ltt_summary["risk_ucl"]) > alpha
                or float(ltt_summary["delta_pass_pp"]) < -margin_pp
            ):
                break
            choice = record

        if choice is None:
            selected[policy_id] = {
                "params": {"disabled": True},
                "selection": "fallback_never_halt",
                "candidates": evaluated,
            }
        else:
            selected[policy_id] = {
                "params": choice["params"],
                "selection": "ltt_first_valid",
                "order": choice["order"],
                "ltt_cal": choice["ltt_cal"],
                "candidates_evaluated": len(evaluated),
            }
    return selected