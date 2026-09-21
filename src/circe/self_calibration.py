from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .calibration import SuccessModel


@dataclass
class FiredEpisode:
    """One episode in which the controller fired (raised a doom certificate).

    Carries exactly the quantities the doubly-robust salvage-loss estimator of
    §1.6 needs. ``outcome_cont`` (the counterfactual ``Y^{κ_∅}``) is observed
    only when the episode was reprieved; otherwise it is ``None``.
    """

    episode_id: str
    round: int
    m_hat: float               # fitted success probability V̂(Z_τ) at firing time
    reprieved: bool            # D_i
    propensity: float          # ε_i
    outcome_cont: int | None   # Y^{κ_∅}, observed iff reprieved
    outcome_halt: int          # Y^{κ_λ} (0 for a mid-trajectory halt)


def dr_counterfactual(ep: FiredEpisode) -> float:
    """Doubly-robust estimate of ``Y^{κ_∅}`` (would the episode have succeeded).

    ``m̂ + (D/ε)(Y^{κ_∅} − m̂)`` — the model term plus an inverse-propensity
    correction that is active only on reprieved episodes.
    """
    if ep.reprieved:
        assert ep.outcome_cont is not None, "reprieved episode must observe its counterfactual outcome"
        return ep.m_hat + (ep.outcome_cont - ep.m_hat) / ep.propensity
    return ep.m_hat


def doubly_robust_salvage_loss(fired: list[FiredEpisode], n_total: int) -> float:
    """Estimate R(λ) = E[Y^{κ_∅} − Y^{κ_λ}] over the fired episodes (§1.6, Prop. 7).

    ``(1/n) Σ_{i∈K_λ}[m̂ + (D_i/ε_i)(Y^{κ_∅} − m̂)] − (1/n) Σ_{i∈K_λ} Y^{κ_λ}``.
    """
    if n_total <= 0:
        return 0.0
    correction = sum(dr_counterfactual(ep) for ep in fired) - sum(ep.outcome_halt for ep in fired)
    return correction / n_total


def reprieve_schedule(t: int, n_configs: int, mean_reprieve_cost: float) -> float:
    """The regret-optimal reprieve rate ``ε_t = min{1, (log|Λ| / (t·c̄))^{1/3}}``.

    ``c̄`` is the mean cost of completing a certified-doomed episode; if it is
    unknown or zero a conservative floor is used so the schedule remains in
    ``(0, 1]`` for ``t ≥ 1``.
    """
    if t < 1:
        return 1.0
    c = max(mean_reprieve_cost, 1e-9)
    raw = (math.log(max(n_configs, 1)) / (t * c)) ** (1.0 / 3.0)
    return min(1.0, raw)


def empirical_bernstein_radius(values: list[float], delta: float, bound: float) -> float:
    """Time-uniform empirical-Bernstein radius for a ``[0, bound]``-valued mean.

    Uses the empirical variance with a ``log(1/δ) + O(log log t)`` time-uniform
    factor, so the resulting interval ``mean ± radius`` is a (1−δ) confidence
    sequence for the running mean (valid simultaneously over all ``t``).
    """
    t = len(values)
    if t == 0:
        return float("inf")
    mean = sum(values) / t
    variance = sum((x - mean) ** 2 for x in values) / t
    log_factor = math.log(2.0 / delta) + 3.0 * math.log(max(1.0, math.log(2 * t + 1) + 1.0))
    return math.sqrt(2.0 * variance * log_factor / t) + (bound * log_factor) / (3.0 * t)


@dataclass
class SelfCalibratingMonitor:
    """Online salvage-loss tracking and drift detection (§1.6).

    Accumulates fired episodes across deployment rounds, maintains the
    doubly-robust salvage-loss estimate together with a time-uniform confidence
    sequence, and flags drift when the lower confidence bound rises above the
    risk budget ``alpha``.
    """

    alpha: float = 0.05
    delta: float = 0.05
    fired: list[FiredEpisode] = field(default_factory=list)
    n_total: int = 0
    _round: int = 0
    model: SuccessModel | None = None

    @property
    def round(self) -> int:
        return self._round

    def begin_round(self, n_episodes: int) -> None:
        self._round += 1
        self.n_total += n_episodes

    def record_fire(self, ep: FiredEpisode) -> None:
        self.fired.append(ep)

    def record_nonfire(self) -> None:
        # Non-fired episodes contribute to the denominator n but not to K_λ.
        pass

    def _dr_terms(self) -> list[float]:
        return [dr_counterfactual(ep) - ep.outcome_halt for ep in self.fired]

    def estimate(self) -> dict[str, Any]:
        """Return R̂(λ), its confidence half-width, and the lower/upper ends."""
        n = max(self.n_total, 1)
        terms = self._dr_terms()
        point = sum(terms) / n
        eps_min = min((ep.propensity for ep in self.fired), default=1.0)
        bound = 1.0 / max(eps_min, 1e-9) + 1.0
        radius = empirical_bernstein_radius(terms, self.delta, bound)
        return {
            "salvage_loss": point,
            "radius": radius,
            "lower": point - radius,
            "upper": point + radius,
            "n_fired": len(self.fired),
            "n_total": self.n_total,
            "round": self._round,
        }

    def drift_detected(self) -> bool:
        est = self.estimate()
        return est["lower"] > self.alpha

    def reprieve_rate(self, n_configs: int, mean_reprieve_cost: float) -> float:
        return reprieve_schedule(self._round, n_configs, mean_reprieve_cost)
