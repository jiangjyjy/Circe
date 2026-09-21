from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# The five interventions of the full controller (§1.7). HALT is special-cased:
# it becomes admissible only after the e-process certificate fires.
CONTINUE, COMPACT, REWIND, ESCALATE, HALT = "cont", "compact", "rewind", "escalate", "halt"

NON_HALT_ACTIONS = (COMPACT, REWIND, ESCALATE)


def optimistic_index(gain: float, sigma: float, cost: float, beta: float = 1.0) -> float:
    """The cost-normalized optimistic index ``I_u = (Δ̂_u + β·σ_u) / ĉ_u`` (§1.7).

    ``gain`` is the estimated gain in success probability, ``sigma`` a conformal
    upper bound on that estimate's error (UCB exploration), and ``cost`` the
    expected added cost. Division by a floor keeps the index finite.
    """
    return (gain + beta * sigma) / max(cost, 1e-9)


@dataclass
class InterventionModels:
    """Per-intervention gain/cost/uncertainty models that feed the index.

    ``gain`` maps an action to ``Callable[[features], float]`` returning the
    estimated success-probability gain ``Δ̂_u`` in ``[0, 1]``. ``sigma`` holds
    the conformal uncertainty (additive on the gain) and ``cost`` the expected
    added monetary cost ``ĉ_u``.
    """

    gain: dict[str, Callable[[dict[str, float]], float]]
    sigma: dict[str, float] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    beta: float = 1.0

    def index(self, action: str, features: dict[str, float]) -> float:
        g = float(self.gain[action](features))
        return optimistic_index(g, self.sigma.get(action, 0.0), self.cost.get(action, 1.0), self.beta)

    def choose(self, features: dict[str, float], actions: tuple[str, ...] = NON_HALT_ACTIONS) -> str:
        """Return the intervention with the largest optimistic index (§1.7).

        Falls back to ``cont`` when the winning intervention's estimated gain is
        non-positive — i.e. no intervention is expected to help, so the episode
        should continue.
        """
        best = max(actions, key=lambda u: self.index(u, features))
        if float(self.gain[best](features)) <= 0.0:
            return CONTINUE
        return best


def threshold_based_models(circe_full: dict[str, Any]) -> InterventionModels:
    """Build a concrete gain model from the ``circe_full`` thresholds.

    This is a default, calibration-free parameterization of Δ̂_u: each
    intervention's estimated gain grows monotonically with the severity of the
    condition it addresses. Replace ``gain``/``sigma``/``cost`` with fitted
    models for the full §1.7 treatment.
    """
    compact_t = float(circe_full["compact_context_threshold"])
    rewind_t = float(circe_full["rewind_error_streak"])
    escalate_t = float(circe_full["escalate_value_threshold"])

    def gain_compact(f: dict[str, float]) -> float:
        occ = f["f4_context_occupancy"]
        return min(1.0, max(0.0, (occ - compact_t) / max(1.0 - compact_t, 1e-9)))

    def gain_rewind(f: dict[str, float]) -> float:
        return min(1.0, f["f2_error_streak"] / max(rewind_t, 1.0))

    def gain_escalate(f: dict[str, float]) -> float:
        v = f.get("value", 0.5)
        return min(1.0, max(0.0, (escalate_t - v) / max(escalate_t, 1e-9)))

    return InterventionModels(
        gain={COMPACT: gain_compact, REWIND: gain_rewind, ESCALATE: gain_escalate},
        sigma={COMPACT: 0.05, REWIND: 0.05, ESCALATE: 0.05},
        cost={COMPACT: 0.012, REWIND: 0.045, ESCALATE: 0.086},
    )
