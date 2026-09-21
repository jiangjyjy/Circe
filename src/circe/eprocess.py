from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BettingEProcess:
    """Nonnegative Bernoulli betting process with a predictable plug-in Kelly bet.

    The draft names aGRAPA but does not provide its update. This implementation
    chooses the log-optimal Bernoulli bet for a smoothed estimate of the
    alternative stall rate and clips gamma below ``(1 - q0)^-1``, which
    guarantees ``1 + gamma * (S - q0)`` remains positive for ``S in {0, 1}``.
    """

    q0: float
    alpha: float
    prior_successes: float = 1.0
    prior_failures: float = 1.0
    log_wealth: float = 0.0
    history: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 < self.q0 <= 1:
            raise ValueError("q0 must be in (0,1]")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if self.prior_successes <= 0 or self.prior_failures <= 0:
            raise ValueError("beta prior parameters must be positive")

    @property
    def wealth(self) -> float:
        return math.exp(min(self.log_wealth, 700.0))

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha

    @property
    def certificate(self) -> bool:
        return self.log_wealth >= math.log(self.threshold)

    def _predictable_gamma(self) -> float:
        # q0=1 is the valid but uninformative calibration fallback: the null
        # cannot be rejected for a Bernoulli stall stream, so wealth stays 1.
        if self.q0 == 1:
            return 0.0
        stalls = sum(self.history)
        total = len(self.history)
        p_hat = (stalls + self.prior_successes) / (
            total + self.prior_successes + self.prior_failures
        )
        raw = (p_hat - self.q0) / (self.q0 * (1.0 - self.q0))
        # Paper §1.4 constrains betting fractions to gamma in [0, (1-q0)^-1),
        # which keeps the S=1 factor positive. Also cap at 1/q0 so the S=0
        # factor 1 - gamma*q0 stays positive; the tighter bound matters once
        # q0 > 0.5.
        gamma_max = min((1.0 - 1e-9) / (1.0 - self.q0), (1.0 - 1e-9) / self.q0)
        if raw <= 0 and stalls > 0:
            raw = 0.5 * gamma_max
        return min(gamma_max, max(0.0, raw))

    def update(self, stall: int) -> tuple[float, float, bool]:
        if stall not in (0, 1):
            raise ValueError("stall must be 0 or 1")
        gamma = self._predictable_gamma()
        factor = 1.0 + gamma * (stall - self.q0)
        if factor <= 0:
            raise ArithmeticError("e-process factor became nonpositive")
        self.log_wealth += math.log(factor)
        self.history.append(stall)
        return gamma, self.wealth, self.certificate

    def reset(self) -> None:
        self.log_wealth = 0.0
        self.history.clear()