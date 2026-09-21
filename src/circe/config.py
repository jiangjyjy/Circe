from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ._io import atomic_write_json, stable_hash

_DEFAULT_STALL = {
    "max_novel_bytes_per_token": 1.0,
    "max_test_pass_delta": 0.0,
    "max_observation_novelty": 0.7,
}
_DEFAULT_EPROCESS = {"prior_successes": 1.0, "prior_failures": 1.0}
_DEFAULT_CIRCE_FULL = {
    "compact_context_threshold": 0.98,
    "rewind_error_streak": 3.0,
    "escalate_value_threshold": 0.5,
    "max_rewinds": 2.0,
    "max_compacts": 2.0,
    "rewind_depth": 1.0,
}


@dataclass(frozen=True)
class ControllerConfig:
    """Frozen configuration for a CIRCE controller.

    ``mode`` is ``"halt"`` for CIRCE-HALT (e-process only) or ``"full"`` for
    CIRCE-FULL (adds COMPACT/REWIND/ESCALATE interventions, which require a
    fitted ``SuccessModel``).
    """

    alpha: float = 0.05
    delta: float = 0.05
    q0: float = 0.1
    patience: int = 1
    reprieve_rate: float = 0.0
    mode: str = "halt"
    stall: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_STALL))
    eprocess: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_EPROCESS))
    circe_full: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_CIRCE_FULL))

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be in (0,1)")
        if not 0 < self.q0 <= 1:
            raise ValueError("q0 must be in (0,1]")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if not 0 <= self.reprieve_rate <= 1:
            raise ValueError("reprieve_rate must be in [0,1]")
        if self.mode not in {"halt", "full"}:
            raise ValueError("mode must be 'halt' or 'full'")
        # Backward-compatible default merge: a config that omits newer nested
        # keys (e.g. max_compacts / rewind_depth) falls back to the defaults
        # instead of raising KeyError downstream.
        for key, defaults in (
            ("stall", _DEFAULT_STALL),
            ("eprocess", _DEFAULT_EPROCESS),
            ("circe_full", _DEFAULT_CIRCE_FULL),
        ):
            current = getattr(self, key)
            for name, default in defaults.items():
                current.setdefault(name, default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ControllerConfig":
        return cls(**value)

    @property
    def config_id(self) -> str:
        return stable_hash(self.to_dict())[:16]

    @property
    def protocol(self) -> dict[str, Any]:
        """A ``protocol``-shaped view for the offline policy/evaluation API."""
        return {
            "alpha": self.alpha,
            "delta": self.delta,
            "stall": dict(self.stall),
            "eprocess": dict(self.eprocess),
            "circe_full": dict(self.circe_full),
            "reprieve_rate": self.reprieve_rate,
        }


def load_config(path: str | Path) -> ControllerConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return ControllerConfig.from_dict(data)


def freeze_config(config: ControllerConfig, path: str | Path) -> Path:
    config_path = Path(path)
    atomic_write_json(config_path, config.to_dict())
    return config_path