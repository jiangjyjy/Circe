from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._io import stable_hash
from .calibration import SuccessModel
from .config import ControllerConfig, load_config
from .eprocess import BettingEProcess
from .features import features_from_steps, is_stall
from .interventions import COMPACT, CONTINUE, ESCALATE, NON_HALT_ACTIONS, REWIND, InterventionModels, threshold_based_models
from .schema import Decision, StepRecord, as_step_record


@dataclass
class CirceController:
    """Online, pluggable CIRCE controller.

    The harness owns the agent loop and calls :meth:`step` once per step with the
    by-product telemetry of that step. The controller is a pure decision function:
    it performs no I/O and issues no model calls.
    """

    config: ControllerConfig
    model: SuccessModel | None = None
    task_id: str = ""
    _task_text: str = ""
    _step_number: int = 0
    _buffer: list[StepRecord] = field(default_factory=list)
    _process: BettingEProcess = field(default=None, init=False, repr=False)
    _cert_streak: int = field(default=0, init=False, repr=False)
    _rewinds_used: int = field(default=0, init=False, repr=False)
    _compacts_used: int = field(default=0, init=False, repr=False)
    _reprieved: bool = field(default=False, init=False, repr=False)
    _best_checkpoint_step: int = field(default=0, init=False, repr=False)
    _best_tests_passed: int = field(default=-1, init=False, repr=False)
    _rng: random.Random = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.mode == "full" and self.model is None:
            raise ValueError("CIRCE-FULL requires a fitted SuccessModel")
        self._models: InterventionModels = threshold_based_models(self.config.circe_full)
        self.start(self.task_id or "episode")

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_frozen(cls, path: str | Path, model: SuccessModel | None = None, task_id: str = "") -> "CirceController":
        return cls(config=load_config(path), model=model, task_id=task_id)

    @classmethod
    def from_config(
        cls,
        config: ControllerConfig,
        model: SuccessModel | None = None,
        task_id: str = "",
    ) -> "CirceController":
        return cls(config=config, model=model, task_id=task_id)

    # -- episode lifecycle ------------------------------------------------- #

    def start(self, task_text: str = "", task_id: str = "") -> None:
        """Begin a new episode, resetting the wealth process and telemetry buffer."""
        if task_text:
            self._task_text = task_text
        if task_id:
            self.task_id = task_id
        self._step_number = 0
        self._buffer.clear()
        self._cert_streak = 0
        self._rewinds_used = 0
        self._compacts_used = 0
        self._reprieved = False
        self._best_checkpoint_step = 0
        self._best_tests_passed = -1
        self._process = BettingEProcess(
            q0=self.config.q0,
            alpha=self._segment_alpha(),
            prior_successes=float(self.config.eprocess.get("prior_successes", 1.0)),
            prior_failures=float(self.config.eprocess.get("prior_failures", 1.0)),
        )
        seed = int(stable_hash([self.task_id, self.config.config_id, "reprieve"])[:16], 16)
        self._rng = random.Random(seed)

    # -- decision ---------------------------------------------------------- #

    def step(self, telemetry: Any) -> Decision:
        self._step_number += 1
        record = as_step_record(telemetry, self._step_number)
        self._buffer.append(record)
        if record.tests_passed > self._best_tests_passed:
            self._best_tests_passed = record.tests_passed
            self._best_checkpoint_step = self._step_number
        features = features_from_steps(self._buffer, self._task_text, self._step_number)
        stall = is_stall(features, self.config.stall)

        gamma, wealth, certificate = self._process.update(stall)

        # A reprieved episode runs to natural termination so its full
        # counterfactual outcome is observed; it is never halted again.
        if self._reprieved:
            return Decision(action="continue", stall=stall, wealth=wealth)

        self._cert_streak = self._cert_streak + 1 if certificate else 0
        if self._cert_streak >= self.config.patience:
            if self._maybe_reprieve():
                self._reprieved = True
                return Decision(action="continue", stall=stall, wealth=wealth)
            return Decision(action="halt", stall=stall, wealth=wealth, artifact=self._artifact_ref())

        intervention = self._check_interventions(features, wealth)
        if intervention is not None:
            return intervention

        return Decision(action="continue", stall=stall, wealth=wealth)

    def _maybe_reprieve(self) -> bool:
        if self.config.reprieve_rate <= 0:
            return False
        return self._rng.random() < self.config.reprieve_rate

    def _check_interventions(self, features: dict[str, float], wealth: float) -> Decision | None:
        if self.config.mode != "full" or self.model is None:
            return None
        value = self._predict_success(features)
        augmented = {**features, "value": value}
        actions = list(NON_HALT_ACTIONS)
        if self._rewinds_used >= int(self.config.circe_full["max_rewinds"]):
            actions.remove(REWIND)
        if self._compacts_used >= int(self.config.circe_full["max_compacts"]):
            actions.remove(COMPACT)
        action = self._models.choose(augmented, tuple(actions))
        if action == CONTINUE:
            return None
        depth = 0
        summary = ""
        if action == REWIND:
            self._rewinds_used += 1
            depth = int(self.config.circe_full.get("rewind_depth", 1.0))
            summary = self._failure_summary()
            self._rewind(depth)
        elif action == COMPACT:
            self._compacts_used += 1
            self._reset_wealth()
        return Decision(
            action=action,
            k=depth,
            level="high" if action == ESCALATE else "",
            summary=summary,
            wealth=wealth,
        )

    def _predict_success(self, features: dict[str, float]) -> float:
        # The value model consumes a 34-vector in FEATURE_NAMES order.
        import numpy as np

        from .features import FEATURE_NAMES

        vector = np.asarray([features[name] for name in FEATURE_NAMES], dtype=float).reshape(1, -1)
        probabilities = self.model.estimator.predict_proba(vector)  # type: ignore[attr-defined]
        classes = list(self.model.estimator.classes_)  # type: ignore[attr-defined]
        if 1 not in classes:
            return 0.0
        return float(probabilities[0, classes.index(1)])

    def _reset_wealth(self) -> None:
        self._process.reset()
        self._cert_streak = 0

    def _rewind(self, depth: int) -> None:
        """Restore the internal buffer and step counter to a checkpoint ``depth`` steps back."""
        depth = min(max(depth, 1), self._step_number)
        del self._buffer[-depth:]
        self._step_number -= depth
        # A best checkpoint that lived inside the rolled-back tail no longer
        # exists; recompute it from the remaining buffer.
        if self._best_checkpoint_step > self._step_number:
            self._best_checkpoint_step = 0
            self._best_tests_passed = -1
            for r in self._buffer:
                if r.tests_passed > self._best_tests_passed:
                    self._best_tests_passed = r.tests_passed
                    self._best_checkpoint_step = r.step
        self._reset_wealth()

    def _segment_alpha(self) -> float:
        """Per-segment false-kill level for the full controller.

        Each reversible intervention (COMPACT/REWIND) resets the wealth process
        and therefore opens a fresh halt opportunity at level alpha. Allocating
        the episode-level budget uniformly across the (bounded) number of
        segments keeps the whole-episode false-kill probability within alpha by
        a union bound.
        """
        if self.config.mode != "full":
            return self.config.alpha
        reversible = 1 + int(self.config.circe_full.get("max_rewinds", 0)) + int(
            self.config.circe_full.get("max_compacts", 0)
        )
        return self.config.alpha / max(reversible, 1)

    def _failure_summary(self) -> str:
        """A human-readable failure summary to re-seed the agent after a rewind."""
        errors = [s.error_signature for s in self._buffer[-4:] if s.error_signature]
        tests = self._buffer[-1].tests_passed if self._buffer else 0
        parts = [f"no progress after {len(self._buffer)} steps; tests_passed={tests}"]
        if errors:
            parts.append("recent errors: " + ", ".join(errors[-3:]))
        return "; ".join(parts)

    def _artifact_ref(self) -> str:
        """Reference to the best partial checkpoint observed so far."""
        if self._best_checkpoint_step <= 0:
            return ""
        return f"checkpoint:{self._best_checkpoint_step}"

    # -- observability ----------------------------------------------------- #

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step": self._step_number,
            "wealth": self._process.wealth,
            "certificate": self._process.certificate,
            "certificate_streak": self._cert_streak,
            "rewinds_used": self._rewinds_used,
            "compacts_used": self._compacts_used,
            "segment_alpha": self._segment_alpha(),
            "best_checkpoint_step": self._best_checkpoint_step,
            "reprieved": self._reprieved,
            "stall_history": list(self._process.history),
            "config_id": self.config.config_id,
        }

    reset = start