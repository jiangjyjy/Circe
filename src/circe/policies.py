from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ._io import stable_hash
from .calibration import SuccessModel
from .eprocess import BettingEProcess
from .features import extract_feature_dict, is_stall
from .interventions import CONTINUE, NON_HALT_ACTIONS, threshold_based_models
from .schema import PolicyResult, Trajectory


@dataclass
class PolicyTrace:
    halted: bool = False
    halt_step: int | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    aux_cost: float = 0.0
    note: str = ""


class Policy:
    policy_id = "UNKNOWN"

    def __init__(self, params: dict[str, Any], protocol: dict[str, Any]):
        self.params = params
        self.protocol = protocol

    @property
    def config_id(self) -> str:
        return stable_hash({"policy": self.policy_id, "params": self.params})[:16]

    def trace(self, trajectory: Trajectory, model: SuccessModel | None, q0: float) -> PolicyTrace:
        raise NotImplementedError


class NeverHaltPolicy(Policy):
    """No-op policy, used as the fallback when a policy is disabled."""

    policy_id = "NEVER"

    def trace(self, trajectory: Trajectory, model: SuccessModel | None, q0: float) -> PolicyTrace:
        return PolicyTrace()


class CirceHaltPolicy(Policy):
    policy_id = "CIRCE-HALT"

    def trace(self, trajectory: Trajectory, model: SuccessModel | None, q0: float) -> PolicyTrace:
        alpha = float(self.params.get("alpha", self.protocol["alpha"]))
        patience = int(self.params.get("patience", 1))
        econfig = self.protocol["eprocess"]
        process = BettingEProcess(
            q0=q0,
            alpha=alpha,
            prior_successes=float(econfig.get("prior_successes", 1.0)),
            prior_failures=float(econfig.get("prior_failures", 1.0)),
        )
        certificate_streak = 0
        actions: list[dict[str, Any]] = []
        for step in trajectory.steps:
            features = extract_feature_dict(trajectory, step.step)
            stall = is_stall(features, self.protocol["stall"])
            gamma, wealth, certificate = process.update(stall)
            actions.append(
                {"step": step.step, "action": "INSPECT", "stall": stall, "gamma": gamma, "wealth": wealth}
            )
            certificate_streak = certificate_streak + 1 if certificate else 0
            if certificate_streak >= patience and step.step < len(trajectory.steps):
                return PolicyTrace(halted=True, halt_step=step.step, actions=actions)
        return PolicyTrace(actions=actions)


class CirceFullPolicy(CirceHaltPolicy):
    policy_id = "CIRCE-FULL"

    def trace(self, trajectory: Trajectory, model: SuccessModel | None, q0: float) -> PolicyTrace:
        halt_trace = super().trace(trajectory, model, q0)
        if halt_trace.halted:
            return halt_trace
        models = threshold_based_models(self.protocol["circe_full"])
        for step in trajectory.steps[:-1]:
            features = extract_feature_dict(trajectory, step.step)
            value = model.predict_success(trajectory, step.step) if model else 0.5
            action = models.choose({**features, "value": value}, NON_HALT_ACTIONS)
            if action == CONTINUE:
                continue
            return PolicyTrace(actions=[{"step": step.step, "action": action.upper()}])
        return halt_trace


POLICY_TYPES = {
    "CIRCE-HALT": CirceHaltPolicy,
    "CIRCE-FULL": CirceFullPolicy,
}


def build_policy(policy_id: str, params: dict[str, Any], protocol: dict[str, Any]) -> Policy:
    if params.get("disabled"):
        policy = NeverHaltPolicy(params, protocol)
        policy.policy_id = policy_id
        return policy
    try:
        policy_type = POLICY_TYPES[policy_id]
    except KeyError as exc:
        raise ValueError(f"Unknown policy {policy_id}") from exc
    return policy_type(params, protocol)


def evaluate_halt_only(
    trajectory: Trajectory,
    policy: Policy,
    model: SuccessModel | None,
    q0: float,
    reprieve_rate: float = 0.0,
    apply_reprieve: bool = False,
) -> PolicyResult:
    """Replay a halt-only policy over a complete trajectory (offline evaluation)."""
    trace = policy.trace(trajectory, model, q0)
    if trace.note.startswith("blocked"):
        return PolicyResult(
            env=trajectory.env,
            task_id=trajectory.task_id,
            split=trajectory.split,
            policy_id=policy.policy_id,
            config_id=policy.config_id,
            y_b1=trajectory.final_outcome,
            y_policy=trajectory.final_outcome,
            cost_b1=trajectory.total_cost_usd,
            cost_policy=trajectory.total_cost_usd,
            halted=False,
            halt_step=None,
            y_halt=None,
            y_continue=None,
            cost_b1_status=trajectory.cost_accounting,
            cost_policy_status=trajectory.cost_accounting,
            status="blocked",
            note=trace.note,
        )
    if not trace.halted or trace.halt_step is None:
        return PolicyResult(
            env=trajectory.env,
            task_id=trajectory.task_id,
            split=trajectory.split,
            policy_id=policy.policy_id,
            config_id=policy.config_id,
            y_b1=trajectory.final_outcome,
            y_policy=trajectory.final_outcome,
            cost_b1=trajectory.total_cost_usd,
            cost_policy=trajectory.total_cost_usd + trace.aux_cost,
            halted=False,
            halt_step=None,
            y_halt=None,
            y_continue=None,
            cost_b1_status=trajectory.cost_accounting,
            cost_policy_status=trajectory.cost_accounting,
            operational_aux_cost=trace.aux_cost,
            note=trace.note,
        )

    halt_step = trace.halt_step
    y_halt = trajectory.steps[halt_step - 1].checkpoint_outcome
    reprieved = False
    if apply_reprieve and reprieve_rate > 0:
        coin_seed = int(stable_hash([trajectory.task_id, policy.config_id, "reprieve"])[:16], 16)
        reprieved = random.Random(coin_seed).random() < reprieve_rate
    y_policy = trajectory.final_outcome if reprieved else y_halt
    cost_policy = (
        trajectory.total_cost_usd if reprieved else trajectory.prefix_cost(halt_step)
    ) + trace.aux_cost
    return PolicyResult(
        env=trajectory.env,
        task_id=trajectory.task_id,
        split=trajectory.split,
        policy_id=policy.policy_id,
        config_id=policy.config_id,
        y_b1=trajectory.final_outcome,
        y_policy=y_policy,
        cost_b1=trajectory.total_cost_usd,
        cost_policy=cost_policy,
        halted=True,
        halt_step=halt_step,
        y_halt=y_halt,
        y_continue=trajectory.final_outcome,
        cost_b1_status=trajectory.cost_accounting,
        cost_policy_status=trajectory.cost_accounting,
        operational_aux_cost=trace.aux_cost,
        note="reprieved" if reprieved else trace.note,
    )
