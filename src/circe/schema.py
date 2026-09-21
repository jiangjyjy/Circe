from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

ACCOUNTING_STATUSES = ("exact", "upper_bound", "unavailable", "unspecified")

SCHEMA_VERSION = "0.2"
ENVIRONMENTS = ("SWE", "TERM", "WEB", "TOOL")
SPLITS = ("train", "q0_cal", "order", "ltt_cal", "test", "audit")


@dataclass(frozen=True)
class TaskSpec:
    env: str
    task_id: str
    split: str
    task_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.env not in ENVIRONMENTS:
            raise ValueError(f"Unknown environment: {self.env}")
        if self.split not in SPLITS:
            raise ValueError(f"Unknown split: {self.split}")
        if not self.task_id:
            raise ValueError("task_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskSpec":
        return cls(**value)


@dataclass
class StepRecord:
    step: int
    assistant_text: str = ""
    observation_text: str = ""
    tool_name: str = "none"
    action_signature: str = "none"
    tool_status: str = "ok"
    error_signature: str = ""
    timed_out: bool = False
    noop: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    compile_ok: bool = True
    files_touched: list[str] = field(default_factory=list)
    bytes_added: int = 0
    bytes_deleted: int = 0
    novel_bytes: int = 0
    reverted_bytes: int = 0
    workspace_hash: str = ""
    workspace_tree: dict[str, str] = field(default_factory=dict)
    state_hash: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    context_limit: int = 1
    compaction_count: int = 0
    cost_usd: float = 0.0
    non_model_cost_usd: float = 0.0
    cache_accounting: str = "unspecified"
    cost_accounting: str = "unspecified"
    request_ids: list[str] = field(default_factory=list)
    usage_records: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_id: str = ""
    checkpoint_outcome: int = 0
    predicted_success: float | None = None

    def __post_init__(self) -> None:
        if self.step < 1:
            raise ValueError("step numbers are 1-based")
        if self.checkpoint_outcome not in (0, 1):
            raise ValueError("checkpoint_outcome must be binary")
        for name in ("prompt_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.cached_tokens > self.prompt_tokens:
            raise ValueError("cached_tokens cannot exceed prompt_tokens")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning_tokens cannot exceed output_tokens")
        if self.cost_usd < 0 or self.non_model_cost_usd < 0:
            raise ValueError("step costs cannot be negative")
        if self.cache_accounting not in ACCOUNTING_STATUSES:
            raise ValueError(f"invalid cache_accounting: {self.cache_accounting}")
        if self.cost_accounting not in ACCOUNTING_STATUSES:
            raise ValueError(f"invalid cost_accounting: {self.cost_accounting}")
        if not all(isinstance(value, str) and value for value in self.request_ids):
            raise ValueError("request_ids must contain non-empty strings")
        if not all(isinstance(value, dict) for value in self.usage_records):
            raise ValueError("usage_records must contain mappings")
        if self.usage_records:
            totals = {
                name: sum(int(record.get(name, 0)) for record in self.usage_records)
                for name in ("prompt_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")
            }
            for name, value in totals.items():
                if value != getattr(self, name):
                    raise ValueError(f"usage_records {name} total does not match step")
            recorded_cost = sum(float(record.get("cost_usd", 0.0)) for record in self.usage_records)
            if abs(recorded_cost + self.non_model_cost_usd - self.cost_usd) > 1e-12:
                raise ValueError("usage plus non-model cost does not match step total")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StepRecord":
        return cls(**value)


@dataclass
class StepTelemetry:
    """The slim per-step contract a harness feeds the online controller.

    Only the fields the feature extractor reads are required; everything else
    has a neutral default. A harness may instead pass a ``StepRecord`` or a
    plain ``Mapping`` of these keys.
    """

    assistant_text: str = ""
    observation_text: str = ""
    tool_name: str = "none"
    action_signature: str = "none"
    tool_status: str = "ok"
    error_signature: str = ""
    timed_out: bool = False
    noop: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    compile_ok: bool = True
    files_touched: tuple[str, ...] = ()
    bytes_added: int = 0
    bytes_deleted: int = 0
    novel_bytes: int = 0
    reverted_bytes: int = 0
    workspace_hash: str = ""
    workspace_tree: dict[str, str] = field(default_factory=dict)
    state_hash: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    context_limit: int = 1_000_000
    compaction_count: int = 0
    cost_usd: float = 0.0

    def to_step_record(self, step: int) -> StepRecord:
        return StepRecord(
            step=step,
            assistant_text=self.assistant_text,
            observation_text=self.observation_text,
            tool_name=self.tool_name,
            action_signature=self.action_signature,
            tool_status=self.tool_status,
            error_signature=self.error_signature,
            timed_out=self.timed_out,
            noop=self.noop,
            tests_passed=self.tests_passed,
            tests_failed=self.tests_failed,
            compile_ok=self.compile_ok,
            files_touched=list(self.files_touched),
            bytes_added=self.bytes_added,
            bytes_deleted=self.bytes_deleted,
            novel_bytes=self.novel_bytes,
            reverted_bytes=self.reverted_bytes,
            workspace_hash=self.workspace_hash,
            workspace_tree=self.workspace_tree,
            state_hash=self.state_hash,
            prompt_tokens=self.prompt_tokens,
            cached_tokens=self.cached_tokens,
            output_tokens=self.output_tokens,
            reasoning_tokens=self.reasoning_tokens,
            context_limit=self.context_limit,
            compaction_count=self.compaction_count,
            cost_usd=self.cost_usd,
        )


@dataclass(frozen=True)
class Decision:
    """A controller instruction for the harness.

    ``action`` is one of ``continue`` / ``halt`` / ``compact`` / ``rewind`` /
    ``escalate``. ``k`` is the rewind depth, ``level`` the escalation effort,
    ``artifact`` the best partial result to return on halt, and ``summary`` a
    failure summary to re-seed the agent on rewind.
    """

    action: str = "continue"
    k: int = 0
    level: str = ""
    artifact: str = ""
    summary: str = ""
    stall: int = 0
    wealth: float = 1.0


@dataclass
class Trajectory:
    env: str
    task_id: str
    split: str
    task_text: str
    episode_id: str
    policy_id: str
    seed: int
    requested_model: str
    actual_model: str
    terminal_reason: str
    final_outcome: int
    native_score: float
    steps: list[StepRecord]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.env not in ENVIRONMENTS:
            raise ValueError(f"Unknown environment: {self.env}")
        if self.split not in SPLITS:
            raise ValueError(f"Unknown split: {self.split}")
        if self.final_outcome not in (0, 1):
            raise ValueError("final_outcome must be binary")
        if not self.steps:
            raise ValueError("trajectory must contain at least one step")
        expected = list(range(1, len(self.steps) + 1))
        actual = [step.step for step in self.steps]
        if actual != expected:
            raise ValueError(f"steps must be contiguous and 1-based: {actual}")

    @property
    def total_cost_usd(self) -> float:
        return float(sum(step.cost_usd for step in self.steps))

    @property
    def total_tokens(self) -> int:
        return sum(step.prompt_tokens + step.output_tokens for step in self.steps)

    @property
    def total_reasoning_tokens(self) -> int:
        return sum(step.reasoning_tokens for step in self.steps)

    @property
    def cache_accounting(self) -> str:
        statuses = {step.cache_accounting for step in self.steps}
        if statuses == {"exact"}:
            return "exact"
        if "upper_bound" in statuses:
            return "upper_bound"
        if "unavailable" in statuses:
            return "unavailable"
        return "unspecified"

    @property
    def cost_accounting(self) -> str:
        statuses = {step.cost_accounting for step in self.steps}
        if statuses == {"exact"}:
            return "exact"
        if "unavailable" in statuses:
            return "unavailable"
        if "upper_bound" in statuses:
            return "upper_bound"
        return "unspecified"

    def prefix_cost(self, step: int) -> float:
        return float(sum(item.cost_usd for item in self.steps[:step]))

    def prefix_tokens(self, step: int) -> int:
        return sum(item.prompt_tokens + item.output_tokens for item in self.steps[:step])

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["steps"] = [step.to_dict() for step in self.steps]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Trajectory":
        data = dict(value)
        data["steps"] = [StepRecord.from_dict(item) for item in data["steps"]]
        return cls(**data)


@dataclass(frozen=True)
class PolicyResult:
    env: str
    task_id: str
    split: str
    policy_id: str
    config_id: str
    y_b1: int
    y_policy: int
    cost_b1: float
    cost_policy: float
    halted: bool
    halt_step: int | None
    y_halt: int | None
    y_continue: int | None
    cost_b1_status: str = "unspecified"
    cost_policy_status: str = "unspecified"
    operational_aux_cost: float = 0.0
    status: str = "ok"
    note: str = ""

    def __post_init__(self) -> None:
        for name in ("y_b1", "y_policy"):
            if getattr(self, name) not in (0, 1):
                raise ValueError(f"{name} must be binary")
        for name in ("y_halt", "y_continue"):
            if getattr(self, name) not in (None, 0, 1):
                raise ValueError(f"{name} must be binary or null")
        if self.cost_b1 < 0 or self.cost_policy < 0 or self.operational_aux_cost < 0:
            raise ValueError("policy costs cannot be negative")
        if self.cost_b1_status not in ACCOUNTING_STATUSES:
            raise ValueError(f"invalid cost_b1_status: {self.cost_b1_status}")
        if self.cost_policy_status not in ACCOUNTING_STATUSES:
            raise ValueError(f"invalid cost_policy_status: {self.cost_policy_status}")
        if self.status not in {"ok", "blocked", "infra_error"}:
            raise ValueError(f"invalid policy result status: {self.status}")
        if self.halted:
            if self.halt_step is None or self.halt_step < 1 or self.y_halt is None:
                raise ValueError("halted result requires halt_step and y_halt")
            if self.split == "audit" and self.y_continue is None:
                raise ValueError("halted audit result requires y_continue")
        elif self.halt_step is not None:
            raise ValueError("non-halted result cannot have halt_step")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_trajectory_dict(value: dict[str, Any]) -> Trajectory:
    """Parse and validate normalized trajectory JSON."""
    return Trajectory.from_dict(value)


def as_step_record(telemetry: StepTelemetry | StepRecord | Mapping[str, Any], step: int) -> StepRecord:
    """Normalize a per-step telemetry input into a ``StepRecord``."""
    if isinstance(telemetry, StepRecord):
        return telemetry
    if isinstance(telemetry, StepTelemetry):
        return telemetry.to_step_record(step)
    if isinstance(telemetry, Mapping):
        data = dict(telemetry)
        data.pop("step", None)
        return StepRecord(step=step, **data)
    raise TypeError(f"unsupported telemetry type: {type(telemetry).__name__}")