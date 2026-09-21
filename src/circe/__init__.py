"""CIRCE: calibrated interruption via risk-controlled e-processes.

A pluggable, harness-agnostic runtime controller for black-box agent harnesses.
The controller observes the harness's own per-step telemetry and returns an
intervention decision (continue / halt / compact / rewind / escalate) with
anytime-valid false-kill control.
"""

from .config import ControllerConfig, freeze_config, load_config
from .controller import CirceController
from .eprocess import BettingEProcess
from .calibration import (
    SuccessModel,
    bernoulli_kl,
    calibrate_q0,
    calibrate_q1,
    train_success_model,
)
from .features import (
    FEATURE_NAMES,
    extract_feature_dict,
    extract_feature_vector,
    features_from_steps,
    is_stall,
)
from .interventions import (
    COMPACT,
    CONTINUE,
    ESCALATE,
    HALT,
    NON_HALT_ACTIONS,
    REWIND,
    InterventionModels,
    optimistic_index,
    threshold_based_models,
)
from .policies import (
    CirceFullPolicy,
    CirceHaltPolicy,
    build_policy,
    evaluate_halt_only,
)
from .selection import clopper_pearson_upper, ltt_select, summarize_selection
from .self_calibration import (
    FiredEpisode,
    SelfCalibratingMonitor,
    doubly_robust_salvage_loss,
    dr_counterfactual,
    empirical_bernstein_radius,
    reprieve_schedule,
)
from .schema import (
    Decision,
    PolicyResult,
    StepRecord,
    StepTelemetry,
    TaskSpec,
    Trajectory,
)

__version__ = "0.1.0"

__all__ = [
    "CirceController",
    "ControllerConfig",
    "load_config",
    "freeze_config",
    "BettingEProcess",
    "Decision",
    "StepTelemetry",
    "StepRecord",
    "TaskSpec",
    "Trajectory",
    "PolicyResult",
    "FEATURE_NAMES",
    "extract_feature_dict",
    "extract_feature_vector",
    "features_from_steps",
    "is_stall",
    "InterventionModels",
    "optimistic_index",
    "threshold_based_models",
    "COMPACT",
    "CONTINUE",
    "REWIND",
    "ESCALATE",
    "HALT",
    "NON_HALT_ACTIONS",
    "calibrate_q0",
    "calibrate_q1",
    "bernoulli_kl",
    "SuccessModel",
    "train_success_model",
    "CirceHaltPolicy",
    "CirceFullPolicy",
    "build_policy",
    "evaluate_halt_only",
    "clopper_pearson_upper",
    "summarize_selection",
    "ltt_select",
    "FiredEpisode",
    "SelfCalibratingMonitor",
    "doubly_robust_salvage_loss",
    "dr_counterfactual",
    "empirical_bernstein_radius",
    "reprieve_schedule",
    "__version__",
]
