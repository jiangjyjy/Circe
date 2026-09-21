# Circe

**Calibrated Interruption via Risk-Controlled E-processes for Black-Box Agent Harnesses** — the core controller library.

CIRCE is a harness-level controller that decides, at every step of an agent's
execution loop, whether to let the agent continue or to intervene. It is:

- **black-box** — it reads only the harness's own logs (never model internals, logits, or gradients), so it works with any API-served policy;
- **zero marginal inference cost** — every feature it consumes is a by-product of the harness; it issues no extra model calls;
- **self-calibrating** — randomized reprieve plus a doubly-robust salvage-loss monitor keeps its false-kill guarantee valid *after* it starts intervening, despite the outcome censoring its own decisions induce.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.11.

## Quickstart (online, pluggable)

The harness owns the agent loop; CIRCE is a sidecar consulted once per step.

```python
from circe import CirceController, ControllerConfig

config = ControllerConfig(alpha=0.05, q0=0.1, patience=1)
controller = CirceController.from_config(config, task_id="task-1")
controller.start(task_text="...", task_id="task-1")

for step in harness.run():
    d = controller.step(harness.telemetry())   # pure decision, no I/O, no model call
    if d.action == "halt":
        harness.stop(best_artifact=d.artifact)
    elif d.action == "rewind":
        harness.rewind(k=d.k, failure_summary=d.summary)
    elif d.action == "compact":
        harness.compact()
    elif d.action == "escalate":
        harness.set_reasoning_effort(d.level)
```

The harness supplies one `StepTelemetry` (or a plain dict) per step — the
same fields it already logs. See `examples/simple_harness.py` for a runnable
self-contained demo.

The full controller (`mode="full"`) selects among CONT/COMPACT/REWIND/ESCALATE
with the cost-normalized optimistic index `I_u = (Δ̂_u + β·σ_u)/ĉ_u` (§1.7);
HALT remains admissible only after the e-process certificate fires.

## Online self-calibration (§1.6)

A `SelfCalibratingMonitor` tracks every fired episode, estimates the salvage
loss with the doubly-robust estimator, wraps it in a time-uniform
empirical-Bernstein confidence sequence, and flags distribution drift when the
lower confidence bound rises above `alpha`:

```python
from circe import FiredEpisode, SelfCalibratingMonitor

monitor = SelfCalibratingMonitor(alpha=0.05, delta=0.05)
monitor.begin_round(n_episodes=128)

# when the controller fires on episode i with fitted success prob m_hat:
monitor.record_fire(FiredEpisode(
    episode_id="ep-1", round=1, m_hat=0.32,
    reprieved=True, propensity=0.05,      # logged propensity ε_i
    outcome_cont=1, outcome_halt=0,       # Y^{κ_∅} observed only because reprieved
))

est = monitor.estimate()        # {salvage_loss, radius, lower, upper, ...}
if monitor.drift_detected():
    reselect_configuration()     # re-run ltt_select on the new reprieve corpus
```

`reprieve_schedule(t, n_configs, mean_reprieve_cost)` returns the regret-optimal
rate `ε_t = min{1, (log|Λ| / (t·c̄))^{1/3}}` (Theorem 8).

## Harness contract

A harness integrates by implementing a tiny capability surface. CIRCE's full
mode requires the harness to support a few primitives; without them it degrades
gracefully to halt-only:


| Action      | Harness must                                                                               |
| ----------- | ------------------------------------------------------------------------------------------ |
| `halt`      | stop and return the best partial artifact                                                  |
| `rewind(k)` | restore workspace/context to a checkpoint`k` steps back and re-seed with a failure summary |
| `compact`   | compress the context                                                                       |
| `escalate`  | raise provider-side reasoning effort for subsequent calls                                  |

Per-step telemetry is a `StepTelemetry` (or dict) of the by-product fields:
`assistant_text`, `observation_text`, `tool_name`, `action_signature`,
`tool_status`, `error_signature`, `timed_out`, `noop`, `tests_passed`,
`tests_failed`, `compile_ok`, `files_touched`, `bytes_added`, `bytes_deleted`,
`novel_bytes`, `reverted_bytes`, `workspace_hash`, `workspace_tree` (an optional
`path -> content-hash` snapshot used by the workspace tree-edit-distance
feature), `state_hash`, token/cost fields.

## Configuration

A frozen config is a small mapping:

```yaml
alpha: 0.05            # anytime-valid false-kill level
delta: 0.05            # q0 calibration coverage
q0: 0.1                # upper bound on salvageable stall rate
patience: 1            # consecutive certificates before halting
reprieve_rate: 0.05    # exploration probability for online reprieve
mode: halt             # "halt" or "full"
stall:
  max_novel_bytes_per_token: 1.0   # novel workspace bytes per output token
  max_test_pass_delta: 0
  max_observation_novelty: 0.7
eprocess:
  prior_successes: 1.0
  prior_failures: 1.0
circe_full:            # only used when mode == "full"
  compact_context_threshold: 0.98
  rewind_error_streak: 3
  escalate_value_threshold: 0.5
  max_rewinds: 2
  max_compacts: 2
  rewind_depth: 1
```

Load with `ControllerConfig` defaults for quick experiments, or write/read a
frozen file with `freeze_config` / `load_config`. Loading merges nested keys
with defaults, so older configs that omit newer keys (e.g. `max_compacts`,
`rewind_depth`) keep working.

In `full` mode each COMPACT/REWIND resets the wealth process and reopens a halt
opportunity, so the episode-level false-kill budget `alpha` is split uniformly
across the `1 + max_rewinds + max_compacts` segments (`segment_alpha` in the
controller's `snapshot()`); `halt` mode uses the undivided `alpha`.

## Layout

```
src/circe/
  controller.py       online CirceController.step()
  eprocess.py         BettingEProcess (doom certificate, Theorem 5)
  features.py         34-feature telemetry + is_stall
  calibration.py      q0/q1 calibration, value model V̂
  selection.py        clopper-pearson, LTT fixed-sequence selection (Theorem 7)
  self_calibration.py doubly-robust salvage-loss estimator + confidence sequence + drift detection (§1.6)
  interventions.py    optimistic index I_u = (Δ̂_u + β·σ_u)/ĉ_u (§1.7)
  policies.py         CIRCE-HALT / CIRCE-FULL and offline evaluation
  schema.py           Trajectory / StepRecord / StepTelemetry / Decision / FiredEpisode types
  config.py           ControllerConfig
```

## License

MIT.
