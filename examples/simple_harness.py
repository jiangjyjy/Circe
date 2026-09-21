"""Minimal example: plug CIRCE into a fake harness.

A real harness would own the agent loop and the model calls. Here we simulate
a handful of steps and show how the controller is consulted once per step.
"""
from circe import CirceController, ControllerConfig, StepTelemetry

# A controller calibrated on historical trajectories. q0 is the upper bound on
# the stall rate of salvageable runs; alpha is the anytime-valid false-kill level.
config = ControllerConfig(
    alpha=0.05,
    q0=0.1,
    patience=1,
    reprieve_rate=0.0,
)
controller = CirceController.from_config(config, task_id="demo-task")
controller.start(task_text="Move final_report.pdf into temp/", task_id="demo-task")


def make_step(i: int, novel_bytes: int, output: str) -> StepTelemetry:
    return StepTelemetry(
        tool_name="bash",
        action_signature=f"cmd-{i}",
        novel_bytes=novel_bytes,
        observation_text=output,
    )


# Simulated trajectory: productive work, then the agent loops with no novelty.
# The stall indicator fires only once the rolling 4-step window contains no
# new workspace bytes, so the tail is made long enough to visibly halt.
steps = [make_step(1, 300, "created temp/"), make_step(2, 150, "moved final_report.pdf")]
steps += [make_step(i, 0, "noop") for i in range(3, 14)]

for step_number, tel in enumerate(steps, 1):
    decision = controller.step(tel)
    print(
        f"step {step_number:2d}  novel={tel.novel_bytes:3d}  wealth={decision.wealth:7.2f}  "
        f"stall={decision.stall}  -> {decision.action}"
    )
    if decision.action == "halt":
        print("  harness stops; best partial artifact:", decision.artifact or "<none>")
        break