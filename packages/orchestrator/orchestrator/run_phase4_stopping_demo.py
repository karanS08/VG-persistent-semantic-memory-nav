"""Phase 4 Step 4.3 verification: construct one episode expected to SUCCEED
(target verified mid-loop) and one expected to hit budget-FAILURE, confirm both
terminate with the correct outcome. The pure check_stopping() formula is already
unit-tested in packages/frontier_agent/tests/test_stopping.py -- this instead
proves it correctly drives a real step loop to termination, not just returns the
right value in isolation.
"""
from __future__ import annotations

from frontier_agent.stopping import check_stopping, EpisodeOutcome


def run_episode(max_steps: int, time_budget: float, verify_at_step: int | None) -> dict:
    elapsed_time = 0.0
    path_length = 0.0
    coverage_ratio = 0.0
    step_log = []

    for step in range(max_steps):
        elapsed_time += 1.0  # 1 "second" per step, synthetic but deterministic
        path_length += 0.5
        coverage_ratio = min(1.0, coverage_ratio + 0.05)
        target_verified = (verify_at_step is not None and step == verify_at_step)

        result = check_stopping(
            target_verified=target_verified,
            elapsed_time=elapsed_time, time_budget=time_budget,
            path_length=path_length, path_budget=1000.0,  # not the constraint under test here
            coverage_ratio=coverage_ratio, coverage_threshold=1000.0,  # not under test here
            best_guess_position=[path_length, 0.0, 0.0],
        )
        step_log.append({"step": step, "elapsed_time": elapsed_time, "outcome": result.outcome.value, "reason": result.reason})

        if result.outcome != EpisodeOutcome.CONTINUE:
            return {"final_outcome": result.outcome.value, "reason": result.reason,
                    "terminated_at_step": step, "best_guess_position": result.best_guess_position,
                    "step_log": step_log}

    return {"final_outcome": "never_terminated", "step_log": step_log}


def main() -> None:
    print("=== Episode A: expected to SUCCEED (target verified at step 3) ===")
    result_a = run_episode(max_steps=20, time_budget=100.0, verify_at_step=3)
    print(f"final_outcome={result_a['final_outcome']} reason={result_a['reason']} terminated_at_step={result_a['terminated_at_step']}")
    assert result_a["final_outcome"] == "success", "Episode A should have succeeded"
    assert result_a["terminated_at_step"] == 3, "Episode A should stop exactly when verified, not run to max_steps"

    print("\n=== Episode B: expected to hit budget-FAILURE (never verified, tiny time_budget) ===")
    result_b = run_episode(max_steps=20, time_budget=5.0, verify_at_step=None)
    print(f"final_outcome={result_b['final_outcome']} reason={result_b['reason']} terminated_at_step={result_b['terminated_at_step']}")
    print(f"best_guess_position logged as diagnostic (NOT a success): {result_b['best_guess_position']}")
    assert result_b["final_outcome"] == "failure", "Episode B should have failed"
    assert result_b["reason"] == "time_budget_exhausted"

    print("\nBoth episodes terminated with the correct success/failure flag.")


if __name__ == "__main__":
    main()
