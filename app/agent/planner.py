# app/agent/planner.py

from __future__ import annotations

from app.agent.schemas import AgentPlan, AgentTask, PlanStep


class BaselinePlanner:
    """
    Deterministic Phase-1 planner.

    This is intentionally not an LLM planner.

    The purpose of Phase 1 is to establish a reliable planning contract.
    The model-driven planner will be introduced in later phases.
    """

    def create_plan(self, task: AgentTask) -> AgentPlan:
        goal = task.goal.strip()

        step = PlanStep(
            step_id=1,
            description=f"Analyze the task and determine the required action: {goal}",
            expected_tool=None,
        )

        return AgentPlan(
            goal=goal,
            steps=[step],
        )