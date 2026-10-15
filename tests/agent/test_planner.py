# tests/agent/test_planner.py

from app.agent.planner import BaselinePlanner
from app.agent.schemas import AgentTask


def test_baseline_planner_creates_plan():
    planner = BaselinePlanner()

    task = AgentTask(
        task_id="task-001",
        goal="Calculate 100 / 4.",
    )

    plan = planner.create_plan(task)

    assert plan.goal == "Calculate 100 / 4."
    assert len(plan.steps) == 1
    assert plan.steps[0].step_id == 1


def test_baseline_planner_preserves_goal():
    planner = BaselinePlanner()

    task = AgentTask(
        task_id="task-002",
        goal="Find information about a company.",
    )

    plan = planner.create_plan(task)

    assert plan.goal == task.goal