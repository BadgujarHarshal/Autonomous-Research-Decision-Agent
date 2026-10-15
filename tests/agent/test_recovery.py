#  tests/agent/test_recovery.py

from app.agent.agent_loop import AgentLoop
from app.agent.controller import AgentController
from app.agent.planner import BaselinePlanner
from app.agent.schemas import (
    AgentStatus,
    AgentTask,
    ToolCall,
)
from app.agent.state import AgentState
from app.tools.base import BaseTool
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class AlwaysFailTool(BaseTool):
    name = "always_fail"
    description = "A tool that always fails for recovery testing."

    @property
    def argument_schema(self):
        return {
            "type": "object",
            "properties": {},
        }

    def execute(self, arguments):
        raise RuntimeError(
            "Intentional test failure."
        )


class FlakyTool(BaseTool):
    name = "flaky_tool"
    description = "Fails once and succeeds on retry."

    def __init__(self):
        self.calls = 0

    @property
    def argument_schema(self):
        return {
            "type": "object",
            "properties": {},
        }

    def execute(self, arguments):
        self.calls += 1

        if self.calls == 1:
            raise RuntimeError(
                "Temporary failure."
            )

        return "recovered"


def create_state(
    tool_name: str,
    arguments=None,
):
    task = AgentTask(
        task_id="recovery-test",
        goal="Test recovery.",
    )

    planner = BaselinePlanner()

    plan = planner.create_plan(task)

    plan.steps[0].expected_tool = tool_name
    plan.steps[0].arguments = arguments or {}

    state = AgentState(task)
    state.set_plan(plan)

    return state


def test_retry_recovers_from_temporary_failure():
    registry = ToolRegistry()

    flaky = FlakyTool()
    registry.register(flaky)

    executor = ToolExecutor(
        registry
    )

    controller = AgentController(
        registry=registry,
        max_steps=5,
    )

    agent = AgentLoop(
        controller=controller,
        executor=executor,
        max_retries_per_step=2,
        max_replans=0,
    )

    state = create_state(
        "flaky_tool"
    )

    result = agent.run(state)

    assert result.status == AgentStatus.COMPLETED
    assert result.metadata["final_answer"] == "recovered"

    assert flaky.calls == 2
    assert len(result.tool_results) == 2
    assert result.tool_results[0].success is False
    assert result.tool_results[1].success is True


def test_failure_after_retry_limit_becomes_failed():
    registry = ToolRegistry()

    registry.register(
        AlwaysFailTool()
    )

    executor = ToolExecutor(
        registry
    )

    controller = AgentController(
        registry=registry,
        max_steps=5,
    )

    agent = AgentLoop(
        controller=controller,
        executor=executor,
        max_retries_per_step=1,
        max_replans=0,
    )

    state = create_state(
        "always_fail"
    )

    result = agent.run(state)

    assert result.status == AgentStatus.FAILED
    assert len(result.tool_results) == 2
    assert result.tool_results[0].success is False
    assert result.tool_results[1].success is False


def test_failure_triggers_replan_when_available():
    registry = ToolRegistry()

    registry.register(
        AlwaysFailTool()
    )

    executor = ToolExecutor(
        registry
    )

    controller = AgentController(
        registry=registry,
        max_steps=10,
    )

    agent = AgentLoop(
        controller=controller,
        executor=executor,
        max_retries_per_step=0,
        max_replans=2,
    )

    state = create_state(
        "always_fail"
    )

    result = agent.run(state)

    assert result.status == AgentStatus.FAILED
    assert result.replan_count == 2
    assert result.metadata["replan_count"] == 2


def test_recovery_records_errors():
    registry = ToolRegistry()

    registry.register(
        AlwaysFailTool()
    )

    executor = ToolExecutor(
        registry
    )

    controller = AgentController(
        registry=registry
    )

    agent = AgentLoop(
        controller=controller,
        executor=executor,
        max_retries_per_step=0,
        max_replans=0,
    )

    state = create_state(
        "always_fail"
    )

    result = agent.run(state)

    assert result.status == AgentStatus.FAILED
    assert len(result.errors) >= 1
    assert "Intentional test failure" in (
        result.errors[0]
    )