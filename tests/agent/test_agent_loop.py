# tests/agent/test_agent_loop.py

from app.agent.agent_loop import AgentLoop
from app.agent.controller import AgentController
from app.agent.planner import BaselinePlanner
from app.agent.schemas import AgentStatus, AgentTask
from app.agent.state import AgentState
from app.tools.calculator import SafeCalculatorTool
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


def create_agent():
    registry = ToolRegistry()
    registry.register(SafeCalculatorTool())

    executor = ToolExecutor(registry)

    controller = AgentController(
        registry=registry,
        max_steps=8,
    )

    return AgentLoop(
        controller=controller,
        executor=executor,
        max_retries_per_step=2,
    )


def create_state(goal: str):
    task = AgentTask(
        task_id="loop-test-001",
        goal=goal,
    )

    planner = BaselinePlanner()
    plan = planner.create_plan(task)

    state = AgentState(task)
    state.set_plan(plan)

    return state


def test_agent_loop_completes_calculation():
    agent = create_agent()
    state = create_state(
        "Calculate 25 * 4"
    )

    result = agent.run(state)

    assert result.status == AgentStatus.COMPLETED
    assert result.metadata["final_answer"] == "100"

    assert len(result.tool_calls) == 1
    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is True
    assert result.tool_results[0].output == 100


def test_agent_loop_records_observation():
    agent = create_agent()
    state = create_state(
        "Calculate 50 + 25"
    )

    result = agent.run(state)

    assert len(result.observations) == 1
    assert result.observations[0].source == "calculate"
    assert result.observations[0].content == 75
    assert result.observations[0].success is True


def test_agent_loop_records_decision():
    agent = create_agent()
    state = create_state(
        "Calculate 10 * 10"
    )

    result = agent.run(state)

    assert len(result.decisions) >= 1
    assert result.decisions[0].action == "execute"


def test_agent_loop_unknown_task():
    agent = create_agent()
    state = create_state(
        "Perform an unsupported task."
    )

    result = agent.run(state)

    assert result.status == AgentStatus.COMPLETED
    assert (
        "could not determine"
        in result.metadata["final_answer"].lower()
    )


def test_agent_loop_requires_plan():
    agent = create_agent()

    task = AgentTask(
        task_id="no-plan",
        goal="Calculate 10 + 20",
    )

    state = AgentState(task)

    try:
        agent.run(state)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "plan" in str(exc).lower()


def test_agent_loop_calculation_with_parentheses():
    agent = create_agent()
    state = create_state(
        "Calculate (10 + 5) * 2"
    )

    result = agent.run(state)

    assert result.status == AgentStatus.COMPLETED
    assert result.metadata["final_answer"] == "30"