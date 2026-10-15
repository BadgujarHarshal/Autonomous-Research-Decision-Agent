# tests/agent/test_controller.py

from app.agent.controller import AgentController
from app.agent.planner import BaselinePlanner
from app.agent.schemas import AgentStatus, AgentTask
from app.agent.state import AgentState
from app.tools.calculator import SafeCalculatorTool
from app.tools.registry import ToolRegistry


def create_controller():
    registry = ToolRegistry()
    registry.register(SafeCalculatorTool())

    return AgentController(
        registry=registry,
        max_steps=8,
    )


def create_state(goal: str):
    task = AgentTask(
        task_id="test-001",
        goal=goal,
    )

    planner = BaselinePlanner()
    plan = planner.create_plan(task)

    state = AgentState(task)
    state.set_plan(plan)

    return state


def test_controller_selects_calculator():
    controller = create_controller()
    state = create_state(
        "Calculate 25 * 4"
    )

    decision = controller.decide(state)

    assert decision.action == "execute"
    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "calculate"
    assert decision.tool_call.arguments["expression"] == "25 * 4"


def test_controller_extracts_math_expression():
    controller = create_controller()
    state = create_state(
        "What is 100 / 5?"
    )

    decision = controller.decide(state)

    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "calculate"
    assert decision.tool_call.arguments["expression"] == "100 / 5"


def test_controller_handles_unknown_goal():
    controller = create_controller()
    state = create_state(
        "Perform an unsupported task."
    )

    decision = controller.decide(state)

    assert decision.action == "finish"
    assert decision.tool_call is None


def test_controller_respects_terminal_state():
    controller = create_controller()
    state = create_state(
        "Calculate 10 + 5"
    )

    state.set_status(AgentStatus.COMPLETED)

    decision = controller.decide(state)

    assert decision.action == "finish"
    assert decision.confidence == 1.0


def test_controller_respects_max_steps():
    controller = create_controller()
    controller.max_steps = 0

    state = create_state(
        "Calculate 10 + 5"
    )

    decision = controller.decide(state)

    assert decision.action == "finish"