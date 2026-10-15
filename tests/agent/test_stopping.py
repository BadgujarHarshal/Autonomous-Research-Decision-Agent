# tests/agent/test_stopping.py

from app.agent.controller import AgentController
from app.agent.planner import BaselinePlanner
from app.agent.schemas import (
    AgentStatus,
    AgentTask,
    ToolCall,
)
from app.agent.state import AgentState
from app.tools.calculator import SafeCalculatorTool
from app.tools.registry import ToolRegistry


def create_controller(
    max_steps=8,
    max_identical_tool_calls=2,
):
    registry = ToolRegistry()

    registry.register(
        SafeCalculatorTool()
    )

    return AgentController(
        registry=registry,
        max_steps=max_steps,
        max_identical_tool_calls=(
            max_identical_tool_calls
        ),
    )


def create_state(goal):
    task = AgentTask(
        task_id="stopping-test",
        goal=goal,
    )

    planner = BaselinePlanner()

    plan = planner.create_plan(task)

    state = AgentState(task)
    state.set_plan(plan)

    return state


def test_controller_stops_when_max_steps_reached():
    controller = create_controller(
        max_steps=1
    )

    state = create_state(
        "Calculate 10 + 20"
    )

    state.add_tool_call(
        ToolCall(
            tool_name="calculate",
            arguments={
                "expression": "10 + 20"
            },
        )
    )

    decision = controller.decide(
        state
    )

    assert decision.action == "finish"
    assert (
        "Maximum agent step limit"
        in decision.reasoning_summary
    )


def test_controller_stops_on_terminal_state():
    controller = create_controller()

    state = create_state(
        "Calculate 10 + 20"
    )

    state.set_status(
        AgentStatus.FAILED
    )

    decision = controller.decide(
        state
    )

    assert decision.action == "finish"
    assert decision.confidence == 1.0


def test_controller_detects_repeated_tool_calls():
    controller = create_controller(
        max_identical_tool_calls=2
    )

    state = create_state(
        "Calculate 10 + 20"
    )

    call = ToolCall(
        tool_name="calculate",
        arguments={
            "expression": "10 + 20"
        },
    )

    state.add_tool_call(call)
    state.add_tool_call(call)

    decision = controller.decide(
        state
    )

    assert decision.action == "finish"
    assert (
        "Repeated identical tool call"
        in decision.reasoning_summary
    )


def test_controller_allows_first_tool_call():
    controller = create_controller()

    state = create_state(
        "Calculate 10 + 20"
    )

    decision = controller.decide(
        state
    )

    assert decision.action == "execute"
    assert decision.tool_call is not None


def test_replan_limit_is_respected():
    controller = create_controller()

    state = create_state(
        "Calculate 10 + 20"
    )

    assert controller.can_replan(
        state,
        max_replans=1,
    )

    state.increment_replan()

    assert not controller.can_replan(
        state,
        max_replans=1,
    )