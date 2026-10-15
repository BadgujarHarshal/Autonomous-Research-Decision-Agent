# tests/agent/test_state.py

from app.agent.schemas import (
    AgentStatus,
    AgentTask,
    Observation,
    ToolCall,
    ToolResult,
)
from app.agent.state import AgentState


def create_state() -> AgentState:
    task = AgentTask(
        task_id="task-001",
        goal="Calculate 10 + 20.",
    )

    return AgentState(task)


def test_initial_state():
    state = create_state()

    assert state.status == AgentStatus.CREATED
    assert state.plan is None
    assert state.current_step_index == 0
    assert state.retry_count == 0
    assert state.replan_count == 0


def test_status_update():
    state = create_state()

    state.set_status(AgentStatus.PLANNING)

    assert state.status == AgentStatus.PLANNING


def test_tool_call_is_recorded():
    state = create_state()

    call = ToolCall(
        tool_name="calculate",
        arguments={"expression": "10 + 20"},
    )

    state.add_tool_call(call)

    assert len(state.tool_calls) == 1
    assert state.tool_calls[0].tool_name == "calculate"


def test_tool_result_is_recorded():
    state = create_state()

    result = ToolResult(
        tool_name="calculate",
        success=True,
        output=30,
    )

    state.add_tool_result(result)

    assert len(state.tool_results) == 1
    assert state.tool_results[0].output == 30


def test_observation_is_recorded():
    state = create_state()

    observation = Observation(
        step_id=1,
        source="calculate",
        content=30,
        success=True,
    )

    state.add_observation(observation)

    assert len(state.observations) == 1
    assert state.observations[0].content == 30


def test_retry_and_replan_counters():
    state = create_state()

    state.increment_retry()
    state.increment_retry()
    state.increment_replan()

    assert state.retry_count == 2
    assert state.replan_count == 1


def test_snapshot():
    state = create_state()

    snapshot = state.snapshot()

    assert snapshot["task_id"] == "task-001"
    assert snapshot["goal"] == "Calculate 10 + 20."
    assert snapshot["status"] == "created"
    assert snapshot["tool_calls"] == 0