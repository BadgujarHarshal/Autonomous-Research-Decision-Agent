#  tests/agent/test_schemas.py

import pytest
from pydantic import ValidationError

from app.agent.schemas import (
    AgentDecision,
    AgentPlan,
    AgentTask,
    ToolCall,
    ToolResult,
)


def test_agent_task_creation():
    task = AgentTask(
        task_id="task-001",
        goal="Research the causes of a market event.",
    )

    assert task.task_id == "task-001"
    assert task.goal == "Research the causes of a market event."


def test_empty_task_id_is_rejected():
    with pytest.raises(ValidationError):
        AgentTask(
            task_id="",
            goal="Research something.",
        )


def test_tool_call_creation():
    call = ToolCall(
        tool_name="calculate",
        arguments={"expression": "10 + 20"},
    )

    assert call.tool_name == "calculate"
    assert call.arguments["expression"] == "10 + 20"


def test_tool_result_success():
    result = ToolResult(
        tool_name="calculate",
        success=True,
        output=30,
        latency_ms=2.5,
    )

    assert result.success is True
    assert result.output == 30
    assert result.latency_ms == 2.5


def test_plan_creation():
    plan = AgentPlan(
        goal="Calculate a value.",
        steps=[],
    )

    assert plan.goal == "Calculate a value."
    assert plan.steps == []


def test_decision_confidence_range():
    with pytest.raises(ValidationError):
        AgentDecision(
            action="execute",
            confidence=1.5,
        )
