# tests/tools/test_executor.py

from app.agent.schemas import ToolCall
from app.tools.calculator import SafeCalculatorTool
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


def create_executor():
    registry = ToolRegistry()
    registry.register(SafeCalculatorTool())

    return ToolExecutor(registry)


def test_executor_success():
    executor = create_executor()

    result = executor.execute(
        ToolCall(
            tool_name="calculate",
            arguments={
                "expression": "10 + 20",
            },
        )
    )

    assert result.success is True
    assert result.output == 30
    assert result.error is None
    assert result.latency_ms >= 0


def test_executor_unknown_tool():
    executor = create_executor()

    result = executor.execute(
        ToolCall(
            tool_name="unknown_tool",
            arguments={},
        )
    )

    assert result.success is False
    assert result.output is None
    assert "Unknown tool" in result.error


def test_executor_tool_error():
    executor = create_executor()

    result = executor.execute(
        ToolCall(
            tool_name="calculate",
            arguments={
                "expression": "1 / 0",
            },
        )
    )

    assert result.success is False
    assert result.error is not None


def test_execute_raw():
    executor = create_executor()

    result = executor.execute_raw(
        "calculate",
        {
            "expression": "5 * 6",
        },
    )

    assert result.success is True
    assert result.output == 30