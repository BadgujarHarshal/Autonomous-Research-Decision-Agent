# tests/tools/test_registry.py

import pytest

from app.tools.base import BaseTool
from app.tools.calculator import SafeCalculatorTool
from app.tools.registry import ToolRegistry


class DummyTool(BaseTool):
    name = "dummy"
    description = "A test tool."

    @property
    def argument_schema(self):
        return {
            "type": "object",
            "properties": {},
        }

    def execute(self, arguments):
        return {"ok": True}


def test_register_and_get_tool():
    registry = ToolRegistry()
    tool = DummyTool()

    registry.register(tool)

    assert registry.has("dummy")
    assert registry.get("dummy") is tool


def test_duplicate_registration_is_rejected():
    registry = ToolRegistry()

    registry.register(DummyTool())

    with pytest.raises(ValueError):
        registry.register(DummyTool())


def test_unknown_tool_is_rejected():
    registry = ToolRegistry()

    with pytest.raises(KeyError):
        registry.get("missing")


def test_unregister_tool():
    registry = ToolRegistry()

    registry.register(DummyTool())
    registry.unregister("dummy")

    assert not registry.has("dummy")


def test_list_tools_is_sorted():
    registry = ToolRegistry()

    class ZTool(DummyTool):
        name = "z_tool"

    class ATool(DummyTool):
        name = "a_tool"

    registry.register(ZTool())
    registry.register(ATool())

    assert registry.list_tools() == [
        "a_tool",
        "z_tool",
    ]


def test_describe_tools():
    registry = ToolRegistry()

    registry.register(SafeCalculatorTool())

    descriptions = registry.describe_tools()

    assert len(descriptions) == 1
    assert descriptions[0]["name"] == "calculate"
    assert "arguments" in descriptions[0]