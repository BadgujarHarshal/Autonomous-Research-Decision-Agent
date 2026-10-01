# app/tools/registry.py

from __future__ import annotations

from typing import Dict, List

from app.tools.base import BaseTool


class ToolRegistry:
    """Central registry of available agent tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError("Only BaseTool instances can be registered.")

        name = tool.name.strip()

        if not name:
            raise ValueError("Tool name cannot be empty.")

        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")

        self._tools[name] = tool

    def unregister(self, name: str) -> None:
        name = name.strip()

        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")

        del self._tools[name]

    def get(self, name: str) -> BaseTool:
        name = name.strip()

        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")

        return self._tools[name]

    def has(self, name: str) -> bool:
        return name.strip() in self._tools

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def describe_tools(self) -> List[dict]:
        return [
            self._tools[name].metadata()
            for name in sorted(self._tools)
        ]

    def __len__(self) -> int:
        return len(self._tools)