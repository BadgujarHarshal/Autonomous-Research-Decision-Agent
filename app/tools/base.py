# app/tools/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """
    Base contract for every agent tool.

    Every tool has:
    - a unique name
    - a description for the planner/model
    - an argument schema
    - an execution method
    """

    name: str = ""
    description: str = ""

    @property
    @abstractmethod
    def argument_schema(self) -> Dict[str, Any]:
        """Return the tool's expected argument schema."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        Execute the tool.

        Tool implementations should return serializable Python objects.
        ToolExecutor is responsible for wrapping the result in ToolResult.
        """
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        """Return metadata exposed to the agent planner."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.argument_schema,
        }

    def validate_arguments(self, arguments: Dict[str, Any]) -> None:
        """Basic argument validation hook."""
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a dictionary.")