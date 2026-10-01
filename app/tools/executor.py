# app/tools/executor.py

from __future__ import annotations

import time
from typing import Any, Dict

from app.agent.schemas import ToolCall, ToolResult
from app.tools.registry import ToolRegistry


class ToolExecutor:
    """
    Controlled execution gateway for registered tools.

    Phase 4 guarantees that:
    - unknown tools become structured failures
    - invalid arguments become structured failures
    - tool exceptions become structured failures
    - execution latency is always recorded
    - the agent loop never receives an unhandled tool exception
    """

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:
        self.registry = registry

    def execute(
        self,
        tool_call: ToolCall,
    ) -> ToolResult:
        start = time.perf_counter()

        if not isinstance(
            tool_call,
            ToolCall,
        ):
            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            return ToolResult(
                tool_name="unknown",
                success=False,
                output=None,
                error=(
                    "Executor received an invalid "
                    "ToolCall object."
                ),
                latency_ms=latency_ms,
            )

        tool_name = (
            tool_call.tool_name.strip()
        )

        if not tool_name:
            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            return ToolResult(
                tool_name="unknown",
                success=False,
                output=None,
                error="Tool name cannot be empty.",
                latency_ms=latency_ms,
            )

        if not self.registry.has(
            tool_name
        ):
            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=(
                    f"Unknown tool: {tool_name}"
                ),
                latency_ms=latency_ms,
            )

        try:
            tool = self.registry.get(
                tool_name
            )
        except Exception as exc:
            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=(
                    f"Tool lookup failed: {exc}"
                ),
                latency_ms=latency_ms,
            )

        try:
            arguments = tool_call.arguments

            if not isinstance(
                arguments,
                dict,
            ):
                raise TypeError(
                    "Tool arguments must be a dictionary."
                )

            tool.validate_arguments(
                arguments
            )

            output = tool.execute(
                arguments
            )

            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output,
                error=None,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                latency_ms=latency_ms,
            )

    def execute_raw(
        self,
        tool_name: str,
        arguments: Dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Convenience method for direct execution.
        """

        try:
            call = ToolCall(
                tool_name=tool_name,
                arguments=arguments or {},
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name or "unknown",
                success=False,
                output=None,
                error=(
                    f"Invalid tool call: {exc}"
                ),
                latency_ms=0.0,
            )

        return self.execute(
            call
        )