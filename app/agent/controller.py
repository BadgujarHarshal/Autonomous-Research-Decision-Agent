# app/agent/controller.py

from __future__ import annotations

from collections import Counter
from typing import Optional

from app.agent.schemas import (
    AgentDecision,
    AgentStatus,
    ToolCall,
)
from app.agent.state import AgentState
from app.tools.registry import ToolRegistry


class AgentController:
    """
    Baseline decision controller for the autonomous agent.

    Phase 4 responsibilities:
    - select tools
    - enforce confidence thresholds
    - detect repeated tool-call loops
    - detect terminal states
    - enforce maximum execution steps
    - produce final answers
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_steps: int = 8,
        minimum_tool_confidence: float = 0.60,
        minimum_final_confidence: float = 0.70,
        max_identical_tool_calls: int = 2,
    ) -> None:
        if max_steps < 1:
            raise ValueError(
                "max_steps must be at least 1."
            )

        if not 0.0 <= minimum_tool_confidence <= 1.0:
            raise ValueError(
                "minimum_tool_confidence must be between 0 and 1."
            )

        if not 0.0 <= minimum_final_confidence <= 1.0:
            raise ValueError(
                "minimum_final_confidence must be between 0 and 1."
            )

        if max_identical_tool_calls < 1:
            raise ValueError(
                "max_identical_tool_calls must be at least 1."
            )

        self.registry = registry
        self.max_steps = max_steps
        self.minimum_tool_confidence = (
            minimum_tool_confidence
        )
        self.minimum_final_confidence = (
            minimum_final_confidence
        )
        self.max_identical_tool_calls = (
            max_identical_tool_calls
        )

    def decide(
        self,
        state: AgentState,
    ) -> AgentDecision:
        """
        Determine the next action.

        Actions:
        - execute
        - finish
        """

        if state.status in {
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.ABORTED,
        }:
            return AgentDecision(
                action="finish",
                reasoning_summary=(
                    "Agent is already in a terminal state."
                ),
                confidence=1.0,
                final_answer=self._build_final_answer(
                    state
                ),
            )

        if len(state.tool_calls) >= self.max_steps:
            return self._stop_decision(
                state,
                "Maximum agent step limit reached.",
            )

        current_step = state.current_step()

        if current_step is None:
            return self._stop_decision(
                state,
                "No remaining planned steps.",
            )

        # Respect an explicitly planned tool.
        if current_step.expected_tool:
            tool_name = current_step.expected_tool

            if not self.registry.has(tool_name):
                return self._stop_decision(
                    state,
                    (
                        f"Planned tool '{tool_name}' "
                        "is not registered."
                    ),
                )

            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=current_step.arguments,
            )

            if self._is_repeated_tool_call(
                state,
                tool_call,
            ):
                return self._stop_decision(
                    state,
                    (
                        "Repeated identical tool call detected. "
                        "Stopping to prevent an execution loop."
                    ),
                )

            return AgentDecision(
                action="execute",
                reasoning_summary=(
                    f"Executing planned tool '{tool_name}'."
                ),
                tool_call=tool_call,
                confidence=0.95,
            )

        # Baseline autonomous tool selection.
        tool_call = self._select_tool_from_goal(
            state.task.goal
        )

        if tool_call is None:
            return self._stop_decision(
                state,
                "No suitable baseline tool could be selected.",
                confidence=0.0,
                final_answer=(
                    "I could not determine a suitable "
                    "tool for the requested task."
                ),
            )

        if self._is_repeated_tool_call(
            state,
            tool_call,
        ):
            return self._stop_decision(
                state,
                (
                    "Repeated identical tool call detected. "
                    "Stopping to prevent an execution loop."
                ),
            )

        confidence = 0.75

        if confidence < self.minimum_tool_confidence:
            return self._stop_decision(
                state,
                (
                    "Tool-selection confidence is below "
                    "the configured threshold."
                ),
                confidence=confidence,
            )

        return AgentDecision(
            action="execute",
            reasoning_summary=(
                f"Selected '{tool_call.tool_name}' using "
                "the deterministic baseline controller."
            ),
            tool_call=tool_call,
            confidence=confidence,
        )

    def should_continue(
        self,
        state: AgentState,
    ) -> bool:
        """
        Determine whether execution can continue.
        """

        if state.status in {
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.ABORTED,
        }:
            return False

        if len(state.tool_calls) >= self.max_steps:
            return False

        if not state.has_remaining_steps():
            return False

        return True

    def can_replan(
        self,
        state: AgentState,
        max_replans: int,
    ) -> bool:
        """
        Determine whether another replan is allowed.
        """

        if max_replans < 0:
            raise ValueError(
                "max_replans cannot be negative."
            )

        return state.replan_count < max_replans

    def mark_replan(
        self,
        state: AgentState,
    ) -> None:
        """
        Record that the controller requested another plan.
        """

        state.increment_replan()

    def _select_tool_from_goal(
        self,
        goal: str,
    ) -> Optional[ToolCall]:
        """
        Conservative baseline tool selection.

        Supports:
        - explicit calculation requests
        - standalone arithmetic expressions
        - research/search requests
        - unsupported analysis requests
        """

        normalized = goal.lower().strip()

        calculation_terms = (
            "calculate",
            "calculation",
            "compute",
            "what is",
            "sum",
            "average",
            "multiply",
            "divide",
            "percentage",
            "percent",
            "square root",
            "sqrt",
        )

        expression = self._extract_expression(
            goal
        )

        is_raw_arithmetic = (
            expression is not None
            and self._is_standalone_arithmetic_expression(
                goal
            )
        )

        if is_raw_arithmetic or any(
            term in normalized
            for term in calculation_terms
        ):
            if expression:
                return ToolCall(
                    tool_name="calculate",
                    arguments={
                        "expression": expression,
                    },
                )

        research_terms = (
            "research",
            "find information",
            "search",
            "look up",
            "who is",
            "what happened",
            "what was",
            "explain",
            "information about",
            "tell me about",
        )

        if any(
            term in normalized
            for term in research_terms
        ):
            return ToolCall(
                tool_name="search_knowledge",
                arguments={
                    "query": goal,
                    "top_k": 5,
                },
            )

        analysis_terms = (
            "analyze data",
            "analyse data",
            "analyze the data",
            "analyse the data",
            "correlation",
            "mean of",
            "average of",
            "maximum of",
            "minimum of",
        )

        if any(
            term in normalized
            for term in analysis_terms
        ):
            return None

        return None

    def _is_repeated_tool_call(
        self,
        state: AgentState,
        tool_call: ToolCall,
    ) -> bool:
        """
        Detect repeated identical calls.

        This prevents the agent from repeatedly executing the
        exact same tool with the exact same arguments.
        """

        signature = self._tool_signature(
            tool_call
        )

        previous_signatures = [
            self._tool_signature(call)
            for call in state.tool_calls
        ]

        count = Counter(
            previous_signatures
        )[signature]

        return count >= self.max_identical_tool_calls

    @staticmethod
    def _tool_signature(
        tool_call: ToolCall,
    ) -> str:
        arguments = repr(
            sorted(
                tool_call.arguments.items(),
                key=lambda item: item[0],
            )
        )

        return (
            f"{tool_call.tool_name}:{arguments}"
        )

    @staticmethod
    def _extract_expression(
        goal: str,
    ) -> Optional[str]:
        markers = (
            "calculate",
            "compute",
            "what is",
        )

        expression = goal.strip()

        for marker in markers:
            if expression.lower().startswith(marker):
                expression = expression[
                    len(marker):
                ].strip()
                break

        expression = expression.rstrip("?.!")

        if not expression:
            return None

        allowed_characters = set(
            "0123456789+-*/().% "
        )

        if not set(expression).issubset(
            allowed_characters
        ):
            return None

        if expression.endswith("%"):
            number = expression[:-1].strip()

            try:
                float(number)
            except ValueError:
                return None

            return f"({number}) / 100"

        return expression

    @staticmethod
    def _is_standalone_arithmetic_expression(
        goal: str,
    ) -> bool:
        """
        Determine whether the complete goal is a standalone
        arithmetic expression.

        The expression may contain:
        - integers
        - decimal numbers
        - parentheses
        - +, -, *, /, %
        - whitespace

        Natural-language calculation requests such as
        "calculate 25 * 4" are handled separately.
        """

        expression = goal.strip()

        if not expression:
            return False

        expression = expression.rstrip("?.!").strip()

        if not expression:
            return False

        allowed_characters = set(
            "0123456789+-*/().% "
        )

        if not set(expression).issubset(
            allowed_characters
        ):
            return False

        if not any(
            operator in expression
            for operator in (
                "+",
                "-",
                "*",
                "/",
                "%",
            )
        ):
            return False

        if (
            expression.startswith("*")
            or expression.startswith("/")
            or expression.startswith("%")
            or expression.endswith("*")
            or expression.endswith("/")
        ):
            return False

        if expression.count("(") != expression.count(")"):
            return False

        try:
            compile(
                expression,
                "<arithmetic-expression>",
                "eval",
            )
        except (
            SyntaxError,
            ValueError,
            TypeError,
        ):
            return False

        return True

    def _stop_decision(
        self,
        state: AgentState,
        reason: str,
        confidence: float = 1.0,
        final_answer: Optional[str] = None,
    ) -> AgentDecision:
        return AgentDecision(
            action="finish",
            reasoning_summary=reason,
            confidence=confidence,
            final_answer=(
                final_answer
                if final_answer is not None
                else self._build_final_answer(state)
            ),
        )

    @staticmethod
    def _build_final_answer(
        state: AgentState,
    ) -> Optional[str]:
        successful_results = [
            result
            for result in state.tool_results
            if result.success
        ]

        if not successful_results:
            if state.errors:
                return (
                    "The agent could not complete "
                    "the task. "
                    f"Reason: {state.errors[-1]}"
                )

            return None

        latest = successful_results[-1]

        if latest.tool_name == "calculate":
            return str(latest.output)

        if isinstance(
            latest.output,
            dict,
        ):
            if "summary" in latest.output:
                return str(
                    latest.output["summary"]
                )

            if "report" in latest.output:
                return str(
                    latest.output["report"]
                )

        if isinstance(
            latest.output,
            list,
        ):
            return "\n".join(
                str(item)
                for item in latest.output
            )

        return str(latest.output)