# app/services/agent_service.py

from __future__ import annotations

from typing import Any, Dict, List
from uuid import uuid4

from app.agent.agent_loop import AgentLoop
from app.agent.controller import AgentController
from app.agent.planner import BaselinePlanner
from app.agent.schemas import AgentTask
from app.agent.state import AgentState
from app.tools.calculator import SafeCalculatorTool
from app.tools.executor import ToolExecutor
from app.tools.knowledge_search import KnowledgeSearchTool
from app.tools.registry import ToolRegistry


class AgentService:
    """
    Application service for the Autonomous Research & Decision Agent.

    Phase 6 keeps the deterministic controller as the production
    execution path because the learned selector is still a candidate.
    """

    def __init__(
        self,
        max_steps: int = 8,
        max_retries_per_step: int = 2,
    ) -> None:
        if max_steps < 1:
            raise ValueError(
                "max_steps must be at least 1."
            )

        if max_retries_per_step < 0:
            raise ValueError(
                "max_retries_per_step cannot be negative."
            )

        self.max_steps = max_steps
        self.max_retries_per_step = (
            max_retries_per_step
        )

        self.registry = self._build_registry()

    def _build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()

        registry.register(
            SafeCalculatorTool()
        )

        registry.register(
            KnowledgeSearchTool()
        )

        return registry

    def _build_agent(
        self,
        max_steps: int,
    ) -> AgentLoop:
        controller = AgentController(
            registry=self.registry,
            max_steps=max_steps,
        )

        executor = ToolExecutor(
            registry=self.registry
        )

        return AgentLoop(
            controller=controller,
            executor=executor,
            max_retries_per_step=(
                self.max_retries_per_step
            ),
        )

    def run(
        self,
        goal: str,
        max_steps: int | None = None,
    ) -> AgentState:
        normalized_goal = goal.strip()

        if not normalized_goal:
            raise ValueError(
                "Goal cannot be empty."
            )

        effective_max_steps = (
            self.max_steps
            if max_steps is None
            else max_steps
        )

        if effective_max_steps < 1:
            raise ValueError(
                "max_steps must be at least 1."
            )

        if effective_max_steps > 50:
            raise ValueError(
                "max_steps cannot exceed 50."
            )

        task = AgentTask(
            task_id=f"task-{uuid4().hex}",
            goal=normalized_goal,
        )

        planner = BaselinePlanner()

        plan = planner.create_plan(
            task
        )

        state = AgentState(
            task=task
        )

        state.set_plan(
            plan
        )

        agent = self._build_agent(
            max_steps=effective_max_steps
        )

        return agent.run(
            state
        )

    def list_tools(self) -> List[str]:
        return [
            "calculate",
            "search_knowledge",
        ]

    @staticmethod
    def _serialize_tool_call(
        tool_call: Any,
    ) -> Dict[str, Any]:
        return {
            "tool_name": tool_call.tool_name,
            "arguments": dict(
                tool_call.arguments
            ),
        }

    @staticmethod
    def _serialize_tool_result(
        result: Any,
    ) -> Dict[str, Any]:
        return {
            "tool_name": result.tool_name,
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "latency_ms": result.latency_ms,
        }

    @staticmethod
    def _serialize_decision(
        decision: Any,
    ) -> Dict[str, Any]:
        tool_call = None

        if decision.tool_call is not None:
            tool_call = {
                "tool_name": (
                    decision.tool_call.tool_name
                ),
                "arguments": dict(
                    decision.tool_call.arguments
                ),
            }

        return {
            "action": decision.action,
            "reasoning_summary": (
                decision.reasoning_summary
            ),
            "confidence": decision.confidence,
            "final_answer": decision.final_answer,
            "tool_call": tool_call,
        }

    @staticmethod
    def _serialize_observation(
        observation: Any,
    ) -> Dict[str, Any]:
        return {
            "step_id": observation.step_id,
            "source": observation.source,
            "content": observation.content,
            "success": observation.success,
            "metadata": dict(
                observation.metadata
            ),
        }

    def serialize_state(
        self,
        state: AgentState,
    ) -> Dict[str, Any]:
        final_answer = state.metadata.get(
            "final_answer"
        )

        return {
            "task_id": state.task.task_id,
            "goal": state.task.goal,
            "status": state.status.value,
            "final_answer": final_answer,
            "tool_calls": [
                self._serialize_tool_call(
                    item
                )
                for item in state.tool_calls
            ],
            "tool_results": [
                self._serialize_tool_result(
                    item
                )
                for item in state.tool_results
            ],
            "decisions": [
                self._serialize_decision(
                    item
                )
                for item in state.decisions
            ],
            "observations": [
                self._serialize_observation(
                    item
                )
                for item in state.observations
            ],
            "errors": list(
                state.errors
            ),
            "retry_count": state.retry_count,
            "replan_count": state.replan_count,
            "metadata": dict(
                state.metadata
            ),
        }