# app/agent/state.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agent.schemas import (
    AgentDecision,
    AgentPlan,
    AgentStatus,
    AgentTask,
    Observation,
    ToolCall,
    ToolResult,
)


class AgentState:
    """
    Runtime state for a single agent task.

    The state object deliberately keeps the execution trace explicit.
    This makes the agent observable and allows later evaluation of
    complete trajectories.
    """

    def __init__(self, task: AgentTask) -> None:
        self.task = task

        self.status: AgentStatus = AgentStatus.CREATED

        self.plan: Optional[AgentPlan] = None

        self.current_step_index: int = 0

        self.tool_calls: List[ToolCall] = []
        self.tool_results: List[ToolResult] = []
        self.observations: List[Observation] = []
        self.decisions: List[AgentDecision] = []

        self.errors: List[str] = []

        self.retry_count: int = 0
        self.replan_count: int = 0

        self.metadata: Dict[str, Any] = {}

    def set_status(self, status: AgentStatus) -> None:
        self.status = status

    def set_plan(self, plan: AgentPlan) -> None:
        self.plan = plan
        self.current_step_index = 0

    def add_tool_call(self, tool_call: ToolCall) -> None:
        self.tool_calls.append(tool_call)

    def add_tool_result(self, result: ToolResult) -> None:
        self.tool_results.append(result)

    def add_observation(self, observation: Observation) -> None:
        self.observations.append(observation)

    def add_decision(self, decision: AgentDecision) -> None:
        self.decisions.append(decision)

    def add_error(self, error: str) -> None:
        error = error.strip()

        if error:
            self.errors.append(error)

    def increment_retry(self) -> None:
        self.retry_count += 1

    def increment_replan(self) -> None:
        self.replan_count += 1

    def advance_step(self) -> None:
        if self.plan is None:
            return

        if self.current_step_index < len(self.plan.steps):
            self.current_step_index += 1

    def has_remaining_steps(self) -> bool:
        if self.plan is None:
            return False

        return self.current_step_index < len(self.plan.steps)

    def current_step(self):
        if self.plan is None:
            return None

        if not self.has_remaining_steps():
            return None

        return self.plan.steps[self.current_step_index]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "goal": self.task.goal,
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "tool_calls": len(self.tool_calls),
            "tool_results": len(self.tool_results),
            "observations": len(self.observations),
            "decisions": len(self.decisions),
            "errors": len(self.errors),
            "retry_count": self.retry_count,
            "replan_count": self.replan_count,
        }