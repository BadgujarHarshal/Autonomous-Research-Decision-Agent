# app/agent/schemas.py

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class AgentStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    EXECUTING = "executing"
    OBSERVING = "observing"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ToolCall(BaseModel):
    tool_name: str = Field(min_length=1)
    arguments: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("tool_name cannot be empty")

        return value


class ToolResult(BaseModel):
    tool_name: str = Field(min_length=1)
    success: bool
    output: Any = None
    error: Optional[str] = None
    latency_ms: float = Field(default=0.0, ge=0.0)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("tool_name cannot be empty")

        return value


class PlanStep(BaseModel):
    step_id: int = Field(ge=1)
    description: str = Field(min_length=1)
    expected_tool: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Optional[ToolResult] = None


class AgentPlan(BaseModel):
    goal: str = Field(min_length=1)
    steps: List[PlanStep] = Field(default_factory=list)

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("goal cannot be empty")

        return value


class Observation(BaseModel):
    step_id: int = Field(ge=1)
    source: str = Field(min_length=1)
    content: Any = None
    success: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    action: str = Field(min_length=1)
    reasoning_summary: str = ""
    tool_call: Optional[ToolCall] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    final_answer: Optional[str] = None


class AgentTask(BaseModel):
    task_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id", "goal")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("value cannot be empty")

        return value


class AgentTrace(BaseModel):
    task_id: str
    status: AgentStatus
    plan: Optional[AgentPlan] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    decisions: List[AgentDecision] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)