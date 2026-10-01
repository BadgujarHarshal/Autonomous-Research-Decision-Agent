# app/api/schemas.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=10000)
    max_steps: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
    )


class ToolCallResponse(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


class ToolResultResponse(BaseModel):
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    latency_ms: float


class DecisionResponse(BaseModel):
    action: str
    reasoning_summary: str
    confidence: float
    final_answer: Optional[str] = None
    tool_call: Optional[ToolCallResponse] = None


class AgentRunResponse(BaseModel):
    task_id: str
    goal: str
    status: str
    final_answer: Optional[str] = None
    tool_calls: List[ToolCallResponse]
    tool_results: List[ToolResultResponse]
    decisions: List[DecisionResponse]
    observations: List[Dict[str, Any]]
    errors: List[str]
    retry_count: int
    replan_count: int
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    tools: List[str]


class ToolListResponse(BaseModel):
    tools: List[str]