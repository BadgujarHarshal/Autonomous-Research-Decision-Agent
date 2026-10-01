# app/api/routes.py

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    HealthResponse,
    ToolListResponse,
)
from app.services.agent_service import AgentService


router = APIRouter(
    prefix="/api",
    tags=["agent"],
)


SERVICE_VERSION = os.getenv(
    "APP_VERSION",
    "0.1.0",
)

agent_service = AgentService(
    max_steps=int(
        os.getenv(
            "AGENT_MAX_STEPS",
            "8",
        )
    ),
    max_retries_per_step=int(
        os.getenv(
            "AGENT_MAX_RETRIES_PER_STEP",
            "2",
        )
    ),
)


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="autonomous-research-decision-agent",
        version=SERVICE_VERSION,
        tools=agent_service.list_tools(),
    )


@router.get(
    "/tools",
    response_model=ToolListResponse,
)
def tools() -> ToolListResponse:
    return ToolListResponse(
        tools=agent_service.list_tools()
    )


@router.post(
    "/agent/run",
    response_model=AgentRunResponse,
)
def run_agent(
    request: AgentRunRequest,
) -> AgentRunResponse:
    try:
        state = agent_service.run(
            goal=request.goal,
            max_steps=request.max_steps,
        )

        payload = agent_service.serialize_state(
            state
        )

        return AgentRunResponse(
            **payload
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Agent execution failed: "
                f"{exc}"
            ),
        ) from exc