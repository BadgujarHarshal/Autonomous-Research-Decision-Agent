# app/main.py

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


APP_NAME = os.getenv(
    "APP_NAME",
    "Autonomous Research & Decision Agent",
)

APP_VERSION = os.getenv(
    "APP_VERSION",
    "0.1.0",
)

FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173",
)

origins = [
    origin.strip()
    for origin in FRONTEND_ORIGINS.split(",")
    if origin.strip()
]


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "API for the Autonomous Research "
        "and Decision Agent."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    router
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
    }