"""
API v1 router configuration.

This module sets up the main router for API version 1,
including all endpoint modules.
"""

from fastapi import APIRouter

from app.api.v1 import chat, health

# Create the main API v1 router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
