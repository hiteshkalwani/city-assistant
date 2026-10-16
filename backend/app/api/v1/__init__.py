"""
API v1 router.

This module initializes the API v1 router and includes all endpoints.
"""

from fastapi import APIRouter

from app.api.v1 import health, chat

# Create the API v1 router
api_router = APIRouter()

# Include all endpoint modules
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
