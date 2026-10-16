"""
Health check endpoints for the API.

This module provides endpoints to check the health status of the service
and its dependencies.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import LLMClient
from app.utils.http_client import HTTPClient

# Create router
router = APIRouter()


class HealthCheck(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Current server timestamp")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Current environment")
    dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of external dependencies",
    )


class HealthCheckResponse(BaseModel):
    """Health check response wrapper."""

    data: HealthCheck


class ErrorResponse(BaseModel):
    """Error response model for health check failures."""

    error: str
    details: Dict[str, Any] = {}


async def check_openai() -> str:
    """Check if OpenAI API is accessible."""
    try:
        llm = LLMClient()
        # Simple completion to test the API
        await llm.ainvoke("Hello, world!")
        return "ok"
    except Exception as e:
        return f"error: {str(e)}"


async def check_weather_api() -> str:
    """Check if OpenWeatherMap API is accessible."""
    if not settings.OPENWEATHER_API_KEY:
        return "not_configured"

    try:
        async with HTTPClient("https://api.openweathermap.org") as client:
            response = await client.get(
                "/data/2.5/weather",
                params={
                    "q": "London,UK",
                    "appid": settings.OPENWEATHER_API_KEY,
                },
            )
            return "ok" if response.status_code == 200 else f"error: HTTP {response.status_code}"
    except Exception as e:
        return f"error: {str(e)}"


async def check_geodb_api() -> str:
    """Check if GeoDB API is accessible."""
    if not settings.GEODB_API_KEY:
        return "not_configured"

    try:
        async with HTTPClient("https://wft-geo-db.p.rapidapi.com") as client:
            response = await client.get(
                "/v1/geo/cities",
                headers={
                    "X-RapidAPI-Key": settings.GEODB_API_KEY,
                    "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com",
                },
                params={"limit": 1},
            )
            return "ok" if response.status_code == 200 else f"error: HTTP {response.status_code}"
    except Exception as e:
        return f"error: {str(e)}"


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    responses={
        200: {"model": HealthCheckResponse},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    },
)
async def health_check() -> Dict[str, HealthCheck]:
    """
    Health check endpoint.

    Returns:
        Health status of the service and its dependencies.
    """
    # Check all dependencies in parallel
    openai_status = await check_openai()
    weather_status = await check_weather_api()
    geodb_status = await check_geodb_api()

    # Determine overall status
    dependencies_ok = all(
        status == "ok" or status == "not_configured"
        for status in [openai_status, weather_status, geodb_status]
    )
    overall_status = "ok" if dependencies_ok else "degraded"

    # Prepare response
    health_data = HealthCheck(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="0.1.0",
        environment=settings.ENVIRONMENT,
        dependencies={
            "openai": openai_status,
            "openweathermap": weather_status,
            "geodb": geodb_status,
        },
    )

    return {"data": health_data}
