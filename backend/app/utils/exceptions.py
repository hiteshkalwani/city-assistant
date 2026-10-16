"""
Custom exceptions for the City Information Assistant.

These exceptions provide more specific error handling for different scenarios.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException, status
from pydantic import BaseModel


class CityAssistantError(Exception):
    """Base exception for all City Assistant errors."""
    
    def __init__(
        self,
        message: str = "An unexpected error occurred",
        code: str = "internal_error",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class LLMError(CityAssistantError):
    """Raised when there's an error with the LLM service."""
    
    def __init__(
        self,
        message: str = "Error with the language model service",
        code: str = "llm_error",
        status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code, status_code, details)


class ToolExecutionError(CityAssistantError):
    """Raised when a tool fails to execute."""
    
    def __init__(
        self,
        tool_name: str,
        message: str = "Tool execution failed",
        code: str = "tool_execution_error",
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.tool_name = tool_name
        details = details or {}
        details["tool"] = tool_name
        
        super().__init__(
            message=f"{tool_name} error: {message}",
            code=code,
            status_code=status_code,
            details=details,
        )


class ExternalAPIError(CityAssistantError):
    """Raised when an external API call fails."""
    
    def __init__(
        self,
        service: str,
        message: str = "External API request failed",
        code: str = "external_api_error",
        status_code: int = status.HTTP_502_BAD_GATEWAY,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.service = service
        details = details or {}
        details["service"] = service
        
        super().__init__(
            message=f"{service} API error: {message}",
            code=code,
            status_code=status_code,
            details=details,
        )


class CityNotFoundError(CityAssistantError):
    """Raised when a city cannot be found."""
    
    def __init__(
        self,
        city: str,
        country: Optional[str] = None,
        message: Optional[str] = None,
        code: str = "city_not_found",
        status_code: int = status.HTTP_404_NOT_FOUND,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.city = city
        self.country = country
        
        if message is None:
            if country:
                message = f"Could not find '{city}, {country}'"
            else:
                message = f"Could not find '{city}'"
        
        details = details or {}
        details.update({
            "city": city,
            "country": country,
        })
        
        super().__init__(message, code, status_code, details)


class ErrorResponse(BaseModel):
    """Standard error response model for the API."""
    
    error: str
    code: str
    details: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_exception(cls, exc: CityAssistantError) -> "ErrorResponse":
        """Create an error response from an exception."""
        return cls(
            error=exc.message,
            code=exc.code,
            details=exc.details,
        )


def handle_http_exception(exc: HTTPException) -> HTTPException:
    """Convert HTTP exceptions to our standard error format."""
    return HTTPException(
        status_code=exc.status_code,
        detail=ErrorResponse(
            error=str(exc.detail),
            code="http_error",
        ).model_dump(),
    )
