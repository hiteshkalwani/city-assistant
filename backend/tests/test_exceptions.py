"""
Comprehensive unit tests for the custom exception classes and error handling utilities.

This module tests all custom exception classes, error response models,
and HTTP exception handling functionality.
"""

import sys
from pathlib import Path
from fastapi import HTTPException, status

# Add the parent directory to Python path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.exceptions import (
    CityAssistantError,
    LLMError,
    ToolExecutionError,
    ExternalAPIError,
    CityNotFoundError,
    ErrorResponse,
    handle_http_exception
)


class TestCityAssistantError:
    """Test cases for the base CityAssistantError class."""

    def test_default_initialization(self):
        """Test CityAssistantError with default parameters."""
        error = CityAssistantError()
        
        assert error.message == "An unexpected error occurred"
        assert error.code == "internal_error"
        assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert error.details == {}
        assert str(error) == "An unexpected error occurred"

    def test_custom_initialization(self):
        """Test CityAssistantError with custom parameters."""
        details = {"context": "test context", "user_id": 123}
        error = CityAssistantError(
            message="Custom error message",
            code="custom_error",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )
        
        assert error.message == "Custom error message"
        assert error.code == "custom_error"
        assert error.status_code == status.HTTP_400_BAD_REQUEST
        assert error.details == details
        assert str(error) == "Custom error message"

    def test_inheritance_from_exception(self):
        """Test that CityAssistantError inherits from Exception."""
        error = CityAssistantError("Test message")
        
        assert isinstance(error, Exception)
        assert isinstance(error, CityAssistantError)

    def test_details_none_handling(self):
        """Test that None details are converted to empty dict."""
        error = CityAssistantError(details=None)
        
        assert error.details == {}


class TestLLMError:
    """Test cases for the LLMError class."""

    def test_default_initialization(self):
        """Test LLMError with default parameters."""
        error = LLMError()
        
        assert error.message == "Error with the language model service"
        assert error.code == "llm_error"
        assert error.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert error.details == {}
        assert isinstance(error, CityAssistantError)

    def test_custom_initialization(self):
        """Test LLMError with custom parameters."""
        details = {"model": "gpt-4", "tokens_used": 1500}
        error = LLMError(
            message="Model quota exceeded",
            code="quota_exceeded",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )
        
        assert error.message == "Model quota exceeded"
        assert error.code == "quota_exceeded"
        assert error.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert error.details == details

    def test_inheritance(self):
        """Test LLMError inheritance."""
        error = LLMError()
        
        assert isinstance(error, CityAssistantError)
        assert isinstance(error, Exception)


class TestToolExecutionError:
    """Test cases for the ToolExecutionError class."""

    def test_default_initialization(self):
        """Test ToolExecutionError with default parameters."""
        error = ToolExecutionError("weather_tool")
        
        assert error.tool_name == "weather_tool"
        assert error.message == "weather_tool error: Tool execution failed"
        assert error.code == "tool_execution_error"
        assert error.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert error.details["tool"] == "weather_tool"

    def test_custom_initialization(self):
        """Test ToolExecutionError with custom parameters."""
        details = {"input": "London", "attempt": 3}
        error = ToolExecutionError(
            tool_name="time_tool",
            message="API timeout",
            code="timeout_error",
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            details=details
        )
        
        assert error.tool_name == "time_tool"
        assert error.message == "time_tool error: API timeout"
        assert error.code == "timeout_error"
        assert error.status_code == status.HTTP_504_GATEWAY_TIMEOUT
        assert error.details["tool"] == "time_tool"
        assert error.details["input"] == "London"
        assert error.details["attempt"] == 3

    def test_details_merging(self):
        """Test that tool name is added to existing details."""
        original_details = {"existing": "data"}
        error = ToolExecutionError("test_tool", details=original_details)
        
        assert error.details["tool"] == "test_tool"
        assert error.details["existing"] == "data"

    def test_empty_details_handling(self):
        """Test handling when details is None."""
        error = ToolExecutionError("test_tool", details=None)
        
        assert error.details == {"tool": "test_tool"}


class TestExternalAPIError:
    """Test cases for the ExternalAPIError class."""

    def test_default_initialization(self):
        """Test ExternalAPIError with default parameters."""
        error = ExternalAPIError("OpenWeatherMap")
        
        assert error.service == "OpenWeatherMap"
        assert error.message == "OpenWeatherMap API error: External API request failed"
        assert error.code == "external_api_error"
        assert error.status_code == status.HTTP_502_BAD_GATEWAY
        assert error.details["service"] == "OpenWeatherMap"

    def test_custom_initialization(self):
        """Test ExternalAPIError with custom parameters."""
        details = {"endpoint": "/weather", "status_code": 401}
        error = ExternalAPIError(
            service="GeoDB",
            message="Authentication failed",
            code="auth_error",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )
        
        assert error.service == "GeoDB"
        assert error.message == "GeoDB API error: Authentication failed"
        assert error.code == "auth_error"
        assert error.status_code == status.HTTP_401_UNAUTHORIZED
        assert error.details["service"] == "GeoDB"
        assert error.details["endpoint"] == "/weather"
        assert error.details["status_code"] == 401

    def test_details_merging(self):
        """Test that service name is added to existing details."""
        original_details = {"response_time": 5000}
        error = ExternalAPIError("TestAPI", details=original_details)
        
        assert error.details["service"] == "TestAPI"
        assert error.details["response_time"] == 5000


class TestCityNotFoundError:
    """Test cases for the CityNotFoundError class."""

    def test_city_only_initialization(self):
        """Test CityNotFoundError with only city parameter."""
        error = CityNotFoundError("UnknownCity")
        
        assert error.city == "UnknownCity"
        assert error.country is None
        assert error.message == "Could not find 'UnknownCity'"
        assert error.code == "city_not_found"
        assert error.status_code == status.HTTP_404_NOT_FOUND
        assert error.details["city"] == "UnknownCity"
        assert error.details["country"] is None

    def test_city_and_country_initialization(self):
        """Test CityNotFoundError with city and country parameters."""
        error = CityNotFoundError("UnknownCity", "UnknownCountry")
        
        assert error.city == "UnknownCity"
        assert error.country == "UnknownCountry"
        assert error.message == "Could not find 'UnknownCity, UnknownCountry'"
        assert error.details["city"] == "UnknownCity"
        assert error.details["country"] == "UnknownCountry"

    def test_custom_message(self):
        """Test CityNotFoundError with custom message."""
        error = CityNotFoundError(
            "TestCity",
            "TestCountry",
            message="Custom not found message"
        )
        
        assert error.message == "Custom not found message"
        assert error.city == "TestCity"
        assert error.country == "TestCountry"

    def test_custom_parameters(self):
        """Test CityNotFoundError with custom code and status."""
        details = {"search_attempts": 3}
        error = CityNotFoundError(
            "TestCity",
            code="search_failed",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )
        
        assert error.code == "search_failed"
        assert error.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert error.details["city"] == "TestCity"
        assert error.details["search_attempts"] == 3

    def test_details_merging(self):
        """Test that city and country are added to existing details."""
        original_details = {"search_radius": 100}
        error = CityNotFoundError("TestCity", "TestCountry", details=original_details)
        
        assert error.details["city"] == "TestCity"
        assert error.details["country"] == "TestCountry"
        assert error.details["search_radius"] == 100


class TestErrorResponse:
    """Test cases for the ErrorResponse model."""

    def test_basic_initialization(self):
        """Test ErrorResponse with basic parameters."""
        response = ErrorResponse(
            error="Test error message",
            code="test_error"
        )
        
        assert response.error == "Test error message"
        assert response.code == "test_error"
        assert response.details is None

    def test_with_details(self):
        """Test ErrorResponse with details."""
        details = {"field": "value", "count": 42}
        response = ErrorResponse(
            error="Validation failed",
            code="validation_error",
            details=details
        )
        
        assert response.error == "Validation failed"
        assert response.code == "validation_error"
        assert response.details == details

    def test_from_exception_method(self):
        """Test creating ErrorResponse from CityAssistantError."""
        details = {"context": "test"}
        original_error = CityAssistantError(
            message="Original error",
            code="original_code",
            details=details
        )
        
        response = ErrorResponse.from_exception(original_error)
        
        assert response.error == "Original error"
        assert response.code == "original_code"
        assert response.details == details

    def test_from_exception_with_subclass(self):
        """Test creating ErrorResponse from CityAssistantError subclass."""
        tool_error = ToolExecutionError("test_tool", "Tool failed")
        
        response = ErrorResponse.from_exception(tool_error)
        
        assert response.error == "test_tool error: Tool failed"
        assert response.code == "tool_execution_error"
        assert response.details["tool"] == "test_tool"

    def test_model_dump(self):
        """Test model serialization."""
        details = {"key": "value"}
        response = ErrorResponse(
            error="Test error",
            code="test_code",
            details=details
        )
        
        dumped = response.model_dump()
        
        assert dumped["error"] == "Test error"
        assert dumped["code"] == "test_code"
        assert dumped["details"] == details

    def test_model_dump_exclude_none(self):
        """Test model serialization excluding None values."""
        response = ErrorResponse(
            error="Test error",
            code="test_code"
        )
        
        dumped = response.model_dump(exclude_none=True)
        
        assert dumped["error"] == "Test error"
        assert dumped["code"] == "test_code"
        assert "details" not in dumped


class TestHandleHttpException:
    """Test cases for the handle_http_exception function."""

    def test_basic_http_exception(self):
        """Test handling basic HTTPException."""
        original_exception = HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found"
        )
        
        result = handle_http_exception(original_exception)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == status.HTTP_404_NOT_FOUND
        
        detail = result.detail
        assert detail["error"] == "Resource not found"
        assert detail["code"] == "http_error"
        assert detail.get("details") is None

    def test_http_exception_with_complex_detail(self):
        """Test handling HTTPException with complex detail."""
        original_exception = HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Validation failed", "field": "email"}
        )
        
        result = handle_http_exception(original_exception)
        
        assert result.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        detail = result.detail
        assert detail["error"] == "{'message': 'Validation failed', 'field': 'email'}"
        assert detail["code"] == "http_error"

    def test_http_exception_preserves_status_code(self):
        """Test that original status code is preserved."""
        test_cases = [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ]
        
        for status_code in test_cases:
            original_exception = HTTPException(
                status_code=status_code,
                detail=f"Error {status_code}"
            )
            
            result = handle_http_exception(original_exception)
            
            assert result.status_code == status_code


class TestExceptionIntegration:
    """Integration tests for exception handling."""

    def test_exception_hierarchy(self):
        """Test that all custom exceptions inherit properly."""
        exceptions = [
            CityAssistantError(),
            LLMError(),
            ToolExecutionError("test_tool"),
            ExternalAPIError("test_service"),
            CityNotFoundError("test_city"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, CityAssistantError)
            assert isinstance(exc, Exception)
            assert hasattr(exc, 'message')
            assert hasattr(exc, 'code')
            assert hasattr(exc, 'status_code')
            assert hasattr(exc, 'details')

    def test_error_response_from_all_exceptions(self):
        """Test ErrorResponse creation from all exception types."""
        exceptions = [
            CityAssistantError("Base error"),
            LLMError("LLM error"),
            ToolExecutionError("test_tool", "Tool error"),
            ExternalAPIError("test_service", "API error"),
            CityNotFoundError("test_city", "test_country"),
        ]
        
        for exc in exceptions:
            response = ErrorResponse.from_exception(exc)
            
            assert isinstance(response, ErrorResponse)
            assert response.error == exc.message
            assert response.code == exc.code
            assert response.details == exc.details

    def test_exception_string_representation(self):
        """Test string representation of all exceptions."""
        exceptions = [
            (CityAssistantError("Test message"), "Test message"),
            (LLMError("LLM failed"), "LLM failed"),
            (ToolExecutionError("weather", "Failed"), "weather error: Failed"),
            (ExternalAPIError("API", "Timeout"), "API API error: Timeout"),
            (CityNotFoundError("Paris"), "Could not find 'Paris'"),
        ]
        
        for exc, expected_str in exceptions:
            assert str(exc) == expected_str

    def test_exception_details_consistency(self):
        """Test that exception details are consistently structured."""
        tool_error = ToolExecutionError("test_tool", details={"input": "test"})
        api_error = ExternalAPIError("test_api", details={"endpoint": "/test"})
        city_error = CityNotFoundError("test_city", "test_country", details={"attempts": 1})
        
        # Tool error should have 'tool' in details
        assert "tool" in tool_error.details
        assert tool_error.details["tool"] == "test_tool"
        assert tool_error.details["input"] == "test"
        
        # API error should have 'service' in details
        assert "service" in api_error.details
        assert api_error.details["service"] == "test_api"
        assert api_error.details["endpoint"] == "/test"
        
        # City error should have 'city' and 'country' in details
        assert "city" in city_error.details
        assert "country" in city_error.details
        assert city_error.details["city"] == "test_city"
        assert city_error.details["country"] == "test_country"
        assert city_error.details["attempts"] == 1
