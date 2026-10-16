"""
HTTP client utilities for making API requests.

This module provides a robust HTTP client with retry logic, timeouts,
and error handling for making external API requests.
"""

import json
import time
from typing import Any, Dict, Optional, Type, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.utils.exceptions import ExternalAPIError

# Type variable for generic response parsing
T = TypeVar("T", bound=BaseModel)

# Default timeout for HTTP requests
DEFAULT_TIMEOUT = 30.0

# Default headers for JSON requests
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": f"CityAssistant/{settings.ENVIRONMENT}",
}


class HTTPClient:
    """HTTP client with retry logic and error handling."""

    def __init__(
        self,
        base_url: str = "",
        headers: Optional[Dict[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        """Initialize the HTTP client.

        Args:
            base_url: Base URL for all requests
            headers: Default headers to include in all requests
            timeout: Default timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.headers = {**DEFAULT_HEADERS, **(headers or {})}
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure the HTTP client with connection pooling
        limits = httpx.Limits(
            max_keepalive_connections=5,
            max_connections=10,
            keepalive_expiry=60.0,
        )

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=timeout,
            limits=limits,
            follow_redirects=True,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close the client."""
        await self.aclose()

    async def aclose(self):
        """Close the HTTP client."""
        await self.client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=(
            retry_if_exception_type((
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
                httpx.RemoteProtocolError,
            ))
        ),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying HTTP request (attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}"
        ) if retry_state.outcome else None,
    )
    async def _request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request (can be relative if base_url is set)
            **kwargs: Additional arguments to pass to httpx.request

        Returns:
            The HTTP response

        Raises:
            ExternalAPIError: If the request fails after all retries
        """
        # Ensure URL is absolute if base_url is set
        if self.base_url and not url.startswith(('http://', 'https://')):
            url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"

        # Merge headers
        headers = {**self.headers, **kwargs.pop('headers', {})}

        # Add request ID for tracing
        request_id = f"req_{int(time.time() * 1000)}"
        headers["X-Request-ID"] = request_id

        # Log the request
        logger.debug(
            f"HTTP {method.upper()} {url} (ID: {request_id})\n"
            f"Headers: {json.dumps(headers, indent=2, default=str)}"
        )

        try:
            # Make the request
            response = await self.client.request(
                method=method,
                url=url,
                headers=headers,
                **kwargs,
            )

            # Log the response
            logger.debug(
                f"HTTP {method.upper()} {url} (ID: {request_id}) -> {response.status_code}\n"
                f"Response: {response.text[:500]}"
            )

            # Check for errors
            response.raise_for_status()
            return response

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code} for {method} {url}"
            logger.error(f"{error_msg}: {e.response.text}")

            # Try to extract error details from the response
            try:
                error_data = e.response.json()
                error_msg = error_data.get("message", error_msg)
                details = error_data.get("details", {})
            except (json.JSONDecodeError, AttributeError):
                details = {"response": e.response.text}

            raise ExternalAPIError(
                service=url.split("/")[2] if "/" in url else url,
                message=error_msg,
                status_code=e.response.status_code,
                details=details,
            ) from e

        except (httpx.RequestError, httpx.TimeoutException) as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(f"{error_msg} for {method} {url}")

            raise ExternalAPIError(
                service=url.split("/")[2] if "/" in url else url,
                message=error_msg,
                status_code=500,
            ) from e

    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Make a GET request."""
        return await self._request("GET", url, params=params, **kwargs)

    async def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Any] = None,
        **kwargs,
    ) -> httpx.Response:
        """Make a POST request."""
        if json_data is not None:
            kwargs["json"] = json_data
        elif data is not None:
            kwargs["data"] = data
        return await self._request("POST", url, **kwargs)

    async def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make a GET request and parse the response as JSON."""
        response = await self.get(url, params=params, **kwargs)
        return response.json()

    async def get_model(
        self,
        model: Type[T],
        url: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> T:
        """
        Make a GET request and parse the response as a Pydantic model.

        Args:
            model: Pydantic model to parse the response into
            url: URL to request
            params: Query parameters
            **kwargs: Additional arguments to pass to the request

        Returns:
            An instance of the model

        Raises:
            ExternalAPIError: If the response cannot be parsed into the model
        """
        response = await self.get(url, params=params, **kwargs)

        try:
            data = response.json()
            return model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            error_msg = f"Failed to parse response as {model.__name__}: {str(e)}"
            logger.error(f"{error_msg}\nResponse: {response.text}")
            raise ExternalAPIError(
                service=url.split("/")[2] if "/" in url else url,
                message=error_msg,
                status_code=response.status_code,
                details={"response": response.text},
            ) from e
