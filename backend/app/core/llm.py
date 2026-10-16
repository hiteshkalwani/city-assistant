"""
LLM client wrapper for OpenAI API.

This module provides a wrapper around the OpenAI API with retry logic,
usage tracking, and integration with LangSmith tracing.
"""

import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.observability import tracer
from app.utils.exceptions import LLMError

# Type variables for generics
T = TypeVar("T", bound=BaseModel)


class TokenUsage(BaseModel):
    """Model for tracking token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def update_usage(self, usage: Dict[str, int]) -> None:
        """Update token usage from an OpenAI API response."""
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)


class LLMClient:
    """Client for interacting with OpenAI's chat models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the LLM client.

        Args:
            model_name: Name of the OpenAI model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to ChatOpenAI
        """
        self.model_name = model_name or settings.OPENAI_MODEL
        self.temperature = temperature or settings.OPENAI_TEMPERATURE
        self.max_tokens = max_tokens or settings.OPENAI_MAX_TOKENS
        self.token_usage = TokenUsage()

        # Initialize the LangChain chat model
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=False,  # We'll handle streaming separately
            **kwargs,
        )

    def _prepare_messages(
        self,
        messages: Union[str, List[Union[BaseMessage, Dict[str, str]]]],
        system_message: Optional[str] = None,
    ) -> List[BaseMessage]:
        """Prepare messages for the LLM."""
        processed_messages = []

        # Add system message if provided
        if system_message:
            processed_messages.append(SystemMessage(content=system_message))

        # Handle different message formats
        if isinstance(messages, str):
            processed_messages.append(HumanMessage(content=messages))
        elif isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, BaseMessage):
                    processed_messages.append(msg)
                elif isinstance(msg, dict):
                    if "role" in msg and "content" in msg:
                        if msg["role"].lower() == "user":
                            processed_messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"].lower() == "assistant":
                            processed_messages.append(AIMessage(content=msg["content"]))
                        elif msg["role"].lower() == "system" and not system_message:
                            # Only add system message if not already provided
                            processed_messages.append(SystemMessage(content=msg["content"]))

        return processed_messages

    def _extract_usage(self, result: Any) -> Optional[Dict[str, int]]:
        """Extract token usage from the result."""
        if hasattr(result, "llm_output") and hasattr(result.llm_output, "token_usage"):
            return result.llm_output.token_usage
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type((
                ConnectionError,
                TimeoutError,
                json.JSONDecodeError,
            ))
        ),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying LLM call (attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}"
        ) if retry_state.outcome else None,
    )
    async def ainvoke(
        self,
        messages: Union[str, List[Union[BaseMessage, Dict[str, str]]]],
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Asynchronously invoke the LLM with the given messages.

        Args:
            messages: Messages to send to the LLM
            system_message: Optional system message to prepend
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            The generated text response

        Raises:
            LLMError: If the LLM call fails
        """
        try:
            # Prepare messages
            processed_messages = self._prepare_messages(messages, system_message)

            # Trace the LLM call
            if tracer.enabled:
                tracer.trace_run(
                    name=f"llm_{self.model_name}",
                    inputs={"messages": [m.dict() for m in processed_messages]},
                    run_type="llm",
                )

            # Call the LLM
            result = await self.llm.ainvoke(processed_messages, **kwargs)

            # Update token usage
            if usage := self._extract_usage(result):
                self.token_usage.update_usage(usage)

            return result.content

        except Exception as e:
            error_msg = f"LLM call failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise LLMError(error_msg) from e

    async def ainvoke_structured(
        self,
        output_model: Type[T],
        messages: Union[str, List[Union[BaseMessage, Dict[str, str]]]],
        system_message: Optional[str] = None,
        **kwargs,
    ) -> T:
        """Asynchronously invoke the LLM and parse the response as a Pydantic model.

        Args:
            output_model: Pydantic model to parse the response into
            messages: Messages to send to the LLM
            system_message: Optional system message to prepend
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            An instance of the output model

        Raises:
            LLMError: If the LLM call fails or the response cannot be parsed
        """
        # Add JSON format instructions to the system message
        json_schema = output_model.model_json_schema()
        format_instructions = (
            f"Respond with a JSON object that matches the following schema:\n"
            f"```json\n{json.dumps(json_schema, indent=2)}\n```"
        )

        if system_message:
            system_message = f"{system_message}\n\n{format_instructions}"
        else:
            system_message = format_instructions

        # Call the LLM
        response = await self.ainvoke(messages, system_message, **kwargs)

        try:
            # Try to parse the response as JSON
            if response.startswith("```json") and response.endswith("```"):
                # Extract JSON from code block
                json_str = response[response.find("{") : response.rfind("}") + 1]
            else:
                json_str = response.strip()

            # Parse the JSON and validate against the model
            return output_model.model_validate_json(json_str)

        except Exception as e:
            error_msg = f"Failed to parse LLM response as {output_model.__name__}: {str(e)}"
            logger.error(f"{error_msg}\nResponse: {response}")
            raise LLMError(error_msg) from e
