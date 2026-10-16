"""
Chat API endpoints for the City Information Assistant.

This module provides REST API endpoints for chat functionality,
using both synchronous and streaming interfaces.
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_community.callbacks import OpenAICallbackHandler
from pydantic import BaseModel, Field

from app.agents.city_agent import CityInformationAgent
from app.core.config import settings
from app.utils.exceptions import ErrorResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


class ChatMessage(BaseModel):
    """A message in the chat."""

    role: Literal["user", "assistant", "system"] = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What's the weather like in London?"
            }
        }


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    messages: List[ChatMessage] = Field(
        ...,
        description="The conversation history",
        min_items=1,
    )
    city: Optional[str] = Field(
        None,
        description="The city to get information about",
    )
    country: Optional[str] = Field(
        None,
        description="Optional country to disambiguate the city",
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0-2)",
    )
    max_tokens: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum number of tokens to generate",
    )
    stream: bool = Field(
        False,
        description="Whether to stream the response",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "What's the weather like in London?"}
                ],
                "city": "London",
                "country": "UK",
                "temperature": 0.7,
                "max_tokens": 500,
                "stream": False,
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    message: ChatMessage = Field(..., description="The assistant's response")
    thinking: Optional[str] = Field(
        None,
        description="The agent's reasoning process and what it's doing"
    )
    usage: Dict[str, int] = Field(
        ...,
        description="Token usage information",
        example={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Tool calls made during the response generation",
    )


class ChatStreamResponse(BaseModel):
    """Streaming response model for chat endpoint."""

    event: str = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event data")

    @classmethod
    def start(cls) -> "ChatStreamResponse":
        """Create a stream start event."""
        return cls(event="start", data={"timestamp": ""})

    @classmethod
    def token(cls, token: str) -> "ChatStreamResponse":
        """Create a token event."""
        return cls(event="token", data={"token": token})

    @classmethod
    def tool_call(cls, tool_call: Dict[str, Any]) -> "ChatStreamResponse":
        """Create a tool call event."""
        return cls(event="tool_call", data={"tool_call": tool_call})

    @classmethod
    def complete(
        cls,
        message: ChatMessage,
        usage: Dict[str, int],
    ) -> "ChatStreamResponse":
        """Create a completion event."""
        return cls(
            event="complete",
            data={
                "message": message.dict(),
                "usage": usage,
            },
        )

    @classmethod
    def error(cls, error: str) -> "ChatStreamResponse":
        """Create an error event."""
        return cls(event="error", data={"error": error})


TOOL_THINKING = {
    "get_weather": "Getting current weather information for {city}",
    "get_time": "Checking the local time in {city}",
    "get_city_facts": "Gathering interesting facts and information about {city}",
    "plan_city_visit": "Planning a comprehensive visit to {city} by combining weather, time, and city information",
}


def build_agent(temperature: Optional[float], max_tokens: Optional[int]) -> CityInformationAgent:
    """Create the agent for one request; `None` falls back to configured defaults."""
    return CityInformationAgent(
        temperature=settings.OPENAI_TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens or settings.OPENAI_MAX_TOKENS,
    )


def build_agent_input(
    messages: List[ChatMessage],
    city: Optional[str] = None,
    country: Optional[str] = None,
) -> Dict[str, Any]:
    """Turn the API conversation into the agent's `input` + `chat_history`."""
    text = messages[-1].content
    if city:
        text = f"[Context: the user is asking about {city}{', ' + country if country else ''}]\n{text}"
    return {
        "input": text,
        "chat_history": [(m.role, m.content) for m in messages[:-1]],
    }


def extract_tool_calls(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten LangChain intermediate steps into plain dicts."""
    return [
        {
            "tool": getattr(action, "tool", "unknown"),
            "input": getattr(action, "tool_input", {}),
            "output": output,
        }
        for action, output in result.get("intermediate_steps", [])
    ]


def summarize_thinking(user_message: str, tool_calls: List[Dict[str, Any]]) -> str:
    """Human-readable summary of what the agent did."""
    parts = [f"I'm analyzing your request: '{user_message}'"]
    for call in tool_calls:
        tool_input = call["input"] if isinstance(call["input"], dict) else {}
        template = TOOL_THINKING.get(call["tool"], "Using {tool} to gather relevant information")
        parts.append(template.format(city=tool_input.get("city", "the city"), tool=call["tool"]))
    return ". ".join(parts) + "."


# Tells nginx not to buffer, so events reach the browser as they are produced.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def sse(event: ChatStreamResponse) -> str:
    return f"data: {event.model_dump_json()}\n\n"


async def process_chat_stream(
    messages: List[ChatMessage],
    city: Optional[str] = None,
    country: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Process a chat request and yield server-sent events."""
    yield sse(ChatStreamResponse.start())
    try:
        agent = build_agent(temperature, max_tokens)
        result = await agent.ainvoke(build_agent_input(messages, city, country))

        for call in extract_tool_calls(result):
            yield sse(ChatStreamResponse.tool_call(call))

        # shortcut: replays the finished answer word by word, switch to agent.astream for true token streaming
        output = result.get("output", "No response generated")
        for token in output.split():
            yield sse(ChatStreamResponse.token(token + " "))

        yield sse(ChatStreamResponse.complete(
            message=ChatMessage(role="assistant", content=output),
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        ))
    except Exception:
        logger.exception("Streaming chat failed")
        yield sse(ChatStreamResponse.error("The assistant could not process this request."))


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        200: {"model": ChatResponse},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def chat(
    request: ChatRequest,
) -> Dict[str, Any]:
    """
    Chat with the City Information Assistant.

    This endpoint allows you to have a conversation with the AI assistant
    about a specific city. The assistant can provide information about
    weather, local time, and interesting facts about the city.

    Args:
        request: The chat request containing messages and city information.

    Returns:
        The assistant's response with token usage information.
    """
    if request.stream:
        return StreamingResponse(
            process_chat_stream(
                messages=request.messages,
                city=request.city,
                country=request.country,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    try:
        agent = build_agent(request.temperature, request.max_tokens)
        usage = OpenAICallbackHandler()
        result = await agent.ainvoke(
            build_agent_input(request.messages, request.city, request.country),
            config={"callbacks": [usage]},
        )
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "The assistant could not process this request."},
        )

    tool_calls = extract_tool_calls(result)
    return {
        "message": {
            "role": "assistant",
            "content": result.get("output", "I'm sorry, I couldn't process your request."),
        },
        "thinking": summarize_thinking(request.messages[-1].content, tool_calls),
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
        "tool_calls": tool_calls or None,
    }


@router.get("/stream")
async def chat_stream(
    request: Request,
    message: str,
    city: Optional[str] = None,
    country: Optional[str] = None,
) -> StreamingResponse:
    """
    Stream chat responses from the assistant.

    This endpoint provides a server-sent events (SSE) stream of the assistant's
    response, allowing for a more interactive chat experience.

    Args:
        request: The incoming request.
        message: The user's message.
        city: The city to get information about.
        country: Optional country to disambiguate the city.

    Returns:
        A streaming response with server-sent events.
    """
    return StreamingResponse(
        process_chat_stream(
            messages=[ChatMessage(role="user", content=message)],
            city=city,
            country=country,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
