"""Unit tests for the pure helpers behind the chat endpoints."""

from unittest.mock import patch

from app.api.v1.chat import (
    ChatMessage,
    build_agent,
    build_agent_input,
    extract_tool_calls,
    summarize_thinking,
)


class _Action:
    tool = "get_weather"
    tool_input = {"city": "Tokyo"}


def test_agent_input_carries_city_context_and_history():
    messages = [
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello!"),
        ChatMessage(role="user", content="Weather?"),
    ]
    agent_input = build_agent_input(messages, city="Tokyo", country="Japan")

    assert agent_input["input"].endswith("Weather?")
    assert "Tokyo, Japan" in agent_input["input"]
    assert agent_input["chat_history"] == [("user", "Hi"), ("assistant", "Hello!")]


def test_agent_input_without_city_is_the_raw_message():
    agent_input = build_agent_input([ChatMessage(role="user", content="Hi")])
    assert agent_input == {"input": "Hi", "chat_history": []}


def test_tool_calls_and_thinking_use_real_tool_names():
    calls = extract_tool_calls({"intermediate_steps": [(_Action(), "Sunny, 21°C")]})

    assert calls == [{"tool": "get_weather", "input": {"city": "Tokyo"}, "output": "Sunny, 21°C"}]
    assert "weather information for Tokyo" in summarize_thinking("Weather?", calls)


@patch("app.api.v1.chat.CityInformationAgent")
def test_zero_temperature_is_not_replaced_by_default(mock_agent):
    build_agent(temperature=0.0, max_tokens=None)
    assert mock_agent.call_args.kwargs["temperature"] == 0.0
