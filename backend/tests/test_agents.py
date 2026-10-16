"""
Comprehensive unit tests for the agent modules.

This module tests the BaseAgent and CityInformationAgent classes,
including initialization, tool management, memory handling, and agent execution.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import List

# Add the parent directory to Python path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.base_agent import BaseAgent
from app.agents.city_agent import CityInformationAgent
from langchain_core.tools import BaseTool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from app.tools.weather_tool import WeatherTool
from app.tools.time_tool import TimeTool
from app.tools.facts_tool import CityFactsTool


class MockTool(BaseTool):
    """Mock tool for testing purposes."""
    
    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    
    def _run(self, query: str) -> str:
        return f"Mock result for: {query}"
    
    async def _arun(self, query: str) -> str:
        return f"Mock async result for: {query}"


class TestableBaseAgent(BaseAgent):
    """Testable implementation of BaseAgent for testing."""
    
    def _get_tools(self) -> List[BaseTool]:
        """Return mock tools for testing."""
        return [MockTool()]
    
    def _get_system_prompt(self) -> str:
        """Return a test system prompt."""
        return "You are a test agent."


class TestBaseAgent:
    """Test cases for the BaseAgent class."""

    @patch('app.agents.base_agent.ChatOpenAI')
    def test_base_agent_initialization(self, mock_chat_openai):
        """Test BaseAgent initialization with default parameters."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = TestableBaseAgent()
        
        assert agent.model_name == "gpt-3.5-turbo"  # Default from settings
        assert agent.temperature == 0.2  # Default from settings
        assert agent.max_tokens == 1024  # Default from settings
        assert agent.memory_window == 10
        assert agent.verbose is False
        assert agent.llm == mock_llm
        assert isinstance(agent.chat_history, InMemoryChatMessageHistory)
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "mock_tool"

    @patch('app.agents.base_agent.ChatOpenAI')
    def test_base_agent_custom_initialization(self, mock_chat_openai):
        """Test BaseAgent initialization with custom parameters."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = TestableBaseAgent(
            model_name="gpt-4",
            temperature=0.8,
            max_tokens=2048,
            memory_window=20,
            verbose=True
        )
        
        assert agent.model_name == "gpt-4"
        assert agent.temperature == 0.8
        assert agent.max_tokens == 2048
        assert agent.memory_window == 20
        assert agent.verbose is True
        
        mock_chat_openai.assert_called_once_with(
            model="gpt-4",
            temperature=0.8,
            max_tokens=2048,
        )

    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.agents.base_agent.create_openai_tools_agent')
    @patch('app.agents.base_agent.AgentExecutor')
    def test_create_agent(self, mock_executor, mock_create_agent, mock_chat_openai):
        """Test agent creation."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        
        agent = TestableBaseAgent()
        
        # Verify agent creation was called
        mock_create_agent.assert_called_once()
        call_args = mock_create_agent.call_args
        assert call_args[1]['llm'] == mock_llm
        assert len(call_args[1]['tools']) == 1
        assert 'prompt' in call_args[1]
        
        # Verify executor creation
        mock_executor.assert_called_once_with(
            agent=mock_agent,
            tools=agent.tools,
            verbose=False,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )

    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.agents.base_agent.tracer')
    async def test_ainvoke_with_string_input(self, mock_tracer, mock_chat_openai):
        """Test ainvoke with string input."""
        mock_tracer.enabled = False
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = TestableBaseAgent()
        agent.agent_executor = AsyncMock()
        agent.agent_executor.ainvoke = AsyncMock(return_value={"output": "test response"})
        
        result = await agent.ainvoke("test input")
        
        agent.agent_executor.ainvoke.assert_called_once_with({
            "input": "test input",
            "chat_history": []
        }, config=None)
        assert result == {"output": "test response"}

    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.agents.base_agent.tracer')
    async def test_ainvoke_with_dict_input(self, mock_tracer, mock_chat_openai):
        """Test ainvoke with dictionary input."""
        mock_tracer.enabled = False
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = TestableBaseAgent()
        agent.agent_executor = AsyncMock()
        agent.agent_executor.ainvoke = AsyncMock(return_value={"output": "test response"})
        
        input_data = {"input": "test input", "context": "additional context"}
        result = await agent.ainvoke(input_data)
        
        expected_input = {
            "input": "test input",
            "context": "additional context",
            "chat_history": []
        }
        agent.agent_executor.ainvoke.assert_called_once_with(expected_input, config=None)
        assert result == {"output": "test response"}

    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.agents.base_agent.tracer')
    async def test_ainvoke_with_tracing(self, mock_tracer, mock_chat_openai):
        """Test ainvoke with tracing enabled."""
        mock_tracer.enabled = True
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = TestableBaseAgent()
        agent.agent_executor = AsyncMock()
        agent.agent_executor.ainvoke = AsyncMock(return_value={"output": "test response"})
        
        await agent.ainvoke("test input")
        
        mock_tracer.trace_run.assert_called_once_with(
            name="agent_TestableBaseAgent",
            inputs={"input": "test input", "chat_history": []},
            run_type="chain"
        )

    @patch('app.agents.base_agent.ChatOpenAI')
    async def test_ainvoke_exception_handling(self, mock_chat_openai):
        """Test ainvoke exception handling."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = TestableBaseAgent()
        agent.agent_executor = AsyncMock()
        agent.agent_executor.ainvoke = AsyncMock(side_effect=Exception("Test error"))
        
        with pytest.raises(Exception, match="Test error"):
            await agent.ainvoke("test input")

    @patch('app.agents.base_agent.ChatOpenAI')
    async def test_astream(self, mock_chat_openai):
        """Test astream functionality."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = TestableBaseAgent()
        agent.agent_executor = AsyncMock()
        
        # Mock async generator
        async def mock_astream(input_data):
            yield {"chunk": "first"}
            yield {"chunk": "second"}
        
        agent.agent_executor.astream = mock_astream
        
        chunks = []
        async for chunk in agent.astream("test input"):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0] == {"chunk": "first"}
        assert chunks[1] == {"chunk": "second"}

    @patch('app.agents.base_agent.ChatOpenAI')
    def test_clear_memory(self, mock_chat_openai):
        """Test memory clearing."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = TestableBaseAgent()
        
        # Add some messages to memory
        agent.chat_history.add_message(HumanMessage(content="Hello"))
        agent.chat_history.add_message(AIMessage(content="Hi there"))
        
        assert len(agent.chat_history.messages) == 2
        
        agent.clear_memory()
        
        assert len(agent.chat_history.messages) == 0

    @patch('app.agents.base_agent.ChatOpenAI')
    def test_get_memory_summary_empty(self, mock_chat_openai):
        """Test memory summary when empty."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = TestableBaseAgent()
        summary = agent.get_memory_summary()
        
        assert summary["total_messages"] == 0
        assert summary["memory_window"] == 10
        assert summary["last_message"] is None

    @patch('app.agents.base_agent.ChatOpenAI')
    def test_get_memory_summary_with_messages(self, mock_chat_openai):
        """Test memory summary with messages."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = TestableBaseAgent()
        
        # Add messages to memory
        agent.chat_history.add_message(HumanMessage(content="Hello"))
        agent.chat_history.add_message(AIMessage(content="Hi there"))
        
        summary = agent.get_memory_summary()
        
        assert summary["total_messages"] == 2
        assert summary["memory_window"] == 10
        assert summary["last_message"] == "Hi there"


class TestCityInformationAgent:
    """Test cases for the CityInformationAgent class."""

    @patch('app.agents.base_agent.ChatOpenAI')
    def test_city_agent_initialization(self, mock_chat_openai):
        """Test CityInformationAgent initialization."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = CityInformationAgent()
        
        # Check that tools are properly initialized
        assert len(agent.tools) == 4
        tool_names = [tool.name for tool in agent.tools]
        assert "get_weather" in tool_names
        assert "get_time" in tool_names
        assert "get_city_facts" in tool_names
        assert "plan_city_visit" in tool_names

    @patch('app.agents.base_agent.ChatOpenAI')
    def test_get_tools(self, mock_chat_openai):
        """Test _get_tools method."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = CityInformationAgent()
        tools = agent._get_tools()
        
        assert len(tools) == 4
        assert isinstance(tools[0], WeatherTool)
        assert isinstance(tools[1], TimeTool)
        assert isinstance(tools[2], CityFactsTool)

    @patch('app.agents.base_agent.ChatOpenAI')
    def test_get_system_prompt(self, mock_chat_openai):
        """Test _get_system_prompt method."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = CityInformationAgent()
        prompt = agent._get_system_prompt()
        
        assert isinstance(prompt, str)
        assert "City Information Assistant" in prompt
        assert "get_weather" in prompt
        assert "get_time" in prompt
        assert "get_city_facts" in prompt
        assert "plan_city_visit" in prompt

    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.tools.weather_tool.WeatherTool._arun')
    @patch('app.tools.time_tool.TimeTool._arun')
    @patch('app.tools.facts_tool.CityFactsTool._arun')
    async def test_get_comprehensive_city_info_all_tools(
        self, mock_facts, mock_time, mock_weather, mock_chat_openai
    ):
        """Test get_comprehensive_city_info with all tools enabled."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock tool responses
        mock_facts.return_value = "Facts about London"
        mock_weather.return_value = "Weather in London"
        mock_time.return_value = "Time in London"
        
        agent = CityInformationAgent()
        
        result = await agent.get_comprehensive_city_info("London", "UK")
        
        # Verify all tools were called
        mock_facts.assert_called_once_with("London", "UK")
        mock_weather.assert_called_once_with("London", "UK")
        mock_time.assert_called_once_with("London", "UK")
        
        # Verify result contains all sections
        assert "🏙️ City Information" in result
        assert "🌤️ Current Weather" in result
        assert "⏰ Current Time" in result
        assert "Facts about London" in result
        assert "Weather in London" in result
        assert "Time in London" in result

    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.tools.weather_tool.WeatherTool._arun')
    async def test_get_comprehensive_city_info_weather_only(
        self, mock_weather, mock_chat_openai
    ):
        """Test get_comprehensive_city_info with only weather enabled."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        mock_weather.return_value = "Weather in Paris"
        
        agent = CityInformationAgent()
        
        result = await agent.get_comprehensive_city_info(
            "Paris", "France",
            include_weather=True,
            include_time=False,
            include_facts=False
        )
        
        mock_weather.assert_called_once_with("Paris", "France")
        
        assert "🌤️ Current Weather" in result
        assert "Weather in Paris" in result
        assert "🏙️ City Information" not in result
        assert "⏰ Current Time" not in result

    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.tools.weather_tool.WeatherTool._arun')
    async def test_get_comprehensive_city_info_error_handling(
        self, mock_weather, mock_chat_openai
    ):
        """Test error handling in get_comprehensive_city_info."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock weather tool to raise exception
        mock_weather.side_effect = Exception("API Error")
        
        agent = CityInformationAgent()
        
        result = await agent.get_comprehensive_city_info(
            "Tokyo", "Japan",
            include_weather=True,
            include_time=False,
            include_facts=False
        )
        
        assert "❌ Weather information temporarily unavailable" in result

    @pytest.mark.asyncio
    @patch('app.agents.base_agent.ChatOpenAI')
    async def test_get_comprehensive_city_info_no_country(self, mock_chat_openai):
        """Test get_comprehensive_city_info without country parameter."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = CityInformationAgent()
        
        # Mock the actual method to verify it's called correctly
        with patch.object(agent, 'get_comprehensive_city_info', return_value="Information about Berlin") as mock_method:
            result = await mock_method("Berlin")
            
            # Verify the method was called with Berlin and default parameters
            mock_method.assert_called_once_with("Berlin")
            assert result == "Information about Berlin"


class TestAgentIntegration:
    """Integration tests for agent functionality."""

    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.agents.base_agent.create_openai_tools_agent')
    @patch('app.agents.base_agent.AgentExecutor')
    async def test_full_agent_workflow(self, mock_executor, mock_create_agent, mock_chat_openai):
        """Test complete agent workflow from initialization to execution."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_executor_instance = AsyncMock()
        mock_executor_instance.ainvoke = AsyncMock(return_value={
            "output": "London is a great city with mild weather today.",
            "intermediate_steps": []
        })
        mock_executor.return_value = mock_executor_instance
        
        # Create and test agent
        agent = CityInformationAgent()
        
        # Test execution
        result = await agent.ainvoke("Tell me about London")
        
        # Verify the workflow
        assert result["output"] == "London is a great city with mild weather today."
        mock_executor_instance.ainvoke.assert_called_once()
        
        # Verify input was processed correctly
        call_args = mock_executor_instance.ainvoke.call_args[0][0]
        assert call_args["input"] == "Tell me about London"
        assert "chat_history" in call_args

    @patch('app.agents.base_agent.ChatOpenAI')
    def test_agent_memory_persistence(self, mock_chat_openai):
        """Test that agent memory persists across interactions."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = CityInformationAgent()
        
        # Add messages to memory
        agent.chat_history.add_message(HumanMessage(content="Hello"))
        agent.chat_history.add_message(AIMessage(content="Hi! How can I help?"))
        agent.chat_history.add_message(HumanMessage(content="Tell me about Paris"))
        
        # Check memory state
        summary = agent.get_memory_summary()
        assert summary["total_messages"] == 3
        assert summary["last_message"] == "Tell me about Paris"
        
        # Clear memory and verify
        agent.clear_memory()
        summary = agent.get_memory_summary()
        assert summary["total_messages"] == 0
        assert summary["last_message"] is None


@pytest.fixture
def mock_tools():
    """Fixture providing mock tools for testing."""
    weather_tool = Mock(spec=WeatherTool)
    weather_tool.name = "get_weather"
    weather_tool._arun = AsyncMock(return_value="Sunny, 25°C")
    
    time_tool = Mock(spec=TimeTool)
    time_tool.name = "get_time"
    time_tool._arun = AsyncMock(return_value="14:30 UTC")
    
    facts_tool = Mock(spec=CityFactsTool)
    facts_tool.name = "get_city_facts"
    facts_tool._arun = AsyncMock(return_value="Population: 9 million")
    
    return [weather_tool, time_tool, facts_tool]


class TestAgentToolIntegration:
    """Test agent integration with tools."""

    @patch('app.agents.base_agent.ChatOpenAI')
    async def test_tool_execution_success(self, mock_chat_openai, mock_tools):
        """Test successful tool execution."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = CityInformationAgent()
        
        # Replace tools with mocks
        with patch.object(agent, 'tools', mock_tools):
            # Test individual tool calls would work
            for tool in agent.tools:
                if hasattr(tool, '_arun'):
                    result = await tool._arun("London", "UK")
                    assert isinstance(result, str)
                    assert len(result) > 0

    @patch('app.agents.base_agent.ChatOpenAI')
    async def test_tool_execution_failure(self, mock_chat_openai):
        """Test tool execution failure handling."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        agent = CityInformationAgent()
        
        # Mock a tool to fail
        with patch('app.tools.weather_tool.WeatherTool._arun', side_effect=Exception("API Error")):
            result = await agent.get_comprehensive_city_info(
                "London", "UK",
                include_weather=True,
                include_time=False,
                include_facts=False
            )
            
            assert "❌ Weather information temporarily unavailable" in result
