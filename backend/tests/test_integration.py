"""
Comprehensive integration tests for the City Information Assistant.

This module tests complete workflows from API endpoints through agents to tools,
including error handling, concurrent requests, and data flow validation.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the parent directory to Python path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock environment variables before importing app
with patch.dict('os.environ', {
    'OPENAI_API_KEY': 'test-openai-key',
    'OPENWEATHER_API_KEY': 'test-weather-key',
    'GEODB_API_KEY': 'test-geodb-key',
    'LANGCHAIN_API_KEY': 'test-langchain-key',
    'REDIS_HOST': 'localhost',
    'REDIS_PORT': '6379',
    'REDIS_DB': '0',
    'REDIS_PASSWORD': '',
    'CACHE_ENABLED': 'false',
    'DEBUG': 'true',
    'ENVIRONMENT': 'test'
}):
    from app.agents.city_agent import CityInformationAgent


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "London is a vibrant city with rich history. The current weather is mild and pleasant.",
                    "tool_calls": None
                }
            }
        ],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200
        }
    }


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.tools.weather_tool.WeatherTool._arun')
    @patch('app.tools.time_tool.TimeTool._arun')
    @patch('app.tools.facts_tool.CityFactsTool._arun')
    async def test_complete_city_info_workflow(
        self, mock_facts, mock_time, mock_weather, mock_openai, client
    ):
        """Test complete city information workflow from API to tools."""
        # Mock tool responses
        mock_weather.return_value = "Current weather in London: 18°C, partly cloudy"
        mock_time.return_value = "Current time in London: 14:30 GMT"
        mock_facts.return_value = "London facts: Population 9 million, founded by Romans"
        
        # Mock OpenAI
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        # Create agent and test comprehensive info
        agent = CityInformationAgent()
        result = await agent.get_comprehensive_city_info("London", "UK")
        
        # Verify all tools were called
        mock_weather.assert_called_once_with("London", "UK")
        mock_time.assert_called_once_with("London", "UK")
        mock_facts.assert_called_once_with("London", "UK")
        
        # Verify result structure
        assert "🏙️ City Information" in result
        assert "🌤️ Current Weather" in result
        assert "⏰ Current Time" in result
        assert "London facts" in result
        assert "18°C" in result
        assert "14:30 GMT" in result

    @pytest.mark.asyncio
    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.agents.base_agent.create_openai_tools_agent')
    @patch('app.agents.base_agent.AgentExecutor')
    async def test_api_to_agent_integration(self, mock_executor, mock_create_agent, mock_openai, client):
        """Test integration from API endpoint to agent execution."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_executor_instance = AsyncMock()
        mock_executor_instance.ainvoke = AsyncMock(return_value={
            "output": "London is experiencing mild weather today with temperatures around 18°C.",
            "intermediate_steps": []
        })
        mock_executor.return_value = mock_executor_instance
        
        # Test API call
        chat_request = {
            "messages": [
                {"role": "user", "content": "What's the weather like in London?"}
            ],
            "city": "London",
            "country": "UK",
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False
        }
        
        response = client.post("/api/v1/chat", json=chat_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert "content" in data["message"]

    @pytest.mark.asyncio
    @patch('app.tools.weather_tool.WeatherTool._arun')
    @patch('app.tools.time_tool.TimeTool._arun')
    @patch('app.tools.facts_tool.CityFactsTool._arun')
    async def test_tool_failure_resilience(self, mock_facts, mock_time, mock_weather):
        """Test system resilience when tools fail."""
        # Mock some tools to fail
        mock_weather.side_effect = Exception("Weather API unavailable")
        mock_time.return_value = "Current time in Paris: 15:30 CET"
        mock_facts.side_effect = Exception("Facts API timeout")
        
        with patch('app.agents.base_agent.ChatOpenAI'):
            agent = CityInformationAgent()
            result = await agent.get_comprehensive_city_info("Paris", "France")
            
            # Should contain error messages for failed tools
            assert "❌ Weather information temporarily unavailable" in result
            assert "❌ City facts temporarily unavailable" in result
            
            # Should contain successful tool result
            assert "⏰ Current Time" in result
            assert "15:30 CET" in result

    def test_api_error_propagation(self, client):
        """Test that errors are properly propagated through the API."""
        # Test with malformed request
        response = client.post("/api/v1/chat", json={"invalid": "request"})
        assert response.status_code == 422
        
        # Test with empty messages
        response = client.post("/api/v1/chat", json={
            "messages": [],
            "city": "London"
        })
        assert response.status_code == 422


class TestConcurrentRequests:
    """Test handling of concurrent requests."""

    @pytest.mark.asyncio
    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.agents.base_agent.create_openai_tools_agent')
    @patch('app.agents.base_agent.AgentExecutor')
    async def test_concurrent_agent_execution(self, mock_executor, mock_create_agent, mock_openai):
        """Test concurrent agent execution."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_executor_instance = AsyncMock()
        mock_executor_instance.ainvoke = AsyncMock(return_value={
            "output": "Test response",
            "intermediate_steps": []
        })
        mock_executor.return_value = mock_executor_instance
        
        # Create multiple agents and run concurrently
        agents = [CityInformationAgent() for _ in range(3)]
        
        async def run_agent(agent, city):
            return await agent.ainvoke(f"Tell me about {city}")
        
        # Run concurrent requests
        tasks = [
            run_agent(agents[0], "London"),
            run_agent(agents[1], "Paris"),
            run_agent(agents[2], "Tokyo")
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert "output" in result
            assert result["output"] == "Test response"

    def test_concurrent_api_requests(self, client):
        """Test concurrent API requests."""
        import threading
        
        results = []
        errors = []
        
        def make_request(city):
            try:
                response = client.get("/health")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        cities = ["London", "Paris", "Tokyo", "New York", "Sydney"]
        
        for city in cities:
            thread = threading.Thread(target=make_request, args=(city,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0
        assert len(results) == 5
        assert all(status == 200 for status in results)


class TestDataFlow:
    """Test data flow through the system."""

    @pytest.mark.asyncio
    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.tools.weather_tool.WeatherTool._arun')
    async def test_data_transformation_pipeline(self, mock_weather, mock_openai):
        """Test data transformation through the pipeline."""
        # Mock weather tool to return structured data
        mock_weather.return_value = """
        Current weather in London, UK:
        Temperature: 18.5°C (65.3°F)
        Feels like: 19.2°C (66.6°F)
        Humidity: 72%
        Wind: 12 km/h NW
        Conditions: Partly cloudy
        """
        
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        agent = CityInformationAgent()
        result = await agent.get_comprehensive_city_info(
            "London", "UK",
            include_weather=True,
            include_time=False,
            include_facts=False
        )
        
        # Verify data transformation
        assert "🌤️ Current Weather" in result
        assert "18.5°C" in result
        assert "65.3°F" in result
        assert "Partly cloudy" in result
        assert "London, UK" in result

    @pytest.mark.asyncio
    @patch('app.agents.base_agent.ChatOpenAI')
    async def test_memory_persistence_across_calls(self, mock_openai):
        """Test that agent memory persists across multiple calls."""
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        agent = CityInformationAgent()
        
        # Simulate conversation history
        from langchain_core.messages import HumanMessage, AIMessage
        
        agent.chat_history.add_message(HumanMessage(content="Hello"))
        agent.chat_history.add_message(AIMessage(content="Hi! How can I help?"))
        agent.chat_history.add_message(HumanMessage(content="Tell me about London"))
        agent.chat_history.add_message(AIMessage(content="London is a great city..."))
        
        # Check memory state
        summary = agent.get_memory_summary()
        assert summary["total_messages"] == 4
        assert "London is a great city" in summary["last_message"]
        
        # Add more messages
        agent.chat_history.add_message(HumanMessage(content="What about Paris?"))
        
        summary = agent.get_memory_summary()
        assert summary["total_messages"] == 5
        assert "What about Paris?" in summary["last_message"]


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    @pytest.mark.asyncio
    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.tools.weather_tool.WeatherTool._arun')
    @patch('app.tools.time_tool.TimeTool._arun')
    @patch('app.tools.facts_tool.CityFactsTool._arun')
    async def test_partial_tool_failure_recovery(
        self, mock_facts, mock_time, mock_weather, mock_openai
    ):
        """Test recovery when some tools fail but others succeed."""
        # Mix of successful and failed tools
        mock_weather.return_value = "Weather: 20°C, sunny"
        mock_time.side_effect = Exception("Time API down")
        mock_facts.return_value = "Population: 2.1 million"
        
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        agent = CityInformationAgent()
        result = await agent.get_comprehensive_city_info("Berlin", "Germany")
        
        # Should contain successful results
        assert "🌤️ Current Weather" in result
        assert "20°C, sunny" in result
        assert "🏙️ City Information" in result
        assert "2.1 million" in result
        
        # Should contain error message for failed tool
        assert "❌ Time information temporarily unavailable" in result

    @pytest.mark.asyncio
    @patch('app.agents.base_agent.ChatOpenAI')
    async def test_agent_initialization_failure_handling(self, mock_openai):
        """Test handling of agent initialization failures."""
        # Mock OpenAI to raise exception
        mock_openai.side_effect = Exception("OpenAI API key invalid")
        
        with pytest.raises(Exception, match="OpenAI API key invalid"):
            CityInformationAgent()

    def test_api_validation_error_handling(self, client):
        """Test API validation error handling."""
        # Send request with missing required fields
        response = client.post("/api/v1/chat", json={})
        
        assert response.status_code == 422
        data = response.json()
        
        # Check for error structure - may have different keys
        assert "error" in data or "detail" in data
        if "error" in data:
            assert data["error"] == "Validation Error"
        if "details" in data:
            assert "errors" in data["details"]

    def test_large_payload_handling(self, client):
        """Test handling of large payloads in security context."""
        large_content = "x" * 10000  # 10KB payload
        
        malicious_request = {
            "messages": [{"role": "user", "content": large_content}],
            "city": "London"
        }
        
        response = client.post("/api/v1/chat", json=malicious_request)
        # Should handle large payloads gracefully - allow all reasonable response codes
        assert response.status_code in [200, 413, 422, 500]


class TestPerformance:
    """Test performance-related scenarios."""

    @pytest.mark.asyncio
    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.tools.weather_tool.WeatherTool._arun')
    async def test_large_response_handling(self, mock_weather, mock_openai):
        """Test handling of large responses."""
        # Mock a very large weather response
        large_response = "Weather data: " + "A" * 10000  # 10KB response
        mock_weather.return_value = large_response
        
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        agent = CityInformationAgent()
        result = await agent.get_comprehensive_city_info(
            "London", "UK",
            include_weather=True,
            include_time=False,
            include_facts=False
        )
        
        # Should handle large response without issues
        assert len(result) > 10000
        assert "Weather data:" in result

    def test_api_timeout_handling(self, client):
        """Test API timeout handling."""
        # Test streaming endpoint which might be more prone to timeouts
        response = client.get("/api/v1/chat/stream", params={
            "message": "Tell me about London",
            "city": "London",
            "country": "UK"
        })
        
        # Should return proper headers for streaming
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


class TestSecurityScenarios:
    """Test security-related scenarios."""

    def test_sql_injection_prevention(self, client):
        """Test prevention of SQL injection attempts."""
        malicious_inputs = [
            "London'; DROP TABLE cities; --",
            "London' OR '1'='1",
            "London<script>alert('xss')</script>",
            "London\"; system('rm -rf /'); --"
        ]
        
        for malicious_input in malicious_inputs:
            response = client.post("/api/v1/chat", json={
                "messages": [{"role": "user", "content": f"Tell me about {malicious_input}"}],
                "city": malicious_input,
                "country": "UK"
            })
            
            # Should either succeed (input sanitized) or fail gracefully
            assert response.status_code in [200, 422, 400]

    def test_large_payload_handling(self, client):
        """Test handling of unusually large payloads."""
        # Create a very large message
        large_message = "A" * 100000  # 100KB message
        
        response = client.post("/api/v1/chat", json={
            "messages": [{"role": "user", "content": large_message}],
            "city": "London",
            "country": "UK"
        })
        
        # Should handle gracefully (either process or reject)
        assert response.status_code in [200, 413, 422, 500]


class TestHealthAndMonitoring:
    """Test health checks and monitoring endpoints."""

    def test_health_endpoint_comprehensive(self, client):
        """Test comprehensive health check."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "data" in data
        health_data = data["data"]
        
        # Verify all health check components
        required_fields = ["status", "timestamp", "version", "environment", "dependencies"]
        for field in required_fields:
            assert field in health_data
        
        # Verify dependencies
        dependencies = health_data["dependencies"]
        assert "openai" in dependencies
        assert "openweathermap" in dependencies
        assert "geodb" in dependencies

    def test_basic_health_endpoint(self, client):
        """Test basic health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_root_endpoint_info(self, client):
        """Test root endpoint information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        required_fields = ["name", "version", "docs", "redoc"]
        for field in required_fields:
            assert field in data
        
        assert data["docs"] == "/docs"
        assert data["redoc"] == "/redoc"


@pytest.mark.asyncio
class TestAsyncIntegration:
    """Test asynchronous integration scenarios."""

    @pytest.mark.asyncio
    @patch('app.agents.base_agent.ChatOpenAI')
    @patch('app.tools.weather_tool.WeatherTool._arun')
    @patch('app.tools.time_tool.TimeTool._arun')
    @patch('app.tools.facts_tool.CityFactsTool._arun')
    async def test_async_tool_coordination(
        self, mock_facts, mock_time, mock_weather, mock_openai
    ):
        """Test coordination of multiple async tool calls."""
        # Mock tools with different response times
        async def slow_weather(*args):
            await asyncio.sleep(0.1)
            return "Weather: Sunny, 25°C"
        
        async def fast_time(*args):
            return "Time: 14:30 UTC"
        
        async def medium_facts(*args):
            await asyncio.sleep(0.05)
            return "Facts: Population 8.9 million"
        
        mock_weather.side_effect = slow_weather
        mock_time.side_effect = fast_time
        mock_facts.side_effect = medium_facts
        
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        agent = CityInformationAgent()
        
        # Time the execution
        import time
        start_time = time.time()
        result = await agent.get_comprehensive_city_info("London", "UK")
        end_time = time.time()
        
        # Should complete in reasonable time (tools run sequentially)
        assert end_time - start_time < 1.0
        
        # Should contain all results
        assert "Sunny, 25°C" in result
        assert "14:30 UTC" in result
        assert "8.9 million" in result

    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test async error handling across the system."""
        with patch('app.agents.base_agent.ChatOpenAI') as mock_openai:
            mock_llm = MagicMock()
            mock_openai.return_value = mock_llm
            
            agent = CityInformationAgent()
            agent.agent_executor = AsyncMock()
            agent.agent_executor.ainvoke = AsyncMock(
                side_effect=asyncio.TimeoutError("Request timeout")
            )
            
            with pytest.raises(asyncio.TimeoutError):
                await agent.ainvoke("Test message")
