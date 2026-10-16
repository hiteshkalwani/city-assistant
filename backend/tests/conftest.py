"""
Pytest configuration and shared fixtures for the City Information Assistant tests.

This module provides reusable fixtures and mock utilities for testing
external API integrations and common test scenarios.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Optional
from fastapi.testclient import TestClient

# Add the parent directory to Python path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app.main import app
except ImportError:
    # Create a mock app for testing if import fails
    from fastapi import FastAPI
    app = FastAPI()


FAKE_AGENT_RESULT = {
    "output": "London is a vibrant city with rich history and culture.",
    "intermediate_steps": [],
}


@pytest.fixture
def client():
    """Test client whose agent never calls OpenAI (tests can still patch deeper)."""
    from langchain.agents import AgentExecutor

    with patch.object(AgentExecutor, "ainvoke", AsyncMock(return_value=FAKE_AGENT_RESULT)):
        yield TestClient(app)


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# External API Mock Fixtures

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client with realistic responses."""
    mock_client = MagicMock()
    
    # Mock chat completions
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                role="assistant",
                content="London is a vibrant city with rich history and culture.",
                tool_calls=None
            )
        )
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )
    
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_openweather_response():
    """Mock OpenWeatherMap API response."""
    return {
        "coord": {"lon": -0.1257, "lat": 51.5085},
        "weather": [
            {
                "id": 801,
                "main": "Clouds",
                "description": "few clouds",
                "icon": "02d"
            }
        ],
        "base": "stations",
        "main": {
            "temp": 18.5,
            "feels_like": 19.2,
            "temp_min": 16.8,
            "temp_max": 20.1,
            "pressure": 1013,
            "humidity": 72
        },
        "visibility": 10000,
        "wind": {
            "speed": 3.2,
            "deg": 180
        },
        "clouds": {"all": 20},
        "dt": 1640995200,
        "sys": {
            "type": 2,
            "id": 2019646,
            "country": "GB",
            "sunrise": 1640939847,
            "sunset": 1640968234
        },
        "timezone": 0,
        "id": 2643743,
        "name": "London",
        "cod": 200
    }


@pytest.fixture
def mock_geodb_response():
    """Mock GeoDB API response."""
    return {
        "data": [
            {
                "id": 2643743,
                "wikiDataId": "Q84",
                "type": "CITY",
                "city": "London",
                "name": "London",
                "country": "United Kingdom",
                "countryCode": "GB",
                "region": "England",
                "regionCode": "ENG",
                "latitude": 51.5085,
                "longitude": -0.1257,
                "population": 8982000,
                "elevation": 11,
                "timezone": "Europe/London"
            }
        ],
        "metadata": {
            "currentOffset": 0,
            "totalCount": 1
        }
    }


@pytest.fixture
def mock_worldtime_response():
    """Mock WorldTimeAPI response."""
    return {
        "abbreviation": "GMT",
        "client_ip": "192.168.1.1",
        "datetime": "2024-01-15T14:30:00.123456+00:00",
        "day_of_week": 1,
        "day_of_year": 15,
        "dst": False,
        "dst_from": None,
        "dst_offset": 0,
        "dst_until": None,
        "raw_offset": 0,
        "timezone": "Europe/London",
        "unixtime": 1705329000,
        "utc_datetime": "2024-01-15T14:30:00.123456+00:00",
        "utc_offset": "+00:00",
        "week_number": 3
    }


# HTTP Client Mock Fixtures

@pytest.fixture
def mock_httpx_client():
    """Mock httpx AsyncClient for external API calls."""
    mock_client = AsyncMock()
    
    # Default successful response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_response.raise_for_status.return_value = None
    
    mock_client.get.return_value = mock_response
    mock_client.post.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_requests_session():
    """Mock requests Session for synchronous API calls."""
    mock_session = MagicMock()
    
    # Default successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_response.raise_for_status.return_value = None
    
    mock_session.get.return_value = mock_response
    mock_session.post.return_value = mock_response
    
    return mock_session


# Tool Mock Fixtures

@pytest.fixture
def mock_weather_tool():
    """Mock WeatherTool with realistic responses."""
    mock_tool = AsyncMock()
    mock_tool.name = "get_weather"
    mock_tool.description = "Get current weather information for a city"
    
    async def mock_arun(city: str, country: Optional[str] = None):
        return f"""Current weather in {city}{f', {country}' if country else ''}:
Temperature: 18°C (64°F)
Feels like: 19°C (66°F)
Humidity: 72%
Wind: 12 km/h NW
Conditions: Partly cloudy
Visibility: 10 km"""
    
    mock_tool._arun = mock_arun
    return mock_tool


@pytest.fixture
def mock_time_tool():
    """Mock TimeTool with realistic responses."""
    mock_tool = AsyncMock()
    mock_tool.name = "get_time"
    mock_tool.description = "Get current time information for a city"
    
    async def mock_arun(city: str, country: Optional[str] = None):
        return f"""Current time in {city}{f', {country}' if country else ''}:
Time: 14:30:00
Date: Monday, January 15, 2024
Timezone: Europe/London (GMT+0)
UTC Offset: +00:00
Daylight Saving Time: No"""
    
    mock_tool._arun = mock_arun
    return mock_tool


@pytest.fixture
def mock_facts_tool():
    """Mock CityFactsTool with realistic responses."""
    mock_tool = AsyncMock()
    mock_tool.name = "get_city_facts"
    mock_tool.description = "Get interesting facts and information about a city"
    
    async def mock_arun(city: str, country: Optional[str] = None):
        return f"""Facts about {city}{f', {country}' if country else ''}:
Population: 8,982,000
Area: 1,572 km²
Founded: 43 AD by the Romans
Coordinates: 51.5074° N, 0.1278° W
Elevation: 11 meters above sea level
Time Zone: Europe/London

Interesting Facts:
• Known as "The Big Smoke"
• Home to over 300 languages
• Has 4 UNESCO World Heritage Sites
• Contains 8 royal parks"""
    
    mock_tool._arun = mock_arun
    return mock_tool


# Agent Mock Fixtures

@pytest.fixture
def mock_city_agent():
    """Mock CityInformationAgent with realistic behavior."""
    with patch('app.agents.base_agent.ChatOpenAI') as mock_openai:
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        try:
            from app.agents.city_agent import CityInformationAgent
            agent = CityInformationAgent()
        except ImportError:
            # Create a mock agent if import fails
            agent = MagicMock()
            agent.get_comprehensive_city_info = AsyncMock(return_value="Mock city info")
        
        # Mock the agent executor
        if hasattr(agent, 'agent_executor'):
            agent.agent_executor = AsyncMock()
            agent.agent_executor.ainvoke = AsyncMock(return_value={
                "output": "I'd be happy to help you with information about that city!",
                "intermediate_steps": []
            })
        
        return agent


# Error Response Fixtures

@pytest.fixture
def mock_api_error_responses():
    """Mock various API error responses."""
    return {
        "timeout": {
            "status_code": 408,
            "json": {"error": "Request timeout", "code": "timeout"}
        },
        "rate_limit": {
            "status_code": 429,
            "json": {"error": "Rate limit exceeded", "code": "rate_limit"}
        },
        "unauthorized": {
            "status_code": 401,
            "json": {"error": "Unauthorized", "code": "unauthorized"}
        },
        "not_found": {
            "status_code": 404,
            "json": {"error": "Resource not found", "code": "not_found"}
        },
        "server_error": {
            "status_code": 500,
            "json": {"error": "Internal server error", "code": "server_error"}
        }
    }


# Configuration Fixtures

@pytest.fixture
def mock_settings():
    """Mock application settings for testing."""
    try:
        from app.core.config import Settings
        
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
            return Settings()
    except ImportError:
        # Return a mock settings object if import fails
        mock_settings = MagicMock()
        mock_settings.OPENAI_API_KEY = 'test-openai-key'
        mock_settings.CACHE_ENABLED = False
        mock_settings.DEBUG = True
        mock_settings.ENVIRONMENT = 'test'
        return mock_settings


# Database Fixtures (for future use)

@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    mock_db = MagicMock()
    mock_db.execute.return_value = MagicMock()
    mock_db.fetchall.return_value = []
    mock_db.fetchone.return_value = None
    return mock_db


# Utility Functions for Tests

def create_mock_response(status_code: int, json_data: Dict[str, Any], headers: Optional[Dict[str, str]] = None):
    """Create a mock HTTP response."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_data
    mock_response.headers = headers or {}
    
    if status_code >= 400:
        mock_response.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        mock_response.raise_for_status.return_value = None
    
    return mock_response


def create_async_mock_response(status_code: int, json_data: Dict[str, Any], headers: Optional[Dict[str, str]] = None):
    """Create a mock async HTTP response."""
    mock_response = AsyncMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_data
    mock_response.headers = headers or {}
    
    if status_code >= 400:
        mock_response.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        mock_response.raise_for_status.return_value = None
    
    return mock_response


# Parametrized Test Data

@pytest.fixture
def sample_cities():
    """Sample city data for parametrized tests."""
    return [
        {"name": "London", "country": "UK", "expected_timezone": "Europe/London"},
        {"name": "Paris", "country": "France", "expected_timezone": "Europe/Paris"},
        {"name": "Tokyo", "country": "Japan", "expected_timezone": "Asia/Tokyo"},
        {"name": "New York", "country": "USA", "expected_timezone": "America/New_York"},
        {"name": "Sydney", "country": "Australia", "expected_timezone": "Australia/Sydney"},
    ]


@pytest.fixture
def sample_weather_conditions():
    """Sample weather conditions for testing."""
    return [
        {"condition": "clear", "temp": 25, "description": "Clear sky"},
        {"condition": "cloudy", "temp": 18, "description": "Partly cloudy"},
        {"condition": "rainy", "temp": 15, "description": "Light rain"},
        {"condition": "snowy", "temp": -2, "description": "Snow"},
        {"condition": "stormy", "temp": 20, "description": "Thunderstorm"},
    ]


# Performance Testing Fixtures

@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# Cleanup Fixtures

@pytest.fixture(autouse=True)
def cleanup_environment():
    """Automatically cleanup environment after each test."""
    yield
    # Cleanup code here if needed
    pass


# Async Context Manager for Testing

class AsyncContextManager:
    """Async context manager for testing async operations."""
    
    def __init__(self, return_value=None, exception=None):
        self.return_value = return_value
        self.exception = exception
    
    async def __aenter__(self):
        if self.exception:
            raise self.exception
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_context_manager():
    """Fixture for async context manager testing."""
    return AsyncContextManager
