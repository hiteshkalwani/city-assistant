"""
Test cases for the City Information Assistant tools.

This module contains tests for all tool implementations including
weather, time, and city facts tools.
"""

import pytest

from app.tools.weather_tool import WeatherTool, WeatherInput, WeatherOutput
from app.tools.time_tool import TimeTool, TimeOutput
from app.tools.facts_tool import CityFactsTool, CityFactsOutput


class TestWeatherTool:
    """Test cases for the WeatherTool."""

    @pytest.fixture
    def weather_tool(self):
        """Create a WeatherTool instance for testing."""
        return WeatherTool()

    @pytest.mark.asyncio
    async def test_weather_tool_with_mock_data(self, weather_tool):
        """Test weather tool with mock data when API key is not available."""
        # Temporarily remove API key to force mock data
        original_api_key = weather_tool.api_key
        weather_tool.api_key = None

        result = await weather_tool._arun("London", "UK")

        assert isinstance(result, str)
        assert "London" in result
        assert "Temperature:" in result
        assert "°C" in result

        # Restore original API key
        weather_tool.api_key = original_api_key

    @pytest.mark.asyncio
    async def test_weather_tool_input_validation(self, weather_tool):
        """Test weather tool input validation."""
        # Test valid input
        valid_input = WeatherInput(city="Paris", country="France")
        assert valid_input.city == "Paris"
        assert valid_input.country == "France"

        # Test input without country
        input_no_country = WeatherInput(city="Tokyo")
        assert input_no_country.city == "Tokyo"
        assert input_no_country.country is None

    def test_weather_output_natural_language(self):
        """Test weather output natural language conversion."""
        weather_data = WeatherOutput(
            city="London",
            country="UK",
            temperature=20.5,
            feels_like=22.0,
            humidity=65,
            pressure=1013,
            description="partly cloudy",
            wind_speed=3.2,
            wind_direction=180,
            visibility=10000
        )

        result = weather_data.to_natural_language()

        assert "London, UK" in result
        assert "20.5°C" in result
        assert "68.9°F" in result  # Fahrenheit conversion
        assert "partly cloudy" in result.lower()
        assert "65%" in result
        assert "3.2 m/s" in result


class TestTimeTool:
    """Test cases for the TimeTool."""

    @pytest.fixture
    def time_tool(self):
        """Create a TimeTool instance for testing."""
        return TimeTool()

    @pytest.mark.asyncio
    async def test_time_tool_with_known_city(self, time_tool):
        """Test time tool with a well-known city."""
        result = await time_tool._arun("London", "UK")

        assert isinstance(result, str)
        assert "London" in result
        assert "Current time" in result
        assert "Timezone:" in result

    @pytest.mark.asyncio
    async def test_time_tool_timezone_mapping(self, time_tool):
        """Test timezone mapping for major cities."""
        # Test known city mappings
        assert time_tool._get_timezone_for_city("london") == "Europe/London"
        assert time_tool._get_timezone_for_city("tokyo") == "Asia/Tokyo"
        assert time_tool._get_timezone_for_city("new york") == "America/New_York"

        # Test country fallback
        assert time_tool._get_timezone_for_city("unknown_city", "uk") == "Europe/London"
        assert time_tool._get_timezone_for_city("unknown_city", "japan") == "Asia/Tokyo"

        # Test default fallback
        assert time_tool._get_timezone_for_city("unknown_city", "unknown_country") == "UTC"

    def test_time_output_natural_language(self):
        """Test time output natural language conversion."""
        time_data = TimeOutput(
            city="Tokyo",
            timezone="Asia/Tokyo",
            current_time="2024-01-15 14:30:00",
            utc_offset="+09:00",
            is_dst=False
        )

        result = time_data.to_natural_language()

        assert "Tokyo" in result
        assert "14:30:00" in result
        assert "Asia/Tokyo" in result
        assert "UTC+09:00" in result
        assert "Daylight Saving Time" not in result  # DST is False


class TestCityFactsTool:
    """Test cases for the CityFactsTool."""

    @pytest.fixture
    def facts_tool(self):
        """Create a CityFactsTool instance for testing."""
        return CityFactsTool()

    @pytest.mark.asyncio
    async def test_facts_tool_with_mock_data(self, facts_tool):
        """Test facts tool with mock data."""
        result = await facts_tool._arun("London", "UK")

        assert isinstance(result, str)
        assert "London" in result
        assert "Population:" in result or "temporarily unavailable" in result

    @pytest.mark.asyncio
    async def test_facts_tool_unknown_city(self, facts_tool):
        """Test facts tool with unknown city."""
        result = await facts_tool._arun("UnknownCity", "UnknownCountry")

        assert isinstance(result, str)
        assert "UnknownCity" in result

    def test_facts_output_natural_language(self):
        """Test facts output natural language conversion."""
        facts_data = CityFactsOutput(
            city="Paris",
            country="France",
            population=2161000,
            region="Île-de-France",
            latitude=48.8566,
            longitude=2.3522,
            elevation=35,
            timezone="Europe/Paris",
            founded="3rd century BC",
            area=105.4,
            facts=[
                "Known as the City of Light",
                "Home to the Eiffel Tower",
                "Capital of France"
            ]
        )

        result = facts_data.to_natural_language()

        assert "Paris, France" in result
        assert "2,161,000" in result  # Formatted population
        assert "48.8566, 2.3522" in result  # Coordinates
        assert "City of Light" in result
        assert "Eiffel Tower" in result
        assert "Capital of France" in result


class TestToolIntegration:
    """Integration tests for all tools working together."""

    @pytest.mark.asyncio
    async def test_all_tools_for_city(self):
        """Test all tools working for the same city."""
        city = "Tokyo"
        country = "Japan"

        # Test weather tool
        weather_tool = WeatherTool()
        weather_result = await weather_tool._arun(city, country)
        assert isinstance(weather_result, str)
        assert city in weather_result

        # Test time tool
        time_tool = TimeTool()
        time_result = await time_tool._arun(city, country)
        assert isinstance(time_result, str)
        assert city in time_result

        # Test facts tool
        facts_tool = CityFactsTool()
        facts_result = await facts_tool._arun(city, country)
        assert isinstance(facts_result, str)
        assert city in facts_result

    @pytest.mark.asyncio
    async def test_tools_error_handling(self):
        """Test error handling across all tools."""
        city = "TestCity"
        country = "TestCountry"

        # All tools should handle unknown cities gracefully
        weather_tool = WeatherTool()
        time_tool = TimeTool()
        facts_tool = CityFactsTool()

        # These should not raise exceptions
        weather_result = await weather_tool._arun(city, country)
        time_result = await time_tool._arun(city, country)
        facts_result = await facts_tool._arun(city, country)

        # All should return strings
        assert isinstance(weather_result, str)
        assert isinstance(time_result, str)
        assert isinstance(facts_result, str)

        # All should contain the city name
        assert city in weather_result
        assert city in time_result
        assert city in facts_result
