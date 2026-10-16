"""
Weather tool for getting current weather information.

This tool integrates with OpenWeatherMap API to provide current weather
conditions for any city worldwide.
"""

from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.cache_decorator import cache_api_call
from app.utils.exceptions import ExternalAPIError, ToolExecutionError
from app.utils.http_client import HTTPClient


class WeatherInput(BaseModel):
    """Input schema for the weather tool."""

    city: str = Field(
        ...,
        description="The name of the city to get weather for",
        example="London"
    )
    country: Optional[str] = Field(
        None,
        description="Optional country code to disambiguate the city (e.g., 'US', 'UK')",
        example="UK"
    )


class WeatherOutput(BaseModel):
    """Output schema for the weather tool."""

    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country name")
    temperature: float = Field(..., description="Current temperature in Celsius")
    feels_like: float = Field(..., description="Feels like temperature in Celsius")
    humidity: int = Field(..., description="Humidity percentage")
    pressure: int = Field(..., description="Atmospheric pressure in hPa")
    description: str = Field(..., description="Weather description")
    wind_speed: float = Field(..., description="Wind speed in m/s")
    wind_direction: Optional[int] = Field(None, description="Wind direction in degrees")
    visibility: Optional[int] = Field(None, description="Visibility in meters")

    def to_natural_language(self) -> str:
        """Convert weather data to natural language description."""
        temp_c = self.temperature
        temp_f = (temp_c * 9/5) + 32

        description = (
            f"Current weather in {self.city}, {self.country}:\n"
            f"🌡️ Temperature: {temp_c:.1f}°C ({temp_f:.1f}°F)\n"
            f"🤔 Feels like: {self.feels_like:.1f}°C\n"
            f"☁️ Conditions: {self.description.title()}\n"
            f"💧 Humidity: {self.humidity}%\n"
            f"🌬️ Wind: {self.wind_speed:.1f} m/s"
        )

        if self.wind_direction is not None:
            # Convert degrees to cardinal direction
            directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            direction = directions[int((self.wind_direction + 11.25) / 22.5) % 16]
            description += f" from {direction}"

        description += f"\n🔍 Pressure: {self.pressure} hPa"

        if self.visibility:
            visibility_km = self.visibility / 1000
            description += f"\n👁️ Visibility: {visibility_km:.1f} km"

        return description


class WeatherTool(BaseTool):
    """Tool for getting current weather information."""

    name: str = "get_weather"
    description: str = (
        "Get current weather information for a city. "
        "Useful for answering questions about current weather conditions, "
        "temperature, humidity, wind, and other meteorological data."
    )
    args_schema: Type[BaseModel] = WeatherInput
    return_direct: bool = False

    # Define API configuration as class attributes
    api_key: Optional[str] = None
    base_url: str = "https://api.openweathermap.org/data/2.5"

    def __init__(self, **kwargs):
        """Initialize the weather tool."""
        super().__init__(**kwargs)
        # Set the API key from settings
        if not self.api_key:
            self.api_key = settings.OPENWEATHER_API_KEY

    def _get_mock_weather(self, city: str, country: Optional[str] = None) -> WeatherOutput:
        """Get mock weather data when API is not available."""
        logger.warning(f"Using mock weather data for {city}")

        return WeatherOutput(
            city=city,
            country=country or "Unknown",
            temperature=20.5,
            feels_like=22.0,
            humidity=65,
            pressure=1013,
            description="partly cloudy",
            wind_speed=3.2,
            wind_direction=180,
            visibility=10000,
        )

    @cache_api_call(ttl=1800, key_prefix="weather")  # Cache for 30 minutes
    async def _arun(
        self,
        city: str,
        country: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously get weather information."""
        try:
            # Check if API key is available
            if not self.api_key:
                logger.warning("OpenWeatherMap API key not configured, using mock data")
                weather_data = self._get_mock_weather(city, country)
                return weather_data.to_natural_language()

            # Prepare the query
            query = city
            if country:
                query += f",{country}"

            # Make the API request
            async with HTTPClient(base_url=self.base_url) as client:
                response = await client.get(
                    "/weather",
                    params={
                        "q": query,
                        "appid": self.api_key,
                        "units": "metric",  # Use Celsius
                    }
                )

                data = response.json()

                # Parse the response
                weather_data = WeatherOutput(
                    city=data["name"],
                    country=data["sys"]["country"],
                    temperature=data["main"]["temp"],
                    feels_like=data["main"]["feels_like"],
                    humidity=data["main"]["humidity"],
                    pressure=data["main"]["pressure"],
                    description=data["weather"][0]["description"],
                    wind_speed=data["wind"]["speed"],
                    wind_direction=data["wind"].get("deg"),
                    visibility=data.get("visibility"),
                )

                logger.info(f"Retrieved weather data for {city}")
                return weather_data.to_natural_language()

        except ExternalAPIError as e:
            logger.error(f"Weather API error: {e.message}")
            # Fall back to mock data
            weather_data = self._get_mock_weather(city, country)
            return f"⚠️ Weather service temporarily unavailable. Here's sample data:\n\n{weather_data.to_natural_language()}"

        except Exception as e:
            error_msg = f"Failed to get weather for {city}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ToolExecutionError(
                tool_name=self.name,
                message=error_msg,
                details={"city": city, "country": country}
            ) from e

    def _run(
        self,
        city: str,
        country: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously get weather information (not implemented)."""
        raise NotImplementedError("Use async version (_arun) instead")
