"""
Time tool for getting current time information.

This tool integrates with WorldTimeAPI to provide current time
and timezone information for any city worldwide.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field

from app.core.cache_decorator import cache_api_call
from app.utils.exceptions import ExternalAPIError, ToolExecutionError
from app.utils.http_client import HTTPClient


class TimeInput(BaseModel):
    """Input schema for the time tool."""
    
    city: str = Field(
        ...,
        description="The name of the city to get time for",
        example="London"
    )
    country: Optional[str] = Field(
        None,
        description="Optional country code to help identify the timezone",
        example="UK"
    )


class TimeOutput(BaseModel):
    """Output schema for the time tool."""
    
    city: str = Field(..., description="City name")
    timezone: str = Field(..., description="Timezone identifier")
    current_time: str = Field(..., description="Current time in the timezone")
    utc_offset: str = Field(..., description="UTC offset")
    is_dst: bool = Field(..., description="Whether daylight saving time is active")
    
    def to_natural_language(self) -> str:
        """Convert time data to natural language description."""
        dst_info = " (Daylight Saving Time)" if self.is_dst else ""
        
        return (
            f"🕐 Current time in {self.city}:\n"
            f"⏰ {self.current_time}\n"
            f"🌍 Timezone: {self.timezone} (UTC{self.utc_offset}){dst_info}"
        )


class TimeTool(BaseTool):
    """Tool for getting current time information."""
    
    name: str = "get_time"
    description: str = (
        "Get current time and timezone information for a city. "
        "Useful for answering questions about what time it is in a specific location, "
        "timezone differences, and daylight saving time status."
    )
    args_schema: Type[BaseModel] = TimeInput
    return_direct: bool = False
    
    # Define API configuration as class attributes
    base_url: str = "http://worldtimeapi.org/api"
    city_timezone_map: Dict[str, str] = {}
    
    def __init__(self, **kwargs):
        """Initialize the time tool."""
        super().__init__(**kwargs)
        
        # Initialize timezone mappings if not already set
        if not self.city_timezone_map:
            self.city_timezone_map = {
                "london": "Europe/London",
                "paris": "Europe/Paris",
                "berlin": "Europe/Berlin",
                "rome": "Europe/Rome",
                "madrid": "Europe/Madrid",
                "amsterdam": "Europe/Amsterdam",
                "new york": "America/New_York",
                "los angeles": "America/Los_Angeles",
                "chicago": "America/Chicago",
                "toronto": "America/Toronto",
                "vancouver": "America/Vancouver",
                "tokyo": "Asia/Tokyo",
                "beijing": "Asia/Shanghai",
                "shanghai": "Asia/Shanghai",
                "hong kong": "Asia/Hong_Kong",
                "singapore": "Asia/Singapore",
                "mumbai": "Asia/Kolkata",
                "delhi": "Asia/Kolkata",
                "sydney": "Australia/Sydney",
                "melbourne": "Australia/Melbourne",
                "dubai": "Asia/Dubai",
                "moscow": "Europe/Moscow",
                "istanbul": "Europe/Istanbul",
                "cairo": "Africa/Cairo",
                "lagos": "Africa/Lagos",
                "johannesburg": "Africa/Johannesburg",
                "sao paulo": "America/Sao_Paulo",
                "rio de janeiro": "America/Sao_Paulo",
                "buenos aires": "America/Argentina/Buenos_Aires",
                "mexico city": "America/Mexico_City",
            }
    
    def _get_timezone_for_city(self, city: str, country: Optional[str] = None) -> str:
        """Get timezone identifier for a city."""
        city_lower = city.lower()
        
        # Check direct mapping first
        if city_lower in self.city_timezone_map:
            return self.city_timezone_map[city_lower]
        
        # Try with country-specific mappings
        if country:
            country_lower = country.lower()
            key = f"{city_lower}, {country_lower}"
            if key in self.city_timezone_map:
                return self.city_timezone_map[key]
            
            # Common country-based fallbacks
            country_timezone_map = {
                "us": "America/New_York",
                "usa": "America/New_York", 
                "uk": "Europe/London",
                "gb": "Europe/London",
                "france": "Europe/Paris",
                "germany": "Europe/Berlin",
                "italy": "Europe/Rome",
                "spain": "Europe/Madrid",
                "japan": "Asia/Tokyo",
                "china": "Asia/Shanghai",
                "india": "Asia/Kolkata",
                "australia": "Australia/Sydney",
                "canada": "America/Toronto",
                "brazil": "America/Sao_Paulo",
                "russia": "Europe/Moscow",
            }
            
            if country_lower in country_timezone_map:
                return country_timezone_map[country_lower]
        
        # Default fallback
        return "UTC"
    
    def _get_mock_time(self, city: str, timezone: str) -> TimeOutput:
        """Get mock time data when API is not available."""
        logger.warning(f"Using mock time data for {city}")
        
        # Use current UTC time as base
        now = datetime.utcnow()
        
        return TimeOutput(
            city=city,
            timezone=timezone,
            current_time=now.strftime("%Y-%m-%d %H:%M:%S"),
            utc_offset="+00:00",
            is_dst=False,
        )
    
    @cache_api_call(ttl=300, key_prefix="time")  # Cache for 5 minutes
    async def _arun(
        self,
        city: str,
        country: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously get time information."""
        try:
            # Get timezone for the city
            timezone = self._get_timezone_for_city(city, country)
            
            # Make the API request
            async with HTTPClient(base_url=self.base_url) as client:
                try:
                    response = await client.get(f"/timezone/{timezone}")
                    data = response.json()
                    
                    # Parse the response
                    datetime_str = data["datetime"]
                    # Extract just the date and time part (remove microseconds and timezone info)
                    clean_datetime = datetime_str.split(".")[0].replace("T", " ")
                    
                    time_data = TimeOutput(
                        city=city,
                        timezone=data["timezone"],
                        current_time=clean_datetime,
                        utc_offset=data["utc_offset"],
                        is_dst=data["dst"],
                    )
                    
                    logger.info(f"Retrieved time data for {city}")
                    return time_data.to_natural_language()
                    
                except ExternalAPIError:
                    # Fall back to mock data if API fails
                    logger.warning(f"WorldTimeAPI failed for {timezone}, using mock data")
                    time_data = self._get_mock_time(city, timezone)
                    return f"⚠️ Time service temporarily unavailable. Approximate time:\n\n{time_data.to_natural_language()}"
                    
        except Exception as e:
            error_msg = f"Failed to get time for {city}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Try to provide mock data as fallback
            try:
                timezone = self._get_timezone_for_city(city, country)
                time_data = self._get_mock_time(city, timezone)
                return f"⚠️ Time service error. Approximate time:\n\n{time_data.to_natural_language()}"
            except:
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
        """Synchronously get time information (not implemented)."""
        raise NotImplementedError("Use async version (_arun) instead")
