"""
City facts tool for getting information about cities.

This tool integrates with GeoDB API and Wikipedia to provide
interesting facts and information about cities worldwide.
"""

from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.cache_decorator import cache_api_call
from app.utils.exceptions import ToolExecutionError
from app.utils.http_client import HTTPClient


class CityFactsInput(BaseModel):
    """Input schema for the city facts tool."""

    city: str = Field(
        ...,
        description="The name of the city to get facts about",
        example="London"
    )
    country: Optional[str] = Field(
        default=None,
        description="Optional country code to disambiguate the city",
        example="UK"
    )


class CityFactsOutput(BaseModel):
    """Output schema for the city facts tool."""

    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country name")
    population: Optional[int] = Field(None, description="City population")
    region: Optional[str] = Field(None, description="Administrative region/state")
    latitude: Optional[float] = Field(None, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, description="Longitude coordinate")
    elevation: Optional[int] = Field(None, description="Elevation in meters")
    timezone: Optional[str] = Field(None, description="Timezone identifier")
    founded: Optional[str] = Field(None, description="When the city was founded")
    area: Optional[float] = Field(None, description="City area in square kilometers")
    facts: List[str] = Field(default_factory=list, description="Interesting facts about the city")

    def to_natural_language(self) -> str:
        """Convert city facts to natural language description."""
        description = f"🏙️ **{self.city}, {self.country}**\n\n"

        if self.population:
            description += f"👥 Population: {self.population:,}\n"

        if self.region:
            description += f"📍 Region: {self.region}\n"

        if self.latitude and self.longitude:
            description += f"🌍 Coordinates: {self.latitude:.4f}, {self.longitude:.4f}\n"

        if self.elevation:
            description += f"⛰️ Elevation: {self.elevation}m above sea level\n"

        if self.area:
            description += f"📏 Area: {self.area:.1f} km²\n"

        if self.founded:
            description += f"🏛️ Founded: {self.founded}\n"

        if self.timezone:
            description += f"🕐 Timezone: {self.timezone}\n"

        if self.facts:
            description += "\n✨ **Interesting Facts:**\n"
            for i, fact in enumerate(self.facts, 1):
                description += f"{i}. {fact}\n"

        return description.strip()


class CityFactsTool(BaseTool):
    """Tool for getting city facts and information."""

    name: str = "get_city_facts"
    description: str = (
        "Get detailed information and interesting facts about a city. "
        "Provides data like population, coordinates, elevation, founding date, "
        "and interesting historical or cultural facts."
    )
    args_schema: Type[BaseModel] = CityFactsInput
    return_direct: bool = False

    # Define API configuration as class attributes
    geodb_api_key: Optional[str] = None
    geodb_host: str = "wft-geo-db.p.rapidapi.com"
    geodb_base_url: str = None

    def __init__(self, **kwargs):
        """Initialize the city facts tool."""
        super().__init__(**kwargs)
        # Set the API configuration from settings
        if not self.geodb_api_key:
            self.geodb_api_key = settings.GEODB_API_KEY
        if not self.geodb_host:
            self.geodb_host = settings.GEODB_API_HOST
        if not self.geodb_base_url:
            self.geodb_base_url = f"https://{self.geodb_host}"

    def _get_mock_facts(self, city: str, country: Optional[str] = None) -> CityFactsOutput:
        """Get mock city facts when APIs are not available."""
        logger.warning(f"Using mock city facts for {city}")

        # Mock data for common cities
        mock_data = {
            "london": {
                "country": "United Kingdom",
                "population": 8982000,
                "region": "Greater London",
                "latitude": 51.5074,
                "longitude": -0.1278,
                "elevation": 35,
                "timezone": "Europe/London",
                "founded": "43 AD (as Londinium)",
                "area": 1572.0,
                "facts": [
                    "London is home to over 8 million people, making it the largest city in the UK",
                    "The city has over 170 museums and more than 11,000 listed buildings",
                    "London's Underground is the oldest subway system in the world, opened in 1863",
                    "The city is built on the River Thames, which flows for 215 miles through southern England",
                    "London has been the capital of England for nearly 1,000 years"
                ]
            },
            "paris": {
                "country": "France",
                "population": 2161000,
                "region": "Île-de-France",
                "latitude": 48.8566,
                "longitude": 2.3522,
                "elevation": 35,
                "timezone": "Europe/Paris",
                "founded": "3rd century BC",
                "area": 105.4,
                "facts": [
                    "Paris is known as the 'City of Light' due to its early adoption of street lighting",
                    "The Eiffel Tower was built for the 1889 World's Fair and was initially criticized",
                    "Paris has 20 administrative districts called arrondissements",
                    "The Louvre Museum is the world's largest art museum",
                    "Paris is home to over 400 parks and gardens"
                ]
            },
            "tokyo": {
                "country": "Japan",
                "population": 13960000,
                "region": "Kantō",
                "latitude": 35.6762,
                "longitude": 139.6503,
                "elevation": 40,
                "timezone": "Asia/Tokyo",
                "founded": "1457 (as Edo)",
                "area": 2194.0,
                "facts": [
                    "Tokyo is the most populous metropolitan area in the world",
                    "The city was originally called Edo before being renamed Tokyo in 1868",
                    "Tokyo has the world's busiest train stations and most extensive urban rail network",
                    "The city is built on the Kantō Plain and sits on Tokyo Bay",
                    "Tokyo will host the Summer Olympics (postponed from 2020)"
                ]
            }
        }

        city_lower = city.lower()
        if city_lower in mock_data:
            data = mock_data[city_lower]
            return CityFactsOutput(city=city, **data)
        else:
            # Generic mock data for unknown cities
            return CityFactsOutput(
                city=city,
                country=country or "Unknown",
                population=500000,
                region="Unknown Region",
                latitude=0.0,
                longitude=0.0,
                elevation=100,
                timezone="UTC",
                founded="Unknown",
                area=100.0,
                facts=[
                    f"{city} is a city with rich history and culture",
                    "The city offers many attractions for visitors",
                    f"{city} has a unique local cuisine and traditions"
                ]
            )

    async def _search_city_geodb(self, city: str, country: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Search for city information using GeoDB API."""
        if not self.geodb_api_key:
            return None

        try:
            query = city
            if country:
                query += f", {country}"

            async with HTTPClient(base_url=self.geodb_base_url) as client:
                response = await client.get(
                    "/v1/geo/cities",
                    headers={
                        "X-RapidAPI-Key": self.geodb_api_key,
                        "X-RapidAPI-Host": self.geodb_host,
                    },
                    params={
                        "namePrefix": city,
                        "limit": 5,
                        "offset": 0,
                        "sort": "-population",
                    }
                )

                data = response.json()

                if data.get("data") and len(data["data"]) > 0:
                    # Find the best match
                    for city_data in data["data"]:
                        city_name = city_data.get("name", "").lower()
                        if city.lower() in city_name or city_name in city.lower():
                            return city_data

                    # If no exact match, return the first result
                    return data["data"][0]

                return None

        except Exception as e:
            logger.error(f"GeoDB API error: {str(e)}")
            return None

    def _generate_city_facts(self, city: str, country: str) -> List[str]:
        """Generate interesting facts about a city."""
        facts = [
            f"{city} is located in {country} and has a rich cultural heritage",
            "The city offers various attractions for tourists and locals alike",
            f"{city} has its own unique architectural style and landmarks",
            f"Local cuisine in {city} reflects the regional traditions of {country}",
            f"The city plays an important role in the economy and culture of {country}"
        ]
        return facts

    @cache_api_call(ttl=7200, key_prefix="city_facts")  # Cache for 2 hours
    async def _arun(
        self,
        city: str,
        country: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously get city facts."""
        try:
            # Try to get data from GeoDB API
            city_data = await self._search_city_geodb(city, country)

            if city_data:
                # Parse GeoDB response
                facts_output = CityFactsOutput(
                    city=city_data.get("name", city),
                    country=city_data.get("country", country or "Unknown"),
                    population=city_data.get("population"),
                    region=city_data.get("region"),
                    latitude=city_data.get("latitude"),
                    longitude=city_data.get("longitude"),
                    elevation=city_data.get("elevationMeters"),
                    timezone=city_data.get("timezone"),
                    founded=city_data.get("foundingDate"),
                    area=city_data.get("area"),
                    facts=self._generate_city_facts(
                        city_data.get("name", city),
                        city_data.get("country", country or "Unknown")
                    )
                )

                logger.info(f"Retrieved city facts for {city} from GeoDB")
                return facts_output.to_natural_language()
            else:
                # Fall back to mock data
                logger.warning(f"No GeoDB data found for {city}, using mock data")
                facts_output = self._get_mock_facts(city, country)
                return f"⚠️ City database temporarily unavailable. Here's general information:\n\n{facts_output.to_natural_language()}"

        except Exception as e:
            error_msg = f"Failed to get city facts for {city}: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Try to provide mock data as fallback
            try:
                facts_output = self._get_mock_facts(city, country)
                return f"⚠️ City facts service error. Here's general information:\n\n{facts_output.to_natural_language()}"
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
        """Synchronously get city facts (not implemented)."""
        raise NotImplementedError("Use async version (_arun) instead")
