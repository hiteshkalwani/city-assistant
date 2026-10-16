"""
Composite tool for planning city visits.

This tool orchestrates multiple tools to provide comprehensive
city visit planning information including facts, weather, and time.
"""

import json
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field

from app.tools.weather_tool import WeatherTool
from app.tools.time_tool import TimeTool
from app.tools.facts_tool import CityFactsTool


class PlanVisitInput(BaseModel):
    """Input schema for the plan visit tool."""

    city: str = Field(
        ...,
        description="The name of the city to plan a visit for",
        example="Paris"
    )
    country: Optional[str] = Field(
        None,
        description="Optional country code to disambiguate the city",
        example="France"
    )


class ToolCall(BaseModel):
    """Model for representing a tool call."""

    tool: str = Field(..., description="Name of the tool")
    parameters: Dict[str, Any] = Field(..., description="Parameters passed to the tool")
    result: str = Field(..., description="Result from the tool")


class PlanVisitOutput(BaseModel):
    """Output schema for the plan visit tool."""

    thinking: str = Field(..., description="Reasoning about what the agent is doing")
    function_calls: List[ToolCall] = Field(..., description="List of tool calls made")
    response: str = Field(..., description="Final comprehensive response")

    def to_json(self) -> str:
        """Convert to JSON string for API responses."""
        return json.dumps(self.model_dump(), indent=2)


class PlanMyCityVisitTool(BaseTool):
    """Composite tool for planning city visits using multiple information sources."""

    name: str = "plan_city_visit"
    description: str = (
        "Plan a comprehensive city visit by gathering information about the city's "
        "facts, current weather, and local time. This tool orchestrates multiple "
        "data sources to provide a complete overview for trip planning."
    )
    args_schema: Type[BaseModel] = PlanVisitInput
    return_direct: bool = False

    # Define sub-tools as class attributes
    weather_tool: Optional[WeatherTool] = None
    time_tool: Optional[TimeTool] = None
    facts_tool: Optional[CityFactsTool] = None

    def __init__(self, **kwargs):
        """Initialize the composite tool with individual tools."""
        super().__init__(**kwargs)
        # Initialize sub-tools if not already set
        if not self.weather_tool:
            self.weather_tool = WeatherTool()
        if not self.time_tool:
            self.time_tool = TimeTool()
        if not self.facts_tool:
            self.facts_tool = CityFactsTool()

    async def _arun(
        self,
        city: str,
        country: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously plan a city visit using multiple tools."""
        try:
            # Initialize the output structure
            thinking = f"To help you plan your visit to {city}, I'll gather comprehensive information by checking city facts, current weather conditions, and local time."
            function_calls = []
            responses = []

            logger.info(f"Planning visit for {city}, {country}")

            # Step 1: Get city facts
            try:
                logger.info(f"Getting city facts for {city}")
                facts_result = await self.facts_tool._arun(city, country)
                function_calls.append(ToolCall(
                    tool="CityFactsTool",
                    parameters={"city": city, "country": country},
                    result=facts_result
                ))
                responses.append(f"📍 **City Information:**\n{facts_result}")
                logger.info(f"Successfully retrieved city facts for {city}")
            except Exception as e:
                logger.error(f"Failed to get city facts: {e}")
                function_calls.append(ToolCall(
                    tool="CityFactsTool",
                    parameters={"city": city, "country": country},
                    result=f"Error: {str(e)}"
                ))
                responses.append("📍 **City Information:** Currently unavailable")

            # Step 2: Get current weather
            try:
                logger.info(f"Getting weather for {city}")
                weather_result = await self.weather_tool._arun(city, country)
                function_calls.append(ToolCall(
                    tool="WeatherTool",
                    parameters={"city": city, "country": country},
                    result=weather_result
                ))
                responses.append(f"🌤️ **Current Weather:**\n{weather_result}")
                logger.info(f"Successfully retrieved weather for {city}")
            except Exception as e:
                logger.error(f"Failed to get weather: {e}")
                function_calls.append(ToolCall(
                    tool="WeatherTool",
                    parameters={"city": city, "country": country},
                    result=f"Error: {str(e)}"
                ))
                responses.append("🌤️ **Current Weather:** Currently unavailable")

            # Step 3: Get local time
            try:
                logger.info(f"Getting time for {city}")
                time_result = await self.time_tool._arun(city, country)
                function_calls.append(ToolCall(
                    tool="TimeTool",
                    parameters={"city": city, "country": country},
                    result=time_result
                ))
                responses.append(f"⏰ **Local Time:**\n{time_result}")
                logger.info(f"Successfully retrieved time for {city}")
            except Exception as e:
                logger.error(f"Failed to get time: {e}")
                function_calls.append(ToolCall(
                    tool="TimeTool",
                    parameters={"city": city, "country": country},
                    result=f"Error: {str(e)}"
                ))
                responses.append("⏰ **Local Time:** Currently unavailable")

            # Combine all responses
            combined_response = "\n\n".join(responses)

            # Add visit planning suggestions
            visit_suggestions = self._generate_visit_suggestions(city, country)
            final_response = f"{combined_response}\n\n{visit_suggestions}"

            # Create the structured output
            output = PlanVisitOutput(
                thinking=thinking,
                function_calls=function_calls,
                response=final_response
            )

            logger.info(f"Successfully planned visit for {city}")
            return output.to_json()

        except Exception as e:
            error_msg = f"Failed to plan visit for {city}: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Return error in the expected format
            error_output = PlanVisitOutput(
                thinking=f"I encountered an error while planning your visit to {city}.",
                function_calls=[],
                response=f"I'm sorry, but I couldn't gather complete information for {city} at this time. Please try again later."
            )

            return error_output.to_json()

    def _generate_visit_suggestions(self, city: str, country: Optional[str] = None) -> str:
        """Generate visit planning suggestions based on the city."""
        suggestions = [
            f"🎯 **Visit Planning Tips for {city}:**",
            "• Check local events and festivals happening during your visit",
            "• Research popular attractions and book tickets in advance",
            "• Consider the weather when packing and planning outdoor activities",
            "• Look into local transportation options and city passes",
            "• Try local cuisine and visit recommended restaurants",
            "• Learn a few basic phrases in the local language",
            "• Check visa requirements and travel advisories",
            "",
            "💡 **Pro Tip:** Use this information to plan your itinerary and make the most of your visit!"
        ]

        return "\n".join(suggestions)

    def _run(
        self,
        city: str,
        country: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously plan a city visit (not implemented)."""
        raise NotImplementedError("Use async version (_arun) instead")
