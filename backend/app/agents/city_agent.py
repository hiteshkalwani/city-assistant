"""
City Information Agent for the City Assistant.

This agent orchestrates multiple tools to provide comprehensive
information about cities including weather, time, and facts.
"""

from typing import List

from langchain_core.tools import BaseTool
from loguru import logger

from app.agents.base_agent import BaseAgent
from app.tools.weather_tool import WeatherTool
from app.tools.time_tool import TimeTool
from app.tools.facts_tool import CityFactsTool
from app.tools.plan_visit_tool import PlanMyCityVisitTool


class CityInformationAgent(BaseAgent):
    """Agent specialized in providing city information."""
    
    def _get_tools(self) -> List[BaseTool]:
        """Get the tools for the city information agent.
        
        Returns:
            List of tools for city information.
        """
        tools = [
            WeatherTool(),
            TimeTool(),
            CityFactsTool(),
            PlanMyCityVisitTool(),  # Composite tool for comprehensive planning
        ]
        
        logger.info(f"Initialized {len(tools)} tools for CityInformationAgent")
        return tools
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the city information agent.
        
        Returns:
            The system prompt for the agent.
        """
        return """You are a helpful and knowledgeable City Information Assistant. Your role is to provide accurate, comprehensive, and engaging information about cities around the world.

You have access to the following tools:
1. **get_weather**: Get current weather conditions, temperature, humidity, wind, and other meteorological data
2. **get_time**: Get current time, timezone information, and daylight saving time status
3. **get_city_facts**: Get detailed information about cities including population, coordinates, elevation, history, and interesting facts
4. **plan_city_visit**: Comprehensive city visit planning tool that orchestrates all other tools to provide complete travel information

## Guidelines:

### Communication Style:
- Be friendly, helpful, and enthusiastic about sharing city information
- Use emojis and formatting to make responses more engaging and readable
- Provide context and explain why information might be useful
- If you're unsure about something, be honest rather than making up information

### Tool Usage:
- Always use the appropriate tools to get current, accurate information
- For comprehensive city planning, use the **plan_city_visit** tool which provides structured output with thinking, function calls, and complete responses
- For specific queries (weather only, time only, facts only), use the individual tools
- You can use multiple tools in a single response to provide comprehensive information

### Structured Responses:
- When using the plan_city_visit tool, the response will include:
  - **Thinking**: Explanation of what you're doing and why
  - **Function Calls**: List of tools called with their parameters and results
  - **Response**: Comprehensive final answer with all gathered information

### Error Handling:
- If a tool returns an error or mock data, acknowledge this to the user
- Provide whatever information is available and suggest trying again later
- Always try to be helpful even when some services are unavailable

### Context Awareness:
- Remember the conversation history and build upon previous interactions
- If a user mentions a city, you can proactively offer related information
- Consider the user's likely intent and provide relevant details

### Examples of Good Responses:
- For "Plan my visit to Paris": Use plan_city_visit tool for comprehensive information
- For "What's the weather in London?": Use get_weather tool for specific weather data
- For "What time is it in Tokyo?": Use get_time tool for time information
- For "Tell me about Rome": Use get_city_facts tool for city information

Remember: Your goal is to be the most helpful and informative city assistant possible. Use your tools wisely and provide engaging, accurate information that helps users learn about and connect with cities around the world."""
    
    async def get_comprehensive_city_info(
        self,
        city: str,
        country: str = None,
        include_weather: bool = True,
        include_time: bool = True,
        include_facts: bool = True,
    ) -> str:
        """Get comprehensive information about a city using all available tools.
        
        Args:
            city: Name of the city
            country: Optional country to disambiguate
            include_weather: Whether to include weather information
            include_time: Whether to include time information
            include_facts: Whether to include city facts
            
        Returns:
            Comprehensive city information as a formatted string
        """
        logger.info(f"Getting comprehensive info for {city}, {country}")
        
        results = []
        
        # Get city facts
        if include_facts:
            try:
                facts_tool = CityFactsTool()
                facts = await facts_tool._arun(city, country)
                results.append(f"## 🏙️ City Information\n{facts}")
            except Exception as e:
                logger.error(f"Failed to get city facts: {e}")
                results.append("❌ City facts temporarily unavailable")
        
        # Get current weather
        if include_weather:
            try:
                weather_tool = WeatherTool()
                weather = await weather_tool._arun(city, country)
                results.append(f"## 🌤️ Current Weather\n{weather}")
            except Exception as e:
                logger.error(f"Failed to get weather: {e}")
                results.append("❌ Weather information temporarily unavailable")
        
        # Get current time
        if include_time:
            try:
                time_tool = TimeTool()
                time_info = await time_tool._arun(city, country)
                results.append(f"## ⏰ Current Time\n{time_info}")
            except Exception as e:
                logger.error(f"Failed to get time: {e}")
                results.append("❌ Time information temporarily unavailable")
        
        return "\n\n".join(results)
