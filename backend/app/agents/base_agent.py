"""
Base agent class for the City Information Assistant.

This module provides the base class for all agents in the system,
implementing common functionality and patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from loguru import logger

from app.core.config import settings
from app.core.observability import tracer


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(
        self,
        model_name: str = settings.OPENAI_MODEL,
        temperature: float = settings.OPENAI_TEMPERATURE,
        max_tokens: int = settings.OPENAI_MAX_TOKENS,
        memory_window: int = 10,
        verbose: bool = False,
    ):
        """Initialize the base agent.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum number of tokens to generate
            memory_window: Number of messages to keep in memory
            verbose: Whether to enable verbose logging
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.verbose = verbose
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Initialize chat history
        self.chat_history: BaseChatMessageHistory = InMemoryChatMessageHistory()
        
        # Initialize tools and agent
        self.tools = self._get_tools()
        self.agent_executor = self._create_agent()
        
        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")
    
    @abstractmethod
    def _get_tools(self) -> List[BaseTool]:
        """Get the tools for this agent.
        
        Returns:
            List of tools to be used by the agent.
        """
        pass
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent.
        
        Returns:
            The system prompt string.
        """
        pass
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor.
        
        Returns:
            Configured agent executor.
        """
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        # Create the agent executor without memory parameter
        # We'll handle chat history manually in the ainvoke method
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )
    
    async def ainvoke(
        self,
        input_data: Union[str, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Asynchronously invoke the agent.
        
        Args:
            input_data: Input to the agent (string or dict)
            config: Optional LangChain run config (e.g. callbacks for token usage)
            **kwargs: Additional arguments
            
        Returns:
            Agent response with output and intermediate steps
        """
        # Prepare input
        if isinstance(input_data, str):
            agent_input = {"input": input_data}
        else:
            agent_input = input_data.copy()
        
        # Add any additional context
        agent_input.update(kwargs)
        
        # Ensure chat_history is provided (empty list if not provided)
        if "chat_history" not in agent_input:
            agent_input["chat_history"] = []
        
        # Trace the agent execution
        if tracer.enabled:
            tracer.trace_run(
                name=f"agent_{self.__class__.__name__}",
                inputs=agent_input,
                run_type="chain",
            )
        
        try:
            # Execute the agent
            result = await self.agent_executor.ainvoke(agent_input, config=config)
            return result
            
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}", exc_info=True)
            raise
    
    async def astream(
        self,
        input_data: Union[str, Dict[str, Any]],
        **kwargs,
    ):
        """Stream the agent's response.
        
        Args:
            input_data: Input to the agent (string or dict)
            **kwargs: Additional arguments
            
        Yields:
            Chunks of the agent's response
        """
        # Prepare input
        if isinstance(input_data, str):
            agent_input = {"input": input_data}
        else:
            agent_input = input_data.copy()
        
        # Add any additional context
        agent_input.update(kwargs)
        
        # Stream the response
        async for chunk in self.agent_executor.astream(agent_input):
            yield chunk
    
    def clear_memory(self) -> None:
        """Clear the agent's conversation memory."""
        self.chat_history.clear()
        logger.info("Agent memory cleared")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's memory.
        
        Returns:
            Dictionary containing memory statistics
        """
        messages = self.chat_history.messages
        return {
            "total_messages": len(messages),
            "memory_window": self.memory_window,
            "last_message": messages[-1].content if messages else None,
        }
