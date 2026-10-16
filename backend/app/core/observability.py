"""
Observability configuration for the application.

This module sets up LangSmith for tracing and monitoring of LLM calls.
"""

import os
from typing import Any, Dict, Optional

from langsmith import Client
from loguru import logger

from app.core.config import settings


def setup_langsmith() -> Optional[Client]:
    """
    Configure LangSmith for tracing and monitoring.
    
    Returns:
        Optional[Client]: Configured LangSmith client if API key is set, else None.
    """
    if not settings.LANGCHAIN_TRACING_V2 or not settings.LANGCHAIN_API_KEY:
        logger.warning(
            "LangSmith tracing is disabled. Set LANGCHAIN_TRACING_V2=true "
            "and LANGCHAIN_API_KEY to enable."
        )
        return None
    
    try:
        # Configure environment variables for LangChain
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
        
        if settings.LANGCHAIN_PROJECT:
            os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        if settings.LANGCHAIN_ENDPOINT:
            os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
        
        # Initialize LangSmith client
        client = Client()
        
        logger.info(
            f"LangSmith tracing initialized. Project: {settings.LANGCHAIN_PROJECT}"
        )
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {str(e)}")
        return None


class LangSmithTracer:
    """Wrapper for LangSmith tracing functionality."""
    
    def __init__(self, client: Optional[Client] = None):
        """Initialize the tracer.
        
        Args:
            client: Optional LangSmith client. If not provided, a new one will be created.
        """
        self.client = client or setup_langsmith()
        self._enabled = self.client is not None
    
    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled
    
    def trace_run(
        self,
        name: str,
        inputs: Dict[str, Any],
        run_type: str = "chain",
        **kwargs
    ) -> None:
        """
        Trace a run in LangSmith.
        
        Args:
            name: Name of the run
            inputs: Inputs to the run
            run_type: Type of the run (e.g., 'llm', 'chain', 'tool')
            **kwargs: Additional arguments to pass to the run
        """
        if not self.enabled:
            return
            
        try:
            self.client.create_run(
                name=name,
                inputs=inputs,
                run_type=run_type,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to trace run in LangSmith: {str(e)}")
    
    def trace_tool_usage(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        **kwargs
    ) -> None:
        """
        Trace a tool usage in LangSmith.
        
        Args:
            tool_name: Name of the tool
            inputs: Inputs to the tool
            outputs: Outputs from the tool
            **kwargs: Additional arguments to pass to the run
        """
        if not self.enabled:
            return
            
        try:
            self.client.create_run(
                name=tool_name,
                inputs=inputs,
                outputs=outputs,
                run_type="tool",
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to trace tool usage in LangSmith: {str(e)}")


# Initialize global tracer
tracer = LangSmithTracer()
