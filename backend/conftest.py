"""
Root conftest.py for pytest configuration.
Ensures proper async test support across all test modules.
"""

import os
import pytest
import asyncio
import sys
from pathlib import Path

# Tests must never need real credentials; set before app settings are imported.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
os.environ.setdefault("CACHE_ENABLED", "false")

# Ensure pytest-asyncio is available
pytest_plugins = ['pytest_asyncio']

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
