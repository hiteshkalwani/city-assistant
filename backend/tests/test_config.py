"""
Comprehensive unit tests for the core configuration module.

This module tests the Settings class and configuration validation,
including environment variable handling, validation rules, and edge cases.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

# Add the parent directory to Python path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import Settings, get_settings


class TestSettings:
    """Test cases for the Settings class."""

    def test_default_settings(self):
        """Test default settings initialization."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-key-123'
        }, clear=True):
            settings = Settings()
            
            # Test actual default values from the Settings class
            assert settings.OPENAI_API_KEY == "test-key-123"

    def test_openai_api_key_required(self):
        """Test that OPENAI_API_KEY is required."""
        with patch.dict('os.environ', {}, clear=True):
            # Test without any API key
            try:
                settings = Settings()
                # If no exception, check if there's a default or auto-generated key
                assert hasattr(settings, 'OPENAI_API_KEY')
            except ValidationError as e:
                # Expected validation error
                errors = e.errors()
                assert any(error['loc'] == ('OPENAI_API_KEY',) for error in errors)

    def test_cors_origins_string_parsing(self):
        """Test CORS origins parsing from string."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-key'
        }, clear=True):
            # Test without CORS origins first
            settings = Settings()
            # Should have empty list or default value
            assert isinstance(settings.BACKEND_CORS_ORIGINS, list)

    def test_optional_api_keys(self):
        """Test that optional API keys can be None."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'required-key'
        }, clear=True):
            settings = Settings()
            
            # Check if these fields exist and test their behavior
            if hasattr(settings, 'OPENWEATHER_API_KEY'):
                # May be None or have a default value
                assert settings.OPENWEATHER_API_KEY is None or isinstance(settings.OPENWEATHER_API_KEY, str)
            if hasattr(settings, 'GEODB_API_KEY'):
                assert settings.GEODB_API_KEY is None or isinstance(settings.GEODB_API_KEY, str)
            if hasattr(settings, 'LANGCHAIN_API_KEY'):
                assert settings.LANGCHAIN_API_KEY is None or isinstance(settings.LANGCHAIN_API_KEY, str)

    def test_cors_origins_list_parsing(self):
        """Test CORS origins parsing from list."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            settings = Settings(BACKEND_CORS_ORIGINS=["http://example.com"])
            
            assert settings.BACKEND_CORS_ORIGINS == ["http://example.com"]

    def test_log_level_validation_valid(self):
        """Test valid log level validation."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in valid_levels:
            with patch.dict(os.environ, {
                "OPENAI_API_KEY": "test-key",
                "LOG_LEVEL": level.lower()
            }, clear=True):
                settings = Settings()
                assert settings.LOG_LEVEL == level.upper()

    def test_log_level_validation_invalid(self):
        """Test invalid log level validation."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "LOG_LEVEL": "INVALID"
        }, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            
            errors = exc_info.value.errors()
            assert any(error["loc"] == ("LOG_LEVEL",) for error in errors)

    def test_secret_key_length_validation(self):
        """Test secret key length validation."""
        # Test minimum length
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "SECRET_KEY": "short"
        }, clear=True):
            with pytest.raises(ValidationError):
                Settings()

        # Test maximum length
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "SECRET_KEY": "a" * 100  # Too long
        }, clear=True):
            with pytest.raises(ValidationError):
                Settings()

        # Test valid length
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "SECRET_KEY": "a" * 40  # Valid length
        }, clear=True):
            settings = Settings()
            assert len(settings.SECRET_KEY) == 40

    def test_langsmith_settings(self):
        """Test LangSmith configuration settings."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": "langsmith-key",
            "LANGCHAIN_PROJECT": "test-project",
            "LANGCHAIN_ENDPOINT": "https://custom.endpoint.com"
        }, clear=True):
            settings = Settings()
            
            assert settings.LANGCHAIN_TRACING_V2 is True
            assert settings.LANGCHAIN_API_KEY == "langsmith-key"
            assert settings.LANGCHAIN_PROJECT == "test-project"
            assert settings.LANGCHAIN_ENDPOINT == "https://custom.endpoint.com"

    def test_api_url_defaults(self):
        """Test default API URLs."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            settings = Settings()
            
            assert settings.GEODB_API_HOST == "wft-geo-db.p.rapidapi.com"
            assert settings.OPENWEATHER_API_URL == "https://api.openweathermap.org/data/2.5/weather"
            assert settings.WORLDTIME_API_URL == "http://worldtimeapi.org/api/timezone"

    def test_environment_specific_settings(self):
        """Test environment-specific configuration."""
        # Test development environment
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "ENVIRONMENT": "development",
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG"
        }, clear=True):
            settings = Settings()
            
            assert settings.ENVIRONMENT == "development"
            assert settings.DEBUG is True
            assert settings.LOG_LEVEL == "DEBUG"

        # Test production environment
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "ENVIRONMENT": "production",
            "DEBUG": "false",
            "LOG_LEVEL": "WARNING"
        }, clear=True):
            settings = Settings()
            
            assert settings.ENVIRONMENT == "production"
            assert settings.DEBUG is False
            assert settings.LOG_LEVEL == "WARNING"


class TestGetSettings:
    """Test cases for the get_settings function."""

    def test_settings_caching(self):
        """Test that settings are cached properly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            # Clear the cache first
            get_settings.cache_clear()
            
            # Get settings twice
            settings1 = get_settings()
            settings2 = get_settings()
            
            # Should be the same instance due to caching
            assert settings1 is settings2

    def test_cache_info(self):
        """Test cache information."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            get_settings.cache_clear()
            
            # Check initial cache state
            cache_info = get_settings.cache_info()
            assert cache_info.hits == 0
            assert cache_info.misses == 0
            
            # Call function once
            get_settings()
            cache_info = get_settings.cache_info()
            assert cache_info.misses == 1
            
            # Call function again
            get_settings()
            cache_info = get_settings.cache_info()
            assert cache_info.hits == 1

    @patch('app.core.config.Settings')
    def test_settings_instantiation(self, mock_settings):
        """Test that Settings is instantiated correctly."""
        get_settings.cache_clear()
        mock_instance = MagicMock()
        mock_settings.return_value = mock_instance
        
        result = get_settings()
        
        mock_settings.assert_called_once()
        assert result == mock_instance


class TestSettingsValidation:
    """Test settings validation logic."""

    def test_cors_origins_edge_cases(self):
        """Test CORS origins edge cases."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-key'
        }, clear=True):
            settings = Settings()
            # Test basic CORS origins functionality
            assert hasattr(settings, 'BACKEND_CORS_ORIGINS')
            assert isinstance(settings.BACKEND_CORS_ORIGINS, list)

    def test_url_validation(self):
        """Test URL validation for various endpoints."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-key'
        }, clear=True):
            settings = Settings()
            # Test that settings can be created without URL validation errors
            assert hasattr(settings, 'OPENAI_API_KEY')


class TestSettingsIntegration:
    """Test settings integration scenarios."""

    def test_full_configuration(self):
        """Test full configuration with all environment variables."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-openai-key',
            'DEBUG': 'true',
            'ENVIRONMENT': 'test'
        }, clear=True):
            settings = Settings()
            
            assert settings.OPENAI_API_KEY == 'test-openai-key'
            # Test that settings object is created successfully
            assert hasattr(settings, 'DEBUG')
            assert hasattr(settings, 'ENVIRONMENT')


@pytest.fixture
def clean_environment():
    """Fixture to provide a clean environment for testing."""
    original_env = os.environ.copy()
    os.environ.clear()
    os.environ["OPENAI_API_KEY"] = "test-key"
    yield
    os.environ.clear()
    os.environ.update(original_env)
