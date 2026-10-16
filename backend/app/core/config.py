"""
Application configuration management.

This module handles loading and validating configuration settings from environment variables.
"""

import logging
from functools import lru_cache
from typing import List, Optional, Union

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Application settings
    APP_NAME: str = "City Information Assistant"
    DEBUG: bool = Field(False, env="DEBUG")
    ENVIRONMENT: str = Field("production", env="ENVIRONMENT")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    SECRET_KEY: str = Field(
        default="change-this-in-production-using-.env-file",
        min_length=32,
        max_length=64,
        env="SECRET_KEY"
    )

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "City Information Assistant API"

    # CORS settings
    BACKEND_CORS_ORIGINS: List[Union[str, AnyHttpUrl]] = Field(
        default=["http://localhost:3000"],
        env="BACKEND_CORS_ORIGINS"
    )

    # Security
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(11520, env="ACCESS_TOKEN_EXPIRE_MINUTES")  # 8 days

    # Redis settings
    REDIS_URL: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(0, env="REDIS_DB")
    REDIS_MAX_CONNECTIONS: int = Field(10, env="REDIS_MAX_CONNECTIONS")
    REDIS_SOCKET_TIMEOUT: int = Field(5, env="REDIS_SOCKET_TIMEOUT")
    REDIS_SOCKET_CONNECT_TIMEOUT: int = Field(5, env="REDIS_SOCKET_CONNECT_TIMEOUT")

    # Cache settings
    CACHE_TTL: int = Field(3600, env="CACHE_TTL")  # 1 hour default
    CACHE_ENABLED: bool = Field(True, env="CACHE_ENABLED")

    # OpenAI settings
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field("gpt-3.5-turbo", env="OPENAI_MODEL")
    OPENAI_TEMPERATURE: float = Field(0.2, env="OPENAI_TEMPERATURE")
    OPENAI_MAX_TOKENS: int = Field(1024, env="OPENAI_MAX_TOKENS")

    # API Keys for external services
    OPENWEATHER_API_KEY: Optional[str] = Field(None, env="OPENWEATHER_API_KEY")
    GEODB_API_KEY: Optional[str] = Field(None, env="GEODB_API_KEY")

    # LangSmith settings (optional)
    LANGCHAIN_TRACING_V2: bool = Field(False, env="LANGCHAIN_TRACING_V2")
    LANGCHAIN_API_KEY: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: Optional[str] = Field(None, env="LANGCHAIN_PROJECT")
    LANGCHAIN_ENDPOINT: Optional[str] = Field(
        "https://api.smith.langchain.com",
        env="LANGCHAIN_ENDPOINT"
    )

    # External API endpoints
    GEODB_API_HOST: str = Field("wft-geo-db.p.rapidapi.com", env="GEODB_API_HOST")
    OPENWEATHER_API_URL: str = Field(
        "https://api.openweathermap.org/data/2.5/weather", 
        env="OPENWEATHER_API_URL"
    )
    WORLDTIME_API_URL: str = Field(
        "http://worldtimeapi.org/api/timezone", 
        env="WORLDTIME_API_URL"
    )

    # Model configuration
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @field_validator("BACKEND_CORS_ORIGINS")
    @classmethod
    def assemble_cors_origins(
        cls, v: Union[str, List[str]]
    ) -> Union[List[str], str]:
        """Parse CORS origins from environment."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate LOG_LEVEL value."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {', '.join(valid_levels)}")
        return v.upper()


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    This function is cached to prevent reloading the .env file on every request.
    """
    return Settings()


# Create settings instance to be imported
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Log the current environment
logger.info("Running in %s mode", settings.ENVIRONMENT)
logger.info("Using OpenAI model: %s", settings.OPENAI_MODEL)
