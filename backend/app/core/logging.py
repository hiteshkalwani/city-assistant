"""
Logging configuration for the application.

This module sets up logging with appropriate formatting and handlers.
"""

import logging
import sys
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from app.core.config import settings


class LoggingConfig(BaseModel):
    """Logging configuration model."""

    LOG_LEVEL: str = settings.LOG_LEVEL
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    LOG_FILE: str = "logs/app.log"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"
    LOG_COMPRESSION: str = "zip"


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """
        Intercept standard logging messages and redirect them to loguru.

        Args:
            record: The log record to process.
        """
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """Configure logging for the application."""
    config = LoggingConfig()

    # Create logs directory if it doesn't exist
    log_path = Path(config.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove all handlers associated with the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        backtrace=True,
        diagnose=settings.DEBUG,
    )

    # Add file handler in production
    if settings.ENVIRONMENT == "production":
        logger.add(
            config.LOG_FILE,
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            compression=config.LOG_COMPRESSION,
            level=config.LOG_LEVEL,
            format=config.LOG_FORMAT,
            backtrace=True,
            diagnose=settings.DEBUG,
        )

    # Configure standard library logging to use loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Set log levels for third-party libraries
    for name in ["uvicorn", "uvicorn.error", "fastapi"]:
        logging.getLogger(name).handlers = [InterceptHandler()]

    # Set log level for SQLAlchemy
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.DEBUG else logging.WARNING
    )

    # Disable noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logger.info("Logging configured successfully")


def get_logger(name: str = None):
    """
    Get a logger instance with the given name.

    Args:
        name: The name of the logger. If None, returns the root logger.

    Returns:
        A logger instance.
    """
    from loguru import logger
    return logger.bind(context=name) if name else logger
