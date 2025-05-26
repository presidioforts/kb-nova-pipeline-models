"""
Logging configuration for KB Nova Pipeline Models.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any

from .config import get_settings, get_log_level


def setup_logging() -> None:
    """
    Setup logging configuration for the application.
    """
    settings = get_settings()
    log_level = get_log_level()
    
    # Create logs directory
    logs_dir = Path(settings.paths.logs)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": str(logs_dir / "app.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": str(logs_dir / "error.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "sentence_transformers": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "transformers": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "torch": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False
            }
        }
    }
    
    # Apply logging configuration
    logging.config.dictConfig(config)
    
    # Set specific log levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """
    Set the log level for all loggers.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Update root logger
    logging.getLogger().setLevel(numeric_level)
    
    # Update console handler
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setLevel(numeric_level)
            break


def configure_model_logging() -> None:
    """Configure logging for ML model libraries to reduce noise."""
    # Suppress HuggingFace progress bars and warnings
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set specific loggers to WARNING level
    noisy_loggers = [
        "sentence_transformers",
        "transformers",
        "torch",
        "tensorflow",
        "sklearn",
        "matplotlib",
        "PIL"
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING) 