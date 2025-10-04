"""Logging configuration and utilities."""

import structlog
from typing import Any, Dict, Optional
from .config import get_settings


def get_logger(name: str = "rework") -> structlog.BoundLogger:
    """Get a configured logger instance."""
    config = get_settings()
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if config.logging.format == "json" 
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logger to any class."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger instance."""
        return get_logger(self.__class__.__name__)


def log_error(logger: structlog.BoundLogger, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log an error with context."""
    logger.error(
        "Error occurred",
        error=str(error),
        error_type=type(error).__name__,
        **(context or {})
    )


def log_audit_event(
    logger: structlog.BoundLogger,
    event_type: str,
    resource_id: str,
    action: str,
    result: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Log an audit event."""
    logger.info(
        "Audit event",
        event_type=event_type,
        resource_id=resource_id,
        action=action,
        result=result,
        **(context or {})
    )
