"""
Structured logging configuration with JSON formatter.

All logs are output as JSON with consistent fields:
- timestamp: ISO format datetime
- level: Log level (DEBUG, INFO, WARNING, ERROR)
- logger: Logger name
- message: Human-readable message
- correlation_id: Unique request ID (from contextvars)
- Additional fields: Custom per-log fields
"""
import logging
import logging.config
import sys
from prodml.config import settings
import json
from datetime import datetime, timezone
from typing import Any, Dict
import contextvars

# Context variable to carry correlation_id through the request
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    'correlation_id',
    default='n/a'
)

logger = logging.getLogger(__name__)

class CorrelationIdFilter(logging.Filter):
    """Injects the current correlation ID from contextvars into every log record."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get()
        return True

class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging supporting both 
    minimal and detailed outputs.
        
    Outputs logs as JSON with standard fields and additional context.
    """
    def __init__(self, detailed: bool = False):
        super().__init__()
        self.detailed = detailed

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.
        
        Args:
            record: The log record to format.
            
        Returns:
            JSON string representation of the log.
        """
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "n/a"),
        }
        
        if self.detailed:
            log_entry.update({
                "filename": record.filename,
                "funcName": record.funcName,
                "lineno": record.lineno,
            })
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        for key, value in record.__dict__.items():
            if key not in [
                "asctime", "created", "filename", "funcName", "levelname",
                "levelno", "lineno", "module", "msecs", "message", "msg",
                "name", "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName", "args", "exc_info",
                "exc_text", "extra", "correlation_id"
            ]:
                if not key.startswith('_'):
                    log_entry[key] = value
                    
        return json.dumps(log_entry, default=str)

def setup_logging() -> None:
    """Configures the application-wide logging system."""
    logs_dir = settings.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "correlation_filter": {
                "()": CorrelationIdFilter,
            },
        },
       "formatters": {
            "minimal_json": {
                "()": JSONFormatter,
                "detailed": False,
            },
            "detailed_json": {
                "()": JSONFormatter,
                "detailed": True,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "minimal_json",
                "filters": ["correlation_filter"],
                "level": "DEBUG",
            },
            "info": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": logs_dir / "info.log",
                "maxBytes": 10485760,  # 1 MB
                "backupCount": 10,
                "formatter": "detailed_json",
                "filters": ["correlation_filter"],
                "level": "INFO",
            },
            "error": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": logs_dir / "error.log",
                "maxBytes": 10485760,  # 1 MB
                "backupCount": 10,
                "formatter": "detailed_json",
                "filters": ["correlation_filter"],
                "level": "ERROR",
            },
        },
        "root": {
            "handlers": ["console", "info", "error"],
            "level": "INFO",
            "propagate": True,
        },
    }

    logging.config.dictConfig(logging_config)