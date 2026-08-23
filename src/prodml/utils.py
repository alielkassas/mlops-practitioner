"""
Utility functions and decorators.

Provides reusable tools like:
- @timed: Timing decorator for performance monitoring
- Logging configuration
"""

import time
import functools
import logging
from typing import Any, Callable, TypeVar, cast

from prodml.logging_config import correlation_id_var

logger = logging.getLogger(__name__)

# Type variable for generic function types
F = TypeVar('F', bound=Callable[..., Any])


# ---------- Timing Decorator ----------
def timed(func: F) -> F:
    """
    Decorator that logs the execution time of a function.
    
    Args:
        func: The function to wrap.
        
    Returns:
        The wrapped function with timing logic.
        
    Example:
        >>> @timed
        ... def slow_function():
        ...     time.sleep(1)
        ...
        >>> slow_function()
        slow_function took 1.0012 seconds
        
    Notes:
        This decorator is particularly useful for monitoring
        API performance and detecting performance degradation.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Log timing information
        func_name = func.__name__
        # DEBUG: Detailed timing information
        logger.debug(
            f"Timing: {func_name}",
            extra={
                "function_name": func_name,
                "execution_time_ms": execution_time_ms
            }
        )     
        return result
    
    return cast(F, wrapper)