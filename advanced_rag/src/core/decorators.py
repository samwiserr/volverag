"""
Decorators for error handling and other cross-cutting concerns.
"""
from functools import wraps
from typing import Callable, TypeVar, Any
from .result import Result, AppError, ErrorType

F = TypeVar('F', bound=Callable)


def handle_errors(
    error_type: ErrorType,
    default_message: str = "Operation failed"
) -> Callable[[F], F]:
    """
    Decorator to convert exceptions to Result.
    
    This decorator wraps a function and automatically converts any raised
    exceptions into Result.err() with the specified error type.
    
    Args:
        error_type: Error type to use for exceptions
        default_message: Default message if exception has no message
        
    Returns:
        Decorator function
        
    Example:
        >>> @handle_errors(ErrorType.PROCESSING_ERROR)
        ... def process_data(data: str) -> Result[str, AppError]:
        ...     # If this raises, it becomes Result.err()
        ...     return Result.ok(process(data))
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
                # If function already returns Result, return it
                if isinstance(result, Result):
                    return result
                # Otherwise wrap in Result.ok()
                return Result.ok(result)
            except Exception as e:
                error_context = {
                    "function": func.__name__,
                    "module": func.__module__,
                }
                if args:
                    error_context["args"] = str(args)[:200]
                return Result.from_exception(
                    e,
                    error_type,
                    context=error_context
                )
        return wrapper  # type: ignore
    return decorator


def to_result(
    error_type: ErrorType = ErrorType.PROCESSING_ERROR
) -> Callable[[F], F]:
    """
    Decorator to ensure function returns Result.
    
    Converts return value to Result.ok() if not already a Result,
    and converts exceptions to Result.err().
    
    Args:
        error_type: Error type for exceptions
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
                if isinstance(result, Result):
                    return result
                return Result.ok(result)
            except Exception as e:
                return Result.from_exception(
                    e,
                    error_type,
                    context={
                        "function": func.__name__,
                        "module": func.__module__,
                    }
                )
        return wrapper  # type: ignore
    return decorator

