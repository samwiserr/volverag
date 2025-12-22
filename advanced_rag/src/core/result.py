"""
Result monad for type-safe error handling.

This module provides a Result type that represents either a success value (Ok)
or an error (Err), following functional programming patterns. This eliminates
the need for bare exception handling and provides explicit error propagation.
"""
from typing import TypeVar, Generic, Optional, Callable, Union, Any
from dataclasses import dataclass
from enum import Enum

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E', bound='AppError')


class ErrorType(Enum):
    """Categorized error types for better handling and logging."""
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND_ERROR = "not_found_error"
    EXTERNAL_API_ERROR = "external_api_error"
    PROCESSING_ERROR = "processing_error"
    CONFIGURATION_ERROR = "configuration_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    CACHE_ERROR = "cache_error"
    RETRIEVAL_ERROR = "retrieval_error"
    LLM_ERROR = "llm_error"


@dataclass(frozen=True)
class AppError:
    """
    Structured error with context for better debugging and user messaging.
    
    Attributes:
        type: Categorized error type
        message: Human-readable error message
        details: Optional additional error details
        original_error: Original exception if this was converted from one
        context: Optional context dictionary for debugging
    """
    type: ErrorType
    message: str
    details: Optional[dict] = None
    original_error: Optional[Exception] = None
    context: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Serialize error for logging/API responses."""
        result = {
            "type": self.type.value,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        if self.context:
            result["context"] = self.context
        if self.original_error:
            result["original_error"] = str(self.original_error)
        return result
    
    def __str__(self) -> str:
        """String representation for logging."""
        msg = f"[{self.type.value}] {self.message}"
        if self.context:
            msg += f" (context: {self.context})"
        return msg


class Result(Generic[T, E]):
    """
    Type-safe Result monad for error handling.
    
    Represents either a successful value (Ok) or an error (Err). This pattern
    eliminates the need for bare exception handling and makes error propagation
    explicit and type-safe.
    
    Examples:
        >>> result = Result.ok(42)
        >>> result.is_ok()
        True
        >>> result.unwrap()
        42
        
        >>> error = AppError(ErrorType.VALIDATION_ERROR, "Invalid input")
        >>> result = Result.err(error)
        >>> result.is_err()
        True
    """
    
    def __init__(self, value: Optional[T] = None, error: Optional[AppError] = None):
        """
        Initialize Result. Should use Result.ok() or Result.err() instead.
        
        Args:
            value: Success value (if Ok)
            error: Error (if Err)
            
        Raises:
            ValueError: If both or neither are provided
        """
        if value is not None and error is not None:
            raise ValueError("Result cannot have both value and error")
        # Allow None as a valid value (represents successful None return)
        # Only raise if error is provided without value being explicitly set
        # (error=None and value=None is valid - it means Result.ok(None))
        if error is not None and value is not None:
            raise ValueError("Result cannot have both value and error")
        self._value: Optional[T] = value
        self._error: Optional[AppError] = error
    
    @classmethod
    def ok(cls, value: Optional[T] = None) -> 'Result[T, AppError]':
        """
        Create successful result.
        
        Args:
            value: Success value (can be None)
            
        Returns:
            Result containing the value
        """
        return cls(value=value, error=None)
    
    @classmethod
    def err(cls, error: AppError) -> 'Result[T, AppError]':
        """
        Create error result.
        
        Args:
            error: AppError instance
            
        Returns:
            Result containing the error
        """
        return cls(value=None, error=error)
    
    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        error_type: ErrorType,
        context: Optional[dict] = None
    ) -> 'Result[T, AppError]':
        """
        Convert exception to Result.
        
        Args:
            exc: Exception to convert
            error_type: Error type category
            context: Optional context for debugging
            
        Returns:
            Result containing the error
        """
        return cls.err(AppError(
            type=error_type,
            message=str(exc),
            original_error=exc,
            context=context
        ))
    
    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self._error is None
    
    def is_err(self) -> bool:
        """Check if result is an error."""
        return self._error is not None
    
    def unwrap(self) -> T:
        """
        Get value or raise RuntimeError.
        
        Returns:
            Success value
            
        Raises:
            RuntimeError: If result is an error
        """
        if self._error:
            raise RuntimeError(f"Unwrapped error result: {self._error.message}")
        return self._value  # type: ignore
    
    def unwrap_or(self, default: T) -> T:
        """
        Get value or return default.
        
        Args:
            default: Default value to return if error
            
        Returns:
            Success value or default
        """
        return self._value if self._error is None else default
    
    def unwrap_or_else(self, func: Callable[[AppError], T]) -> T:
        """
        Get value or compute from error.
        
        Args:
            func: Function to compute value from error
            
        Returns:
            Success value or computed value
        """
        if self._error:
            return func(self._error)
        return self._value  # type: ignore
    
    def map(self, func: Callable[[T], U]) -> 'Result[U, AppError]':
        """
        Map over successful value.
        
        Args:
            func: Function to apply to value
            
        Returns:
            New Result with mapped value, or same error
        """
        if self._error:
            return Result.err(self._error)
        try:
            return Result.ok(func(self._value))  # type: ignore
        except Exception as e:
            return Result.from_exception(e, ErrorType.PROCESSING_ERROR)
    
    def map_err(self, func: Callable[[AppError], AppError]) -> 'Result[T, AppError]':
        """
        Map over error.
        
        Args:
            func: Function to transform error
            
        Returns:
            New Result with transformed error, or same value
        """
        if self._error:
            return Result.err(func(self._error))
        return Result.ok(self._value)  # type: ignore
    
    def and_then(self, func: Callable[[T], 'Result[U, AppError]']) -> 'Result[U, AppError]':
        """
        Chain operations (monadic bind).
        
        Args:
            func: Function that returns a Result
            
        Returns:
            Result from function, or same error if this is an error
        """
        if self._error:
            return Result.err(self._error)
        return func(self._value)  # type: ignore
    
    def or_else(self, func: Callable[[AppError], 'Result[T, AppError]']) -> 'Result[T, AppError]':
        """
        Handle error by returning alternative Result.
        
        Args:
            func: Function to compute alternative Result from error
            
        Returns:
            This Result if successful, or alternative from func
        """
        if self._error:
            return func(self._error)
        return Result.ok(self._value)  # type: ignore
    
    def value(self) -> Optional[T]:
        """
        Get value (may be None if error).
        
        Returns:
            Value or None
        """
        return self._value
    
    def error(self) -> Optional[AppError]:
        """
        Get error (may be None if successful).
        
        Returns:
            Error or None
        """
        return self._error
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        if self._error:
            return f"Result.err({self._error})"
        return f"Result.ok({self._value!r})"
    
    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        if not isinstance(other, Result):
            return False
        return self._value == other._value and self._error == other._error

