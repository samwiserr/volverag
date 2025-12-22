"""
Security utilities for rate limiting and input sanitization.

This module provides rate limiting using token bucket algorithm and
enhanced input sanitization for production security.
"""
import time
import re
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock

from .config import get_config
from .logging import get_logger
from .result import Result, AppError, ErrorType

logger = get_logger(__name__)


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.
    
    Implements the token bucket algorithm for smooth rate limiting
    with burst capacity support.
    """
    capacity: int  # Maximum tokens
    refill_rate: float  # Tokens per second
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)
    _lock: Lock = field(default_factory=Lock)
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if rate limit exceeded
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def available(self) -> int:
        """Get number of available tokens."""
        with self._lock:
            self._refill()
            return int(self.tokens)


class RateLimiter:
    """
    Rate limiter with per-user/IP tracking.
    
    Supports multiple rate limiters for different endpoints or operations.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute per user
        """
        self._buckets: Dict[str, TokenBucket] = {}
        self._requests_per_minute = requests_per_minute
        self._lock = Lock()
        # Refill rate: requests_per_minute / 60 seconds
        self._refill_rate = requests_per_minute / 60.0
    
    def _get_bucket(self, identifier: str) -> TokenBucket:
        """Get or create token bucket for identifier."""
        with self._lock:
            if identifier not in self._buckets:
                bucket = TokenBucket(
                    capacity=self._requests_per_minute,
                    refill_rate=self._refill_rate
                )
                # Initialize with full capacity so users can make requests immediately
                bucket.tokens = float(self._requests_per_minute)
                self._buckets[identifier] = bucket
            return self._buckets[identifier]
    
    def check_rate_limit(self, identifier: str) -> Result[bool, AppError]:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: User/IP identifier
            
        Returns:
            Result with True if allowed, False if rate limited
        """
        try:
            bucket = self._get_bucket(identifier)
            allowed = bucket.consume(1)
            
            if not allowed:
                available_in = int((1.0 - bucket.tokens) / bucket.refill_rate)
                return Result.err(AppError(
                    type=ErrorType.RATE_LIMIT_ERROR,
                    message=f"Rate limit exceeded. Try again in {available_in} seconds.",
                    details={"identifier": identifier, "available_tokens": bucket.available()}
                ))
            
            return Result.ok(True)
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}", exc_info=True)
            return Result.err(AppError(
                type=ErrorType.PROCESSING_ERROR,
                message="Rate limit check failed",
                details={"error": str(e)}
            ))
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        bucket = self._get_bucket(identifier)
        return bucket.available()
    
    def reset(self, identifier: Optional[str] = None) -> None:
        """Reset rate limit for identifier (or all if None)."""
        with self._lock:
            if identifier:
                self._buckets.pop(identifier, None)
            else:
                self._buckets.clear()


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        try:
            config = get_config()
            rpm = config.max_requests_per_minute
        except Exception:
            rpm = 60  # Default: 60 requests per minute
        _rate_limiter = RateLimiter(requests_per_minute=rpm)
    return _rate_limiter


def rate_limit(identifier_func: Optional[Callable] = None):
    """
    Decorator for rate limiting function calls.
    
    Args:
        identifier_func: Function to extract user/IP identifier from args/kwargs
        
    Returns:
        Decorated function with rate limiting
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get identifier
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                # Default: use first string argument or "default"
                identifier = next(
                    (str(arg) for arg in args if isinstance(arg, str)),
                    "default"
                )
            
            # Check rate limit
            limiter = get_rate_limiter()
            result = limiter.check_rate_limit(identifier)
            
            if result.is_err():
                error = result.error()
                raise ValueError(error.message)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def sanitize_input(text: str, max_length: int = 2000) -> Result[str, AppError]:
    """
    Enhanced input sanitization.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Result with sanitized text or error
    """
    if not isinstance(text, str):
        return Result.err(AppError(
            type=ErrorType.VALIDATION_ERROR,
            message="Input must be a string"
        ))
    
    # Length check
    if len(text) > max_length:
        return Result.err(AppError(
            type=ErrorType.VALIDATION_ERROR,
            message=f"Input too long (max {max_length} characters)"
        ))
    
    # Remove null bytes and control characters
    sanitized = text.replace('\x00', '')
    sanitized = ''.join(
        char for char in sanitized
        if ord(char) >= 32 or char in '\n\t\r'
    )
    
    # Check for dangerous patterns
    dangerous_patterns = [
        (r'<script', 'Script tags'),
        (r'javascript:', 'JavaScript protocol'),
        (r'onerror=', 'Event handlers'),
        (r'onload=', 'Event handlers'),
        (r'eval\(', 'Eval function'),
        (r'exec\(', 'Exec function'),
    ]
    
    import re
    text_lower = sanitized.lower()
    for pattern, description in dangerous_patterns:
        if re.search(pattern, text_lower):
            return Result.err(AppError(
                type=ErrorType.VALIDATION_ERROR,
                message=f"Input contains potentially dangerous pattern: {description}"
            ))
    
    return Result.ok(sanitized.strip())

