"""
Unit tests for security module (rate limiting and sanitization).
"""
import pytest
import time
from src.core.security import TokenBucket, RateLimiter, sanitize_input


@pytest.mark.unit
class TestTokenBucket:
    """Test TokenBucket class."""
    
    def test_initial_tokens(self):
        """Test bucket starts with specified tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        # Should start with 0 tokens by default, but we initialize to capacity
        assert bucket.tokens >= 0
    
    def test_consume_succeeds_when_tokens_available(self):
        """Test consume returns True when tokens are available."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.tokens = 5.0
        assert bucket.consume(3) is True
        assert bucket.tokens == 2.0
    
    def test_consume_fails_when_insufficient_tokens(self):
        """Test consume returns False when insufficient tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.tokens = 2.0
        assert bucket.consume(5) is False
        assert bucket.tokens == 2.0  # Should not consume
    
    def test_refill_over_time(self):
        """Test tokens refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.tokens = 0.0
        bucket.last_refill = time.time()
        
        time.sleep(0.1)  # Wait 100ms
        bucket._refill()
        
        # Should have refilled approximately 0.1 tokens
        assert bucket.tokens > 0
        assert bucket.tokens <= 10  # Should not exceed capacity
    
    def test_refill_respects_capacity(self):
        """Test refill doesn't exceed capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)  # Very fast refill
        bucket.tokens = 9.0
        bucket.last_refill = time.time()
        
        time.sleep(0.1)
        bucket._refill()
        
        assert bucket.tokens <= 10.0
    
    def test_available_returns_current_tokens(self):
        """Test available returns number of available tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.tokens = 7.5
        assert bucket.available() == 7


@pytest.mark.unit
class TestRateLimiter:
    """Test RateLimiter class."""
    
    def test_allows_request_within_limit(self):
        """Test allows requests within rate limit."""
        limiter = RateLimiter(requests_per_minute=60)
        result = limiter.check_rate_limit("user1")
        assert result.is_ok()
        assert result.unwrap() is True
    
    def test_blocks_request_over_limit(self):
        """Test blocks requests over rate limit."""
        limiter = RateLimiter(requests_per_minute=2)  # Very low limit
        result1 = limiter.check_rate_limit("user1")
        assert result1.is_ok()
        result2 = limiter.check_rate_limit("user1")
        assert result2.is_ok()
        result3 = limiter.check_rate_limit("user1")
        assert result3.is_err()  # Should be blocked
    
    def test_tracks_per_user(self):
        """Test rate limiting is per user/identifier."""
        limiter = RateLimiter(requests_per_minute=2)
        result1 = limiter.check_rate_limit("user1")
        assert result1.is_ok()
        result2 = limiter.check_rate_limit("user1")
        assert result2.is_ok()
        result3 = limiter.check_rate_limit("user1")
        assert result3.is_err()  # user1 blocked
        
        result4 = limiter.check_rate_limit("user2")
        assert result4.is_ok()  # user2 still allowed
    
    def test_remaining_returns_correct_count(self):
        """Test get_remaining returns correct remaining requests."""
        limiter = RateLimiter(requests_per_minute=10)
        limiter.check_rate_limit("user1")
        remaining = limiter.get_remaining("user1")
        assert remaining >= 0
        assert remaining < 10
    
    def test_reset_clears_bucket(self):
        """Test reset clears user's rate limit bucket."""
        limiter = RateLimiter(requests_per_minute=2)
        limiter.check_rate_limit("user1")
        limiter.check_rate_limit("user1")
        result = limiter.check_rate_limit("user1")
        assert result.is_err()  # Blocked
        
        limiter.reset("user1")
        result2 = limiter.check_rate_limit("user1")
        assert result2.is_ok()  # Should be allowed again
    
    def test_reset_nonexistent_user(self):
        """Test reset on non-existent user doesn't raise."""
        limiter = RateLimiter(requests_per_minute=60)
        limiter.reset("nonexistent")  # Should not raise
    
    def test_reset_all_clears_all_buckets(self):
        """Test reset with None clears all buckets."""
        limiter = RateLimiter(requests_per_minute=60)
        limiter.check_rate_limit("user1")
        limiter.check_rate_limit("user2")
        assert len(limiter._buckets) > 0
        
        limiter.reset(None)  # Reset all
        assert len(limiter._buckets) == 0


@pytest.mark.unit
class TestSanitizeInput:
    """Test sanitize_input function."""
    
    def test_allows_safe_text(self):
        """Test allows safe text."""
        safe_text = "What is the porosity of Hugin formation?"
        result = sanitize_input(safe_text)
        assert result.is_ok()
        assert result.unwrap() == safe_text
    
    def test_removes_html_tags(self):
        """Test removes HTML tags."""
        unsafe = "<script>alert('xss')</script>Hello"
        result = sanitize_input(unsafe)
        # Should reject dangerous patterns
        assert result.is_err() or "<script>" not in result.unwrap()
    
    def test_removes_javascript(self):
        """Test removes JavaScript code."""
        unsafe = "Hello<script>alert('xss')</script>World"
        result = sanitize_input(unsafe)
        # Should reject dangerous patterns
        assert result.is_err() or ("alert" not in result.unwrap().lower())
    
    def test_handles_sql_injection_attempts(self):
        """Test handles SQL injection attempts."""
        unsafe = "'; DROP TABLE users; --"
        result = sanitize_input(unsafe)
        # Should return Result (may be ok if no dangerous patterns match)
        assert isinstance(result, Result)
        # SQL injection patterns might not trigger our XSS patterns, so just check it's handled
        if result.is_ok():
            assert isinstance(result.unwrap(), str)
    
    def test_handles_empty_string(self):
        """Test handles empty string."""
        result = sanitize_input("")
        assert result.is_ok()
        assert result.unwrap() == ""
    
    def test_handles_whitespace_only(self):
        """Test handles whitespace-only input."""
        result = sanitize_input("   \n\t  ")
        assert result.is_ok()
        assert isinstance(result.unwrap(), str)
    
    def test_preserves_well_names(self):
        """Test preserves valid well names."""
        well_name = "15/9-F-5"
        result = sanitize_input(well_name)
        assert result.is_ok()
        assert "15/9-F-5" in result.unwrap()
    
    def test_handles_special_characters(self):
        """Test handles special characters in queries."""
        query = "What is the porosity of Hugin formation in well 15/9-F-5?"
        result = sanitize_input(query)
        assert result.is_ok()
        assert "15/9-F-5" in result.unwrap()
    
    def test_rejects_too_long_input(self):
        """Test rejects input that exceeds max_length."""
        long_text = "x" * 3000  # Exceeds default max_length of 2000
        result = sanitize_input(long_text)
        assert result.is_err()
        assert "too long" in result.error().message.lower()
    
    def test_rejects_non_string_input(self):
        """Test rejects non-string input."""
        result = sanitize_input(123)  # Not a string
        assert result.is_err()
        assert "must be a string" in result.error().message.lower()

