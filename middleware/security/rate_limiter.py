"""
Rate Limiting Middleware for FastAPI

This module implements rate limiting for the API server using Redis for distributed
rate limiting and various strategies for different API endpoints.
"""

import time
from typing import Dict, Optional, Callable, Any
import logging
from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
import redis
import json
import hashlib
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class RateLimitSettings(BaseModel):
    """Settings for rate limiting configuration."""
    enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    default_rate: int = 60  # requests per minute
    default_burst: int = 100  # maximum burst size
    whitelist_ips: list = []
    environment: str = "production"  # 'development', 'test', or 'production'

class RateLimiter:
    """Redis-based rate limiter for API requests."""
    
    def __init__(self, settings: RateLimitSettings):
        """
        Initialize the rate limiter.
        
        Args:
            settings: Configuration settings for rate limiting
        """
        self.settings = settings
        self.enabled = settings.enabled
        
        # Don't enable rate limiting in development or test environments
        if settings.environment in ('development', 'test'):
            self.enabled = False
        
        # Initialize Redis connection if enabled
        self.redis = None
        if self.enabled:
            try:
                self.redis = redis.from_url(settings.redis_url)
                # Test connection
                self.redis.ping()
                logger.info("Rate limiter initialized with Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.enabled = False
        
        # Define rate limit configurations for different endpoint groups
        self.rate_configs = {
            "default": {"rate": settings.default_rate, "burst": settings.default_burst},
            "auth": {"rate": 10, "burst": 20},  # Stricter limits for auth endpoints
            "contribution": {"rate": 30, "burst": 50},  # Moderate limits for contributions
            "model": {"rate": 20, "burst": 30},  # Moderate limits for model requests
            "public": {"rate": 120, "burst": 200},  # More lenient for public endpoints
        }
        
        # Cache of rate limits to avoid repeated calculations
        self.cache = {}
        self.cache_expiry = {}
        self.cache_ttl = 60  # 1 minute cache TTL
    
    def get_rate_limit_key(self, request: Request) -> str:
        """
        Generate a key for rate limiting based on client IP and endpoint.
        
        Args:
            request: FastAPI request object
            
        Returns:
            String key for rate limiting
        """
        # Extract client IP
        client_ip = self._get_client_ip(request)
        
        # Determine endpoint group for specific rate limiting
        path = request.url.path
        endpoint_group = self._get_endpoint_group(path)
        
        # Generate a unique key
        return f"ratelimit:{endpoint_group}:{client_ip}"
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extract the client IP from request headers, handling proxies.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client IP address string
        """
        # Try to get IP from X-Forwarded-For header
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs, use the first one (client IP)
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            # Fall back to the direct client IP
            client_ip = request.client.host
        
        return client_ip
    
    def _get_endpoint_group(self, path: str) -> str:
        """
        Determine which endpoint group a path belongs to for rate limiting.
        
        Args:
            path: Request path
            
        Returns:
            String identifying the endpoint group
        """
        if path.startswith("/api/login") or path.startswith("/api/auth"):
            return "auth"
        elif path.startswith("/api/contributions"):
            return "contribution"
        elif path.startswith("/api/model"):
            return "model"
        elif path.startswith("/api/stats") or path.startswith("/api/health"):
            return "public"
        else:
            return "default"
    
    def is_rate_limited(self, request: Request) -> tuple[bool, Dict[str, Any]]:
        """
        Check if a request should be rate limited.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Tuple of (is_limited, limit_info)
        """
        if not self.enabled or not self.redis:
            return False, {}
        
        # Check whitelist
        client_ip = self._get_client_ip(request)
        if client_ip in self.settings.whitelist_ips:
            return False, {}
        
        # Get rate limit key
        key = self.get_rate_limit_key(request)
        
        # Check cache first
        current_time = time.time()
        if key in self.cache and self.cache_expiry.get(key, 0) > current_time:
            cache_result = self.cache[key]
            if cache_result["limited"]:
                return True, cache_result["info"]
            
            # Update request count in cache
            cache_result["current"] += 1
            if cache_result["current"] > cache_result["limit"]:
                cache_result["limited"] = True
                return True, cache_result["info"]
            
            return False, cache_result["info"]
        
        # Get endpoint group and rate config
        endpoint_group = self._get_endpoint_group(request.url.path)
        config = self.rate_configs.get(endpoint_group, self.rate_configs["default"])
        rate = config["rate"]
        burst = config["burst"]
        
        # Implement token bucket algorithm with Redis
        # Using Redis to ensure distributed rate limiting works across multiple instances
        
        # Script to update and check token bucket atomically
        script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local rate = tonumber(ARGV[2])
        local burst = tonumber(ARGV[3])
        local requested = tonumber(ARGV[4])
        
        -- Initialize bucket if it doesn't exist
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1])
        local last_refill = tonumber(bucket[2])
        
        if tokens == nil then
            tokens = burst
            last_refill = now
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
            redis.call('EXPIRE', key, 60)  -- Set expiry to avoid stale keys
        end
        
        -- Refill tokens based on time elapsed
        local elapsed = now - last_refill
        local new_tokens = math.min(burst, tokens + (elapsed * rate / 60))
        local allowed = new_tokens >= requested
        
        -- Update bucket state if request is allowed
        if allowed then
            redis.call('HMSET', key, 'tokens', new_tokens - requested, 'last_refill', now)
            redis.call('EXPIRE', key, 60)  -- Refresh expiry
        end
        
        -- Return rate limiting info
        return {
            allowed and 1 or 0,
            new_tokens, 
            burst, 
            rate, 
            math.max(0, math.ceil((requested - new_tokens) * 60 / rate))
        }
        """
        
        try:
            # Execute the rate limiting script
            result = self.redis.eval(
                script,
                1,  # Number of keys
                key,  # KEYS[1]
                int(time.time()),  # ARGV[1]: current timestamp
                rate,  # ARGV[2]: rate (tokens per minute)
                burst,  # ARGV[3]: burst capacity
                1  # ARGV[4]: requested tokens (1 for a single request)
            )
            
            allowed = bool(result[0])
            current = float(result[1])
            limit = float(result[2])
            rate_per_min = float(result[3])
            retry_after = int(result[4])
            
            # Calculate reset time and remaining requests
            remaining = int(current) if allowed else 0
            reset = int(time.time() + (limit - current) * 60 / rate_per_min)
            
            # Build rate limit info
            limit_info = {
                "limit": limit,
                "remaining": remaining,
                "reset": reset,
                "retry_after": retry_after if not allowed else None
            }
            
            # Cache the result
            self.cache[key] = {
                "limited": not allowed,
                "current": 1 if allowed else 0,
                "limit": limit,
                "info": limit_info
            }
            self.cache_expiry[key] = current_time + self.cache_ttl
            
            return not allowed, limit_info
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # On error, fail open to avoid blocking legitimate requests
            return False, {}
    
    def clean_cache(self):
        """Clean expired cache entries to prevent memory growth."""
        current_time = time.time()
        expired_keys = [k for k, v in self.cache_expiry.items() if v <= current_time]
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.cache_expiry:
                del self.cache_expiry[key]
        
        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired rate limit cache entries")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, app: FastAPI, limiter: RateLimiter):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            limiter: Rate limiter instance
        """
        super().__init__(app)
        self.limiter = limiter
        self.last_cache_cleanup = time.time()
        self.cleanup_interval = 300  # Clean cache every 5 minutes
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process a request through the middleware.
        
        Args:
            request: FastAPI request
            call_next: Next middleware in the chain
            
        Returns:
            FastAPI response
        """
        # Clean cache if needed
        current_time = time.time()
        if current_time - self.last_cache_cleanup > self.cleanup_interval:
            self.limiter.clean_cache()
            self.last_cache_cleanup = current_time
        
        # Check rate limits
        is_limited, limit_info = self.limiter.is_rate_limited(request)
        
        if is_limited:
            # Return 429 Too Many Requests
            logger.warning(f"Rate limit exceeded for {request.client.host}: {request.url.path}")
            
            # Create response with appropriate headers
            retry_after = limit_info.get("retry_after", 60)
            headers = {
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(limit_info.get("limit", "")),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(limit_info.get("reset", "")),
            }
            
            return Response(
                content=json.dumps({
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": retry_after
                }),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers=headers,
                media_type="application/json"
            )
        
        # Process the request normally
        response = await call_next(request)
        
        # Add rate limit headers to response
        if limit_info:
            response.headers["X-RateLimit-Limit"] = str(limit_info.get("limit", ""))
            response.headers["X-RateLimit-Remaining"] = str(limit_info.get("remaining", ""))
            response.headers["X-RateLimit-Reset"] = str(limit_info.get("reset", ""))
        
        return response


# Helper function to create and initialize the rate limiter
def create_rate_limiter(settings: Optional[RateLimitSettings] = None) -> RateLimiter:
    """
    Create and initialize a rate limiter instance.
    
    Args:
        settings: Rate limit settings
        
    Returns:
        Initialized rate limiter
    """
    if settings is None:
        settings = RateLimitSettings()
    
    return RateLimiter(settings)

# Add rate limiting to a FastAPI app
def add_rate_limiting(app: FastAPI, settings: Optional[RateLimitSettings] = None):
    """
    Add rate limiting middleware to a FastAPI app.
    
    Args:
        app: FastAPI application
        settings: Rate limit settings
    """
    limiter = create_rate_limiter(settings)
    app.add_middleware(RateLimitMiddleware, limiter=limiter)
    
    # Store limiter in the app state for potential access from endpoints
    app.state.limiter = limiter
    
    logger.info(f"Rate limiting {'enabled' if limiter.enabled else 'disabled'}")


# Example usage
if __name__ == "__main__":
    from fastapi import FastAPI
    
    app = FastAPI()
    
    # Configure rate limiting
    settings = RateLimitSettings(
        enabled=True,
        redis_url="redis://localhost:6379/0",
        default_rate=60,
        default_burst=100,
        whitelist_ips=["127.0.0.1"],
        environment="development"
    )
    
    # Add rate limiting middleware
    add_rate_limiting(app, settings)
    
    @app.get("/")
    async def root():
        return {"message": "Hello World"}
    
    # Run the app (uncomment to test)
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
