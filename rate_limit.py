import time
from functools import wraps
from fastapi import Request, status, Response
from functools import wraps
from fastapi.responses import JSONResponse


# Global in-memory store for request logs by IP
request_logs = {}

def rate_limited(max_calls: int, time_window: int):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, response: Response, *args, **kwargs):
            ip = request.client.host
            now = time.time()
            log = request_logs.get(ip, [])

            # Clean up old requests that are outside of the time window
            log = [timestamp for timestamp in log if now - timestamp < time_window]

            # Check if the number of requests within the time window exceeds the limit
            if len(log) >= max_calls:
                reset_time = int(time_window - (now - log[0]))
                headers = {
                    "X-RateLimit-Limit": str(max_calls),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "X-RateLimit-Try-Again-After": f"Please try again in {reset_time} seconds."
                }
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded. Try again later."},
                    headers=headers
                )

            # Log the current request
            log.append(now)
            request_logs[ip] = log

            # Include rate limit headers in the response
            remaining_time = int(time_window - (now - log[0]) if log else time_window)
            response.headers["X-RateLimit-Limit"] = str(max_calls)
            response.headers["X-RateLimit-Remaining"] = str(max_calls - len(log))
            response.headers["X-RateLimit-Reset"] = str(remaining_time)
            response.headers["X-RateLimit-Try-Again-After"] = f"Please try again in {remaining_time} seconds."

            return await func(request, response, *args, **kwargs)

        return wrapper
    return decorator
