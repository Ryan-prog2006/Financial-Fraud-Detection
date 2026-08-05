"""Rate limiting and request logging middleware for FinShield API."""

from __future__ import annotations

import os
import time
import json
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Rate limit tracking: {ip: [timestamps]}
_rate_limits: dict[str, list[float]] = {}

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        if client_ip not in _rate_limits:
            _rate_limits[client_ip] = []
            
        # Clean up timestamps older than 60 seconds
        _rate_limits[client_ip] = [ts for ts in _rate_limits[client_ip] if now - ts <= 60.0]
        
        if len(_rate_limits[client_ip]) >= 60:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Max 60 requests/minute."}
            )
            
        _rate_limits[client_ip].append(now)
        response = await call_next(request)
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        processing_time_ms = (time.perf_counter() - t0) * 1000
        
        log_entry = {
            "timestamp": time.time(),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time_ms": round(processing_time_ms, 2)
        }
        
        os.makedirs("data", exist_ok=True)
        with open("data/api_logs.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        return response
