"""
Reusable middleware for the sentiment analysis API.
"""

import logging
import time
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from config import APIConfig

logger = logging.getLogger(__name__)


class RequestTimingMiddleware(BaseHTTPMiddleware):
	"""Add an `X-Process-Time` response header."""

	async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
		start_time = time.perf_counter()
		response = await call_next(request)
		response.headers["X-Process-Time"] = f"{time.perf_counter() - start_time:.6f}"
		return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
	"""Log incoming requests and their response status codes."""

	async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
		logger.info("%s %s", request.method, request.url.path)
		response = await call_next(request)
		logger.info("%s %s -> %s", request.method, request.url.path, response.status_code)
		return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
	"""Reject oversized request bodies before the route handler runs."""

	def __init__(self, app, max_request_bytes: int):
		super().__init__(app)
		self.max_request_bytes = max_request_bytes

	async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
		if request.method in {"POST", "PUT", "PATCH"}:
			content_length = request.headers.get("content-length")
			if content_length and content_length.isdigit():
				if int(content_length) > self.max_request_bytes:
					return JSONResponse(
						status_code=413,
						content={
							"detail": (
								f"Request body too large. Max size: {self.max_request_bytes} bytes"
							)
						},
					)

		return await call_next(request)


def configure_api_middleware(
	app: FastAPI,
	*,
	max_request_bytes: Optional[int] = None,
	enable_logging: bool = True,
	enable_timing: bool = True,
) -> None:
	"""Register the API middleware stack on a FastAPI app."""

	request_limit = max_request_bytes or (APIConfig.MAX_TEXT_LENGTH * 10)

	app.add_middleware(RequestSizeLimitMiddleware, max_request_bytes=request_limit)

	if enable_logging:
		app.add_middleware(RequestLoggingMiddleware)

	if enable_timing:
		app.add_middleware(RequestTimingMiddleware)
