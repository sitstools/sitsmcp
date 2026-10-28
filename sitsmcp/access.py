#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Access module."""

import hmac
import math
import re
import time
from collections import deque
from collections.abc import Callable
from typing import Annotated, Any

import pydash as py_
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.middleware import Middleware
from pydantic import (
	BaseModel,
	Field,
	StringConstraints,
	ValidationError,
	field_validator,
)
from pydantic_core import ErrorDetails
from starlette.datastructures import Headers
from starlette.middleware import Middleware as AsgiMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from sitsmcp.config import Settings

#
# Constants
#
MIN_KEY_LENGTH = 16
"""Fewest characters of an API key."""

PROBLEMS = {
	"string_pattern_mismatch": "name must be 1 to 32 characters of a-z, 0-9, _ or -",
	"string_too_short": f"key must have at least {MIN_KEY_LENGTH} characters",
}
"""The problem to report for each kind of error of ``ApiKey``."""

WINDOW_SECONDS = 60.0
"""Length of the sliding window of the rate limit, in seconds."""

HEALTH_PATH = "/health"
"""Path that needs no key."""


#
# Public functions
#
def parse_api_keys(raw: str) -> "ApiKeys":
	"""Parse api keys setting.

	Args:
		raw (str): The api keys setting.

	Returns:
		ApiKeys: The api keys.

	Raises:
		ValueError: Raised when the api keys setting is invalid.
	"""
	# nothing configured
	if not raw.strip():
		return ApiKeys()

	# parse api keys
	api_keys = py_.map_(py_.split(raw, ","), _parse_entry)

	# check if api keys are unique
	_check_unique(api_keys)

	# return api keys
	return ApiKeys({api_key.name: api_key.key for api_key in api_keys})


def client_of_request() -> str | None:
	"""Get the client of the current HTTP request.

	Returns:
		str | None: The client of the current HTTP request.
	"""
	# find the request
	try:
		request = get_http_request()
	
	except RuntimeError:
		return None

	# return client
	return getattr(request.state, "sitsmcp_client", None) or "ip:unknown"


def http_middleware(settings: Settings) -> list[AsgiMiddleware]:
	"""Build the ASGI middleware for the HTTP app.

	Args:
		settings (Settings): The active settings.
	
	Returns:
		list[AsgiMiddleware]: The ASGI middleware for the HTTP app.
	"""
	# get the api keys setting
	raw = settings.api_keys.get_secret_value() if settings.api_keys else ""

	# return the ASGI middleware for the HTTP app
	return [AsgiMiddleware(ApiKeyMiddleware, keys=parse_api_keys(raw))]


def install(mcp: FastMCP, settings: Settings) -> None:
	"""Install the tool rate limit on the server.

	Args:
		mcp (FastMCP): The FastMCP server.

		settings (Settings): The active settings.
	"""
	# forget the earlier limiter
	mcp.middleware[:] = [
		middleware
		for middleware in mcp.middleware
		if not isinstance(middleware, ToolRateLimit)
	]

	# add the tool rate limit
	mcp.add_middleware(ToolRateLimit(settings.rate_limit_open, settings.rate_limit_key))


#
# Internal functions
#
def _parse_entry(entry: str, index: int) -> "ApiKey":
	"""Parse one api key entry.

	Args:
		entry (str): The api key entry.

		index (int): The 0-based position of the api key entry in the setting.

	Returns:
		ApiKey: The api key.

	Raises:
		ValueError: Raised when the api key entry is invalid.
	"""
	position = index + 1
	name, colon, key = entry.strip().partition(":")

	# reject an empty entry
	if not entry.strip():
		raise ValueError(f"API key entry {position}: is empty")
	
	# reject an entry without a ':' between name and key
	if not colon:
		raise ValueError(f"API key entry {position}: has no ':' between name and key")

	# validate the pair
	try:
		return ApiKey(name=name, key=key)
	
	except ValidationError as error:
		# get the problem
		problem = _problem_of(error.errors(include_input=False)[0])

	# raise an error with the problem
	raise ValueError(f"API key entry {position}: {problem}")


def _problem_of(error: ErrorDetails) -> str:
	"""Get the problem of an api key error.

	Args:
		error (ErrorDetails): The first error of pydantic, without the input.

	Returns:
		str: The problem of the api key error.
	"""
	return PROBLEMS.get(error["type"]) or str(error["ctx"]["error"])


def _check_unique(api_keys: list["ApiKey"]) -> None:
	"""Check that no api key is used twice.

	Args:
		api_keys (list[ApiKey]): The api keys, in the order of the setting.

	Raises:
		ValueError: Raised when the first api key is used twice.
	"""
	names = set()
	keys = set()

	# find a repeat
	for index, api_key in enumerate(api_keys, start=1):
		# reject a duplicate name
		if api_key.name in names:
			raise ValueError(
				f"API key entry {index}: name {api_key.name!r} is used twice"
			)

		# reject a duplicate key
		if api_key.key in keys:
			raise ValueError(f"API key entry {index}: key is used twice")

		# store the api key
		names.add(api_key.name)
		keys.add(api_key.key)


#
# Classes
#
class ApiKey(BaseModel):
	"""One API key: the name of a client and its secret."""

	name: Annotated[str, StringConstraints(pattern=r"^[a-z0-9_-]{1,32}$")]
	"""What the API key calls its client."""

	key: Annotated[str, StringConstraints(min_length=MIN_KEY_LENGTH)] = Field(
		repr=False
	)
	"""The secret; it never shows in a ``repr`` or an error message."""

	@field_validator("key")
	@classmethod
	def _check_characters(cls, key: str) -> str:
		"""Reject whitespace and anything but printable ASCII characters.

		Args:
			key (str): The key.

		Returns:
			str: The same key.
		"""
		# reject whitespace
		if re.search(r"\s", key):
			raise ValueError("API key must not contain whitespace")
		
		# reject non-printable ASCII characters
		if not key.isascii() or not key.isprintable():
			raise ValueError("API key must be printable ASCII")

		# return!
		return key


class ApiKeys(dict[str, str]):
	"""The API keys, with a ``repr`` that hides the keys."""

	def __repr__(self) -> str:
		"""Show the API keys and never the keys.

		Returns:
			str: The API keys, each with a masked key.
		"""
		return "ApiKeys({" + ", ".join(f"{api_key.name!r}: '***'" for api_key in self) + "})"

	__str__ = __repr__


class ApiKeyMiddleware:
	"""API key middleware handler."""

	def __init__(self, app: ASGIApp, keys: dict[str, str]) -> None:
		"""Wrap an ASGI application.

		Args:
			app (ASGIApp): The ASGI application to protect.

			keys (dict[str, str]): The API keys.
		"""
		self.app = app
		self.keys = keys

	async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
		"""Handle one ASGI event.

		Args:
			scope (Scope): The ASGI scope.

			receive (Receive): The ASGI receive channel.

			send (Send): The ASGI send channel.
		"""
		# pass other scopes through
		if scope["type"] != "http":
			await self.app(scope, receive, send)
			return

		# name the client
		client = self._client(scope)

		# reject a bad key
		if client is None:
			response = JSONResponse(
				{"error": "invalid API key"},
				status_code=401,
				headers={"WWW-Authenticate": "Bearer"},
			)

			# send the response
			await response(scope, receive, send)

			# return!
			return

		# remember the API key
		if client:
			scope.setdefault("state", {})["sitsmcp_client"] = client

		# call the app
		await self.app(scope, receive, send)

	def _client(self, scope: Scope) -> str | None:
		"""Name the API key of a request.

		Args:
			scope (Scope): The ASGI scope.

		Returns:
			str | None: The API key, or ``None`` for a wrong or malformed key.
		"""
		# exempt the health check
		if scope["path"] == HEALTH_PATH:
			return ""

		# find the address
		address = scope.get("client")
		host = f"ip:{address[0]}" if address else "ip:unknown"

		# open use
		headers = Headers(scope=scope).getlist("authorization")
		if not headers or not self.keys:
			return host

		# reject more than one header
		if len(headers) > 1:
			return None

		# get the header
		header = headers[0]

		# read the API key
		parts = header.strip().split(None, 1)
		if len(parts) != 2 or parts[0].lower() != "bearer":
			return None

		# get the API key
		api_key_value = parts[1].strip().encode()

		# compare with every key
		api_key_valid = None
		
		for name, key in self.keys.items():
			if hmac.compare_digest(api_key_value, key.encode()):
				api_key_valid = name

		# return client
		return None if api_key_valid is None else f"key:{api_key_valid}"


class ToolRateLimit(Middleware):
	"""Limit the tool calls of each client to a sliding window of 60 seconds."""

	_calls_by_client: dict[str, deque[float]]
	"""The times of the recent tool calls of each client."""

	def __init__(
		self,
		open_limit: int,
		key_limit: int,
		clock: Callable[[], float] = time.monotonic,
	) -> None:
		"""Create the limiter.

		Args:
			open_limit (int): Calls a minute of a client without a key. Set to 0 to disable.

			key_limit (int): Calls a minute of a client with a key. Set to 0 to disable.

			clock (Callable[[], float]): Seconds, increasing. A test injects its own.
		"""
		self.open_limit = open_limit
		self.key_limit = key_limit
		self.clock = clock

		# initialize the calls and swept times
		self._calls_by_client = {}
		self._swept_by_clients = clock()

	def _prune(self, now: float) -> None:
		"""Forget the clients idle for over a window.

		Args:
			now (float): The current time.
		"""
		# sweep once a window
		if now - self._swept_by_clients < WINDOW_SECONDS:
			return

		# update the swept time
		self._swept_by_clients = now

		# find idle clients
		clients = [
			client for client, calls in self._calls_by_client.items() if now - calls[-1] > WINDOW_SECONDS
		]

		# drop them
		for client in clients:
			del self._calls_by_client[client]

	def _check(self, client: str) -> None:
		"""Count a call of a client.

		Args:
			client (str): The client name.

		Raises:
			ToolError: When the client is over its limit.
		"""
		is_key = client.startswith("key:")
		limit = self.key_limit if is_key else self.open_limit

		# no limit
		if limit <= 0:
			return

		# get the current time
		now = self.clock()

		# prune idle clients
		self._prune(now)

		# get the calls by client
		calls = self._calls_by_client.setdefault(client, deque())

		# slide the window
		while calls and now - calls[0] >= WINDOW_SECONDS:
			calls.popleft()

		# refuse a call over the limit
		if len(calls) >= limit:
			# calculate the wait time
			wait = max(1, math.ceil(calls[0] + WINDOW_SECONDS - now))

			# who is over the limit
			who = f"key {client[4:]}" if is_key else "open use"

			# more information
			more = "" if is_key else ", or use an API key"

			# raise the error with the proper message
			raise ToolError(
				f"Rate limit: {limit} tool calls per minute for {who}; "
				f"retry in {wait} s{more}"
			)

		# count the call
		calls.append(now)

	async def on_call_tool(self, context: Any, call_next: Any) -> Any:
		"""Count a tool call before it runs.

		Args:
			context (Any): The middleware context.

			call_next (Any): The next handler.

		Returns:
			Any: The result of the tool.
		"""
		client = client_of_request()

		# limit only HTTP clients
		if client is not None:
			self._check(client)

		# return result
		return await call_next(context)
