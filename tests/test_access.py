#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Unit tests for the access module."""

import asyncio

import pytest
from fastmcp.exceptions import ToolError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from sitsmcp import access
from sitsmcp.access import ApiKeyMiddleware, ToolRateLimit, parse_api_keys

#
# Constants
#
CHAT_KEY = "test-key-0123456789abcdef"
"""A fake key; it must never appear in an error."""

NB_KEY = "test-key-fedcba9876543210"
"""Another fake key."""


#
# Internal functions
#
def _app(keys: dict[str, str]) -> TestClient:
	"""Build a client of a tiny app that echoes the client."""

	async def echo(request: Request) -> PlainTextResponse:
		return PlainTextResponse(request.state.sitsmcp_client)

	async def health(request: Request) -> PlainTextResponse:
		return PlainTextResponse("ok")

	# build the app
	app = Starlette(routes=[Route("/echo", echo), Route("/health", health)])

	# return client
	return TestClient(ApiKeyMiddleware(app, keys))


class _Clock:
	"""A clock a test moves."""

	def __init__(self) -> None:
		self.now = 1000.0

	def __call__(self) -> float:
		return self.now


def _call(limit: ToolRateLimit) -> str:
	"""Make one tool call through the limiter."""

	async def call_next(context):
		return "done"

	# return result
	return asyncio.run(limit.on_call_tool(None, call_next))


def _as(monkeypatch: pytest.MonkeyPatch, client: str | None) -> None:
	"""Name the client of the next request."""
	monkeypatch.setattr(access, "client_of_request", lambda: client)


#
# Tests - parse_api_keys
#
def test_parse_api_keys():
	# arrange
	raw = f"chat:{CHAT_KEY},nb:{NB_KEY}"

	# act
	keys = parse_api_keys(raw)

	# assert
	assert keys == {"chat": CHAT_KEY, "nb": NB_KEY}
	assert repr(keys) == str(keys) == "ApiKeys({'chat': '***', 'nb': '***'})"


@pytest.mark.parametrize(
	("raw", "entry"),
	[
		("nocolonkey0123456789", 1),
		(f":{CHAT_KEY}", 1),
		(f"chat:{CHAT_KEY},Chat!:{NB_KEY}", 2),
		("chat:fifteen-chars-x", 1),
		("chat:has space inside-key", 1),
		(f"chat:{CHAT_KEY},chat:{NB_KEY}", 2),
		(f"chat:{CHAT_KEY},nb:{CHAT_KEY}", 2),
		(f"chat:{CHAT_KEY},,nb:{NB_KEY}", 2),
	],
)
def test_parse_api_keys_rejects(raw, entry):
	# act
	with pytest.raises(ValueError) as error:
		parse_api_keys(raw)

	# assert
	assert f"API key entry {entry}:" in str(error.value)
	assert CHAT_KEY not in str(error.value)
	assert "has space inside-key" not in str(error.value)


@pytest.mark.parametrize(
	"raw",
	[
		"nocolonkey0123456789",
		f":{CHAT_KEY}",
		f"chat:{CHAT_KEY},Chat!:{NB_KEY}",
		"chat:fifteen-chars-x",
		"chat:has space inside-key",
		f"chat:{CHAT_KEY},chat:{NB_KEY}",
		f"chat:{CHAT_KEY},nb:{CHAT_KEY}",
		f"chat:{CHAT_KEY},,nb:{NB_KEY}",
		"chat:caf\u00e9-key-0123456789",
	],
)
def test_parse_api_keys_never_shows_the_key(raw):
	# act
	with pytest.raises(ValueError) as error:
		parse_api_keys(raw)

	# assert
	shown = repr(error.value) + str(error.value) + str(error.value.__context__)
	assert not error.value.__cause__

	# check each key
	for entry in raw.split(","):
		key = entry.partition(":")[2].strip()
		assert not key or key not in shown


#
# Tests - ApiKeyMiddleware
#
def test_no_header_is_open_use():
	# arrange
	client = _app({"chat": CHAT_KEY})

	# act
	response = client.get("/echo")

	# assert
	assert response.text == "ip:testclient"


@pytest.mark.parametrize(
	"header",
	[f"Bearer {CHAT_KEY}", f"bearer {CHAT_KEY}", f"  Bearer   {CHAT_KEY}  "],
)
def test_known_key_names_the_client(header):
	# arrange
	client = _app({"chat": CHAT_KEY, "nb": NB_KEY})

	# act
	response = client.get("/echo", headers={"Authorization": header})

	# assert
	assert response.text == "key:chat"


@pytest.mark.parametrize(
	"header", ["Bearer wrong-key-0123456789", "Basic abc", "Bearer", ""]
)
def test_wrong_key_is_401(header):
	# arrange
	client = _app({"chat": CHAT_KEY})

	# act
	response = client.get("/echo", headers={"Authorization": header})

	# assert
	assert response.status_code == 401
	assert response.json() == {"error": "invalid API key"}
	assert response.headers["www-authenticate"] == "Bearer"


def test_health_is_exempt():
	# arrange
	client = _app({"chat": CHAT_KEY})

	# act
	response = client.get("/health", headers={"Authorization": "Bearer wrong"})

	# assert
	assert response.status_code == 200


#
# Tests - ToolRateLimit
#
def test_open_use_limit(monkeypatch):
	# arrange
	_as(monkeypatch, "ip:1.2.3.4")
	limit = ToolRateLimit(open_limit=30, key_limit=300, clock=_Clock())

	# act
	for _ in range(30):
		_call(limit)

	with pytest.raises(ToolError) as error:
		_call(limit)

	# assert
	assert "30 tool calls per minute for open use" in str(error.value)
	assert "retry in" in str(error.value)
	assert "or use an API key" in str(error.value)


def test_key_limit_is_separate(monkeypatch):
	# arrange
	limit = ToolRateLimit(open_limit=1, key_limit=300, clock=_Clock())

	_as(monkeypatch, "ip:1.2.3.4")
	_call(limit)

	with pytest.raises(ToolError):
		_call(limit)

	# act
	_as(monkeypatch, "key:chat")

	for _ in range(300):
		_call(limit)

	with pytest.raises(ToolError) as error:
		_call(limit)

	# assert
	assert "300 tool calls per minute for key chat" in str(error.value)
	assert "API key;" not in str(error.value)
	assert "or use an API key" not in str(error.value)


def test_window_slides(monkeypatch):
	# arrange
	clock = _Clock()
	_as(monkeypatch, "ip:1.2.3.4")

	limit = ToolRateLimit(open_limit=2, key_limit=0, clock=clock)

	_call(limit)
	_call(limit)

	# act
	clock.now += 59.2  # move the clock forward by 59.2 seconds

	with pytest.raises(ToolError) as error:
		_call(limit)

	clock.now += 0.8

	# assert
	assert "retry in 1 s" in str(error.value)
	assert _call(limit) == "done"


def test_zero_means_unlimited(monkeypatch):
	# arrange
	_as(monkeypatch, "ip:1.2.3.4")

	limit = ToolRateLimit(open_limit=0, key_limit=0, clock=_Clock())

	# act
	results = [_call(limit) for _ in range(1000)]

	# assert
	assert len(results) == 1000
