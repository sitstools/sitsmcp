#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Unit tests for the MCP server."""

import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from conftest import EMBEDDING_DIM, _fake_embed
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError
from starlette.testclient import TestClient

from sitsmcp import access, operations, server, store
from sitsmcp.config import Settings
from sitsmcp.models import (
	Chapter,
	FunctionLookup,
	Hit,
	IndexInfo,
	SearchResult,
	Section,
	SourceInfo,
)


def _call(tool, args):
	"""Call a tool through an in-memory client."""

	async def run():
		async with Client(server.mcp) as client:
			return await client.call_tool(tool, args)

	return asyncio.run(run())


def test_search_docs_returns_text_and_structure(monkeypatch):
	"""Test that a search return."""
	# arrange
	hit = Hit(
		id="c1",
		source="book",
		title="5.2 Regularizing",
		chapter=Chapter(number="5", title="Building regular data cubes"),
		section=Section(id="regularizing", number="5.2", title="Regularizing"),
		url="https://example.org/page.html#regularizing",
		page_url="https://example.org/page.html",
		language=None,
		text="body text",
		score=0.5,
	)

	# stub the search_docs function
	monkeypatch.setattr(
		operations, "search_docs", lambda _query: SearchResult(hits=[hit])
	)

	# act
	result = _call("sits_search_docs", {"query": "regularize"})

	# assert
	assert result.structured_content["hits"][0]["url"].endswith("#regularizing")
	assert result.content[0].text.startswith("# 5.2 Regularizing")


def test_get_function_not_found_is_structured_none(monkeypatch):
	# arrange
	monkeypatch.setattr(
		operations,
		"get_function",
		lambda name: FunctionLookup(name=name, function=None),
	)

	# act
	result = _call("sits_get_function", {"name": "nope"})

	# assert
	assert result.structured_content["function"] is None
	assert "No SITS function named 'nope'" in result.content[0].text


def test_index_info_returns_text_and_structure(monkeypatch):
	"""Test that the index report returns structured and formatted text."""
	# arrange
	info = IndexInfo(
		sources=[
			SourceInfo(
				source="function",
				indexed_at="2026-10-25T10:00:00Z",
				built_at="2026-01-14T18:19:00Z",
				version="1.5.4",
				chunks=1656,
			)
		],
		embedding_model="m",
		key_reranker=None,
		default_reranker=None,
	)

	# stub the index_info function
	monkeypatch.setattr(operations, "index_info", lambda: info)

	# act
	result = _call("sits_index_info", {})

	# assert
	assert IndexInfo.model_validate(result.structured_content) == info
	assert result.content[0].text == operations.format_index_info(info)


#
# Tests - /health
#
@pytest.fixture
def database(tmp_path, monkeypatch):
	"""Point the server at a temporary database that does not exist yet."""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setenv("SITSMCP_DB_PATH", str(tmp_path / "health.db"))

	# clear the cache
	operations._settings.cache_clear()
	operations._engine.cache_clear()

	# yield the database
	yield tmp_path / "health.db"

	# clear the cache
	operations._settings.cache_clear()
	operations._engine.cache_clear()


def _seed(path, chunks):
	"""Create a database with ``chunks``.

	Args:
		path (Path): The database file.

		chunks (list[dict]): The chunks to store.
	"""
	# create the engine
	engine = store.create_db_engine(path)

	# initialize the database
	store.init_db(engine, EMBEDDING_DIM)

	# store the chunks
	if chunks:
		store.store_chunks(engine, "stub", chunks)

	# dispose the engine
	engine.dispose()


def _chunk(chunk_id, source):
	"""Build a chunk of ``source``."""
	return {
		"id": chunk_id,
		"text": f"text of {chunk_id}",
		"source": source,
		"title": chunk_id,
		"url": f"https://example.org/{chunk_id}",
		"function_name": None,
	}


def _health(headers=None):
	"""Ask ``/health`` of the server through an in-process client."""
	with TestClient(server.mcp.http_app(stateless_http=True)) as client:
		return client.get("/health", headers=headers)


def test_health_ok(database, monkeypatch):
	# arrange
	monkeypatch.setattr(store, "embed_texts", _fake_embed)

	# seed the database
	_seed(database, [_chunk("a", "book"), _chunk("b", "book"), _chunk("c", "function")])

	# set the environment variables
	monkeypatch.setenv("SITSMCP_KEY_RERANKER_MODEL", "cohere/rerank-v5.0-fast")
	monkeypatch.setenv("SITSMCP_RERANKER_API_KEY", "fake-cohere-key")
	monkeypatch.setenv(
		"SITSMCP_DEFAULT_RERANKER_MODEL", "fastembed/Xenova/ms-marco-MiniLM-L-6-v2"
	)

	# ask the health endpoint
	response = _health()

	# assert
	assert response.status_code == 200
	assert response.json() == {
		"status": "ok",
		"chunks": {
			"book": 2,
			"function": 1,
		},
		"key_reranker": "cohere/rerank-v5.0-fast",
		"default_reranker": "fastembed/Xenova/ms-marco-MiniLM-L-6-v2",
	}


def test_health_no_database_is_503(database):
	# act
	response = _health()

	# assert
	assert response.status_code == 503
	assert response.json() == {"status": "unavailable", "reason": "no database"}
	assert not database.exists()


#
# Tests - limited tool calls over HTTP
#
def _free_port() -> int:
	"""Find a free port."""
	with socket.socket() as sock:
		sock.bind(("127.0.0.1", 0))
		return sock.getsockname()[1]


def test_second_tool_call_over_http_is_limited(monkeypatch, restore_mcp):
	# arrange
	monkeypatch.setattr(
		operations,
		"index_info",
		lambda: operations.IndexInfo(
			sources=[],
			embedding_model="stub",
			key_reranker=None,
			default_reranker=None,
		),
	)

	# set the settings
	settings = Settings(_env_file=None, rate_limit_open=1, rate_limit_key=0)

	# install the MCP
	access.install(server.mcp, settings)
	
	# create the HTTP app
	app = server.mcp.http_app(
		stateless_http=True, middleware=access.http_middleware(settings)
	)

	port = _free_port()
	uvi = uvicorn.Server(
		uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
	)

	# start the server
	thread = threading.Thread(target=uvi.run, daemon=True)
	thread.start()

	# wait for the server to start
	for _ in range(100):
		if uvi.started:
			break
		time.sleep(0.05)

	# assert the server started
	assert uvi.started

	# define the tool call
	async def run():
		async with Client(
			StreamableHttpTransport(f"http://127.0.0.1:{port}/mcp")
		) as client:
			# call the tool
			first = await client.call_tool("sits_index_info", {})

			# call the tool again
			with pytest.raises(ToolError) as error:
				await client.call_tool("sits_index_info", {})

			# return the results
			return first, str(error.value)

	# try to call the tool
	try:
		first, message = asyncio.run(run())

	finally:
		uvi.should_exit = True
		thread.join(timeout=10)

	# assert
	assert first.structured_content["embedding_model"] == "stub"
	assert "Rate limit: 1 tool calls per minute for open use" in message
	assert "or use an API key" in message


def test_search_over_http_reranks_for_the_client_of_the_request(
	monkeypatch, restore_mcp, tmp_path
):
	"""Test that the tool, run in a worker thread, still sees its client."""
	# arrange
	clients = []

	def record(settings, client):
		clients.append(client)
		return []

	# stub the rerankers_for function
	monkeypatch.setattr(operations, "rerankers_for", record)
	monkeypatch.setenv("SITSMCP_DB_PATH", str(tmp_path / "unused.db"))
	monkeypatch.setattr(store, "search_hybrid", lambda *args, **kwargs: [])
	
	# set the environment variables
	key = "test-key-0123456789abcdef"
	monkeypatch.setenv("SITSMCP_API_KEYS", f"chat:{key}")
	
	# update flask not atualizado
	settings = Settings(_env_file=None, rate_limit_open=0, rate_limit_key=0)
	
	access.install(server.mcp, settings)
	app = server.mcp.http_app(
		stateless_http=True, middleware=access.http_middleware(settings)
	)

	# create the HTTP app
	port = _free_port()
	uvi = uvicorn.Server(
		uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
	)

	# start the server
	thread = threading.Thread(target=uvi.run, daemon=True)
	thread.start()

	# wait for the server to start
	for _ in range(100):
		if uvi.started:
			break
		
		time.sleep(0.05)

	# assert the server started
	assert uvi.started

	# define the search function
	async def search(headers):
		async with Client(
			StreamableHttpTransport(f"http://127.0.0.1:{port}/mcp", headers=headers)
		) as client:
			return await client.call_tool("sits_search_docs", {"query": "q"})

	# try to search
	try:
		asyncio.run(search(None))
		asyncio.run(search({"Authorization": f"Bearer {key}"}))
	
	finally:
		uvi.should_exit = True
		thread.join(timeout=10)

	# assert
	assert clients == ["ip:127.0.0.1", "key:chat"]
