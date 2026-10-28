#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - MCP server module."""

import sqlite3
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from sqlalchemy import event
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware as AsgiMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from sitsmcp import operations, store
from sitsmcp.models import FunctionLookup, IndexInfo, SearchResult
from sitsmcp.rerank import active_rerankers

#
# Constants
#
mcp = FastMCP("sitsmcp")
"""The FastMCP server instance."""

HEALTH_BUSY_MS = 1000
"""Milliseconds ``/health`` waits for a locked database."""


#
# Public functions
#
@mcp.tool(output_schema=FunctionLookup.model_json_schema())
def sits_get_function(name: str) -> ToolResult:
	"""Get the complete reference documentation for a SITS function.

	Use this when you need the exact signature, arguments, return value or
	examples of a specific SITS function and you already know its name
	(for example ``sits_classify``, ``sits_cube`` or ``sits_train``).

	Args:
		name (str): Function name (e.g., ``sits_classify``).

	Returns:
		str: Function documentation if found, otherwise a not-found message.
			The result also carries the function as structured data.
	"""
	lookup = operations.get_function(name)

	# return!
	return ToolResult(
		content=operations.format_function(lookup),
		structured_content=lookup.model_dump(),
	)


@mcp.tool(output_schema=IndexInfo.model_json_schema())
def sits_index_info() -> ToolResult:
	"""Report when the index was built, from which book build and sits version, and how it searches."""
	info = operations.index_info()

	# return!
	return ToolResult(
		content=operations.format_index_info(info),
		structured_content=info.model_dump(),
	)


@mcp.tool(output_schema=SearchResult.model_json_schema())
def sits_search_docs(query: str) -> ToolResult:
	"""Search the SITS book for concepts, workflows and explanations.

	Use this to learn how SITS works and how to combine its tools. Topics
	include: the classification workflow, building data cubes, machine- and
	deep-learning models, Self-Organising Maps (SOM), sampling and accuracy
	assessment, and the reasoning behind SITS methods.

	Args:
		query (str): User query.

	Returns:
		ToolResult: Readable text, one block per hit, plus structured hits.
	"""
	result = operations.search_docs(query)

	# return!
	return ToolResult(
		content=[
			TextContent(type="text", text=operations.format_hit(hit))
			for hit in result.hits
		],
		structured_content=result.model_dump(),
	)


@mcp.tool(output_schema=SearchResult.model_json_schema())
def sits_search_functions(query: str) -> ToolResult:
	"""Find SITS functions by what they do.

	Use this when you know what a function should do but not its name (for
	example "make a regular cube" or "remove clouds"). Each hit is a section
	or an argument of a function's reference page, and its `function_name`
	names the function. Then call `sits_get_function` with that name for the
	full page.

	Args:
		query (str): User query.

	Returns:
		ToolResult: Readable text, one block per hit, plus structured hits.
	"""
	result = operations.search_functions(query)

	# return!
	return ToolResult(
		content=[
			TextContent(type="text", text=operations.format_hit(hit))
			for hit in result.hits
		],
		structured_content=result.model_dump(),
	)


@mcp.tool(output_schema=SearchResult.model_json_schema())
def sits_search_references(query: str) -> ToolResult:
	"""Search academic articles and references that use SITS.

	Use this to find how SITS has been applied in research: case studies,
	methods, study areas and results reported in papers that use the package.

	Args:
		query (str): User query.

	Returns:
		ToolResult: Readable text, one block per hit, plus structured hits.
	"""
	result = operations.search_references(query)

	# return!
	return ToolResult(
		content=[
			TextContent(type="text", text=operations.format_hit(hit))
			for hit in result.hits
		],
		structured_content=result.model_dump(),
	)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
	"""Say whether search can work.

	Args:
		request (Request): The request.

	Returns:
		JSONResponse: 200 with the chunk counts and the key and default
			rerankers, or 503 with the reason.
	"""
	try:
		settings = operations._settings()

		# a missing file is not created
		if not settings.db_path.exists():
			return _unavailable("no database")

		# count the chunks
		counts = await run_in_threadpool(_count_chunks, settings.db_path)
		if not sum(counts.values()):
			return _unavailable("no chunks")

		# return!
		return JSONResponse(
			{
				"status": "ok",
				"chunks": counts,
				**active_rerankers(settings),
			}
		)

	except Exception as error:
		return _unavailable(type(error).__name__)


#
# Internal functions
#
def _count_chunks(db_path: Path) -> dict[str, int]:
	"""Count the chunks on a connection of its own, off the event loop.

	Args:
		db_path (Path): The database file.

	Returns:
		dict[str, int]: The number of chunks of each source.
	"""
	engine = store.create_db_engine(db_path)

	@event.listens_for(engine, "connect")
	def _short_wait(dbapi_conn: sqlite3.Connection, _record: object) -> None:
		dbapi_conn.execute(f"PRAGMA busy_timeout = {HEALTH_BUSY_MS}")

	# count the chunks
	try:
		return store.chunk_counts(engine)
	
	finally:
		engine.dispose()


def _unavailable(reason: str) -> JSONResponse:
	"""Build the answer of an unavailable server.

	Args:
		reason (str): Why search cannot work.

	Returns:
		JSONResponse: The 503 answer.
	"""
	return JSONResponse({"status": "unavailable", "reason": reason}, status_code=503)


#
# MCP Server
#
def main(host: str, port: int, middleware: list[AsgiMiddleware] | None = None) -> None:
	"""Run the MCP server over HTTP.

	Args:
		host (str): Host interface to bind to.

		port (int): Port to listen on.

		middleware (list[AsgiMiddleware] | None): ASGI middleware of the app.
	"""
	mcp.run(transport="http", host=host, port=port, middleware=middleware)
