#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - MCP server module."""

from fastmcp import FastMCP

from sitsmcp import operations

#
# Constants
#
mcp = FastMCP("sitsmcp")
"""The FastMCP server instance."""


#
# Public functions
#
@mcp.tool
def sits_get_function(name: str) -> str:
	"""Get the complete reference documentation for a SITS function.

	Use this when you need the exact signature, arguments, return value or
	examples of a specific SITS function and you already know its name
	(for example ``sits_classify``, ``sits_cube`` or ``sits_train``).

	Args:
		name (str): Function name (e.g., ``sits_classify``).

	Returns:
		str: Function documentation if found, otherwise a not-found message.
	"""
	return operations.get_function(name)


@mcp.tool
def sits_search_docs(query: str) -> list[str]:
	"""Search the SITS book for concepts, workflows and explanations.

	Use this to learn how SITS works and how to combine its tools. Topics
	include: the classification workflow, building data cubes, machine- and
	deep-learning models, Self-Organising Maps (SOM), sampling and accuracy
	assessment, and the reasoning behind SITS methods.

	Args:
		query (str): User query.

	Returns:
		list[str]: Results formatted as strings.
	"""
	return operations.search_docs(query)


@mcp.tool
def sits_search_references(query: str) -> list[str]:
	"""Search academic articles and references that use SITS.

	Use this to find how SITS has been applied in research: case studies,
	methods, study areas and results reported in papers that use the package.

	Args:
		query (str): User query.

	Returns:
		list[str]: Results formatted as strings.
	"""
	return operations.search_references(query)


def main(host: str, port: int) -> None:
	"""Run the MCP server over HTTP.

	Args:
		host (str): Host interface to bind to.

		port (int): Port to listen on.
	"""
	mcp.run(transport="http", host=host, port=port)
