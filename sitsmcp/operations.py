#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Operations module."""

from functools import lru_cache

from sqlalchemy import Engine

from sitsmcp import store
from sitsmcp.config import Settings


#
# Internal functions
#
@lru_cache(maxsize=1)
def _settings() -> Settings:
	"""Return the process-wide settings.

	Returns:
		Settings: The settings instance.
	"""
	return Settings()


@lru_cache(maxsize=1)
def _engine() -> Engine:
	"""Return the process-wide database engine.

	Returns:
		Engine: The SQLAlchemy engine bound to the configured database.
	"""
	return store.create_db_engine(_settings().db_path)


def _format_hit(hit: dict) -> str:
	"""Format a search hit.

	Args:
		hit (dict): A chunk dictionary from ``store.search_hybrid``.

	Returns:
		str: The hit formatted as a string.
	"""
	title = hit.get("title") or "Untitled"
	url = hit.get("url") or ""

	header = f"# {title}\n{url}".rstrip()

	# return!
	return f"{header}\n\n{hit['text']}"


def _search(query: str, source: str) -> list[str]:
	"""Run a hybrid search.

	Args:
		query (str): User query.

		source (str): The corpus tag to search.

	Returns:
		list[str]: Results formatted as strings.
	"""
	settings = _settings()
	engine = _engine()

	hits = store.search_hybrid(
		engine=engine,
		model_name=settings.embedding_model,
		query=query,
		source=source,
		top_k=settings.top_k,
		candidate_k=settings.candidate_k,
		rrf_k=settings.rrf_k,
	)

	return [_format_hit(hit) for hit in hits]


#
# Public functions
#
def get_function(name: str) -> str:
	"""Return the complete documentation for a SITS function.

	Args:
		name (str): Function name (e.g., ``sits_classify``).

	Returns:
		str: FUnction documentation if found, otherwise a not-found message.
	"""
	engine = _engine()

	# get the function documentation
	doc = store.get_function(engine, name)

	# check if the function was found
	if doc is None:
		# inform user function was not found
		return (
			f"No SITS function named '{name}' was found. "
			"Use the exact function name (for example 'sits_classify')."
		)

	return doc


def search_docs(query: str) -> list[str]:
	"""Search the SITS book.

	Args:
		query (str): User query.

	Returns:
		list[str]: Results formatted as strings.
	"""
	return _search(
		query=query,
		source="book",
	)


def search_references(query: str) -> list[str]:
	"""Search the indexed reference articles.

	Args:
		query (str): User query.

	Returns:
		list[str]: Results formatted as strings.
	"""
	return _search(
		query=query,
		source="reference_article",
	)
