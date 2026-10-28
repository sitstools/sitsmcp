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

from sitsmcp import access, store
from sitsmcp.config import Settings
from sitsmcp.models import (
	Chapter,
	FunctionDoc,
	FunctionLookup,
	Hit,
	IndexInfo,
	SearchResult,
	Section,
	SourceInfo,
)
from sitsmcp.rerank import active_rerankers, rerank_search, rerankers_for


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


def _search(query: str, source: str) -> SearchResult:
	"""Run a hybrid search, reranked for the client of the request.

	Args:
		query (str): User query.

		source (str): The corpus tag to search.

	Returns:
		SearchResult: The hits, best first, and the reranker that ordered them.
	"""
	settings = _settings()
	engine = _engine()
	rerankers = rerankers_for(settings, access.client_of_request())

	# run the search
	hits, reranker = rerank_search(
		engine, settings, rerankers, query, source, settings.top_k
	)

	# build the result
	return SearchResult(hits=[hit_from_row(row) for row in hits], reranker=reranker)


#
# Public functions
#
def hit_from_row(row: dict) -> Hit:
	"""Build a hit from a chunk dictionary.

	Args:
		row (dict): A chunk dictionary from `store.search_hybrid`.

	Returns:
		Hit: The search hit.
	"""
	url = row.get("url") or ""

	# locate the chunk in the book
	chapter = None
	if row.get("chapter_number") or row.get("chapter_title"):
		chapter = Chapter(
			number=row.get("chapter_number"),
			title=row.get("chapter_title"),
		)

	# locate the chunk in the section
	section = None
	if row.get("section_id"):
		section = Section(
			id=row["section_id"],
			number=row.get("section_number"),
			title=row.get("section_title"),
		)

	# build the hit
	return Hit(
		id=row["chunk_id"],
		source=row["source"],
		title=row.get("title") or "Untitled",
		chapter=chapter,
		section=section,
		url=url,
		page_url=row.get("page_url") or url,
		language=row.get("language"),
		text=row["text"],
		score=row["score"],
		rerank_score=row.get("rerank_score"),
		function_name=row.get("function_name"),
	)


def format_hit(hit: Hit) -> str:
	"""Format a search hit as text.

	Args:
		hit (Hit): The hit.

	Returns:
		str: The hit formatted as a string.
	"""
	header = f"# {hit.title}\n{hit.url}".rstrip()

	# return!
	return f"{header}\n\n{hit.text}"


def format_function(lookup: FunctionLookup) -> str:
	"""Format a function lookup as text.

	Args:
		lookup (FunctionLookup): The lookup result.

	Returns:
		str: Function documentation if found, otherwise a not-found message.
	"""
	if lookup.function is None:
		# inform user function was not found
		return (
			f"No SITS function named '{lookup.name}' was found. "
			"Use the exact function name (for example 'sits_classify')."
		)

	# return!
	return lookup.function.text


def index_info() -> IndexInfo:
	"""Report when the index was built and how it ranks and searches.

	Returns:
		IndexInfo: The sources with their dates, the embedding model and the
			active key and default rerankers, each `None` when off.
	"""
	settings = _settings()

	# read the rows
	rows = store.index_info(_engine())

	# return!
	return IndexInfo(
		sources=[SourceInfo(**row) for row in rows],
		embedding_model=settings.embedding_model,
		**active_rerankers(settings),
	)


def format_index_info(info: IndexInfo) -> str:
	"""Format the index report as text.

	Args:
		info (IndexInfo): The report.

	Returns:
		str: One line per source, then the embedding model and the two rerankers,
			each `None` when off.
	"""
	lines = []

	# describe each source
	for source in info.sources:
		# build the source line
		line = (
			f"{source.source}: {source.chunks} chunks, "
			f"indexed {source.indexed_at or 'unknown'}, "
			f"built {source.built_at or 'unknown'}"
		)

		# add the version if it is known
		if source.version:
			line += f", sits {source.version}"

		# add the line to the lines
		lines.append(line)

	# describe the embedding model and the rerankers
	lines.append(f"embedding model: {info.embedding_model}")
	lines.append(f"key reranker: {info.key_reranker or 'none'}")
	lines.append(f"default reranker: {info.default_reranker or 'none'}")

	# return!
	return "\n".join(lines)


def get_function(name: str) -> FunctionLookup:
	"""Look up the complete documentation for a SITS function.

	Args:
		name (str): Function name (e.g., ``sits_classify``).

	Returns:
		FunctionLookup: The lookup; ``function`` is ``None`` if not found.
	"""
	engine = _engine()

	# get the function documentation
	doc = store.get_function(engine, name)

	# check if the function was found
	if doc is None:
		return FunctionLookup(name=name, function=None)

	return FunctionLookup(name=name, function=FunctionDoc(**doc))


def search_docs(query: str) -> SearchResult:
	"""Search the SITS book.

	Args:
		query (str): User query.

	Returns:
		SearchResult: The hits, best first.
	"""
	return _search(
		query=query,
		source="book",
	)


def search_functions(query: str) -> SearchResult:
	"""Search the SITS function reference.

	Args:
		query (str): User query.

	Returns:
		SearchResult: The hits, best first.
	"""
	return _search(
		query=query,
		source="function",
	)


def search_references(query: str) -> SearchResult:
	"""Search the indexed reference articles.

	Args:
		query (str): User query.

	Returns:
		SearchResult: The hits, best first.
	"""
	return _search(
		query=query,
		source="reference_article",
	)
