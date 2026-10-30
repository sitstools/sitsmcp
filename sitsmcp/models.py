#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Models module."""

from pydantic import BaseModel, Field


#
# Public classes
#
class Chapter(BaseModel):
	"""The book chapter that holds a hit."""

	number: str | None
	title: str | None


class Section(BaseModel):
	"""The book section that holds a hit."""

	id: str
	number: str | None
	title: str | None


class Hit(BaseModel):
	"""A search hit."""

	id: str
	source: str
	title: str
	chapter: Chapter | None
	section: Section | None
	url: str
	page_url: str
	language: str | None
	text: str
	score: float
	rerank_score: float | None = None
	function_name: str | None = None


class SearchResult(BaseModel):
	"""The result of a search, best hit first.

	`reranker` names the reranker that ordered the hits. `None` means fused ranking.
	"""

	hits: list[Hit]
	reranker: str | None = None


class ReferenceSection(BaseModel):
	"""A section of a function reference page."""

	id: str | None
	name: str
	url: str
	text: str


class Argument(BaseModel):
	"""An argument of a function."""

	name: str
	url: str
	text: str


class FunctionDoc(BaseModel):
	"""The documentation of a SITS function."""

	name: str
	title: str | None
	url: str | None
	text: str
	version: str | None = None
	sections: list[ReferenceSection] = Field(default_factory=list)
	arguments: list[Argument] = Field(default_factory=list)


class FunctionLookup(BaseModel):
	"""The result of a function lookup."""

	name: str
	function: FunctionDoc | None


class SourceInfo(BaseModel):
	"""When one source was indexed and from which build it was built."""

	source: str
	indexed_at: str | None
	built_at: str | None
	version: str | None
	chunks: int


class IndexInfo(BaseModel):
	"""When the index was built and how it ranks and searches."""

	sources: list[SourceInfo]
	embedding_model: str
	key_reranker: str | None
	default_reranker: str | None
