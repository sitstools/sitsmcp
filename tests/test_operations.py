#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Unit tests for the operations layer."""

import logging

from sitsmcp import access, operations, rerank, store
from sitsmcp.config import Settings
from sitsmcp.rerank import RerankError

SECTION_FIELDS = (
	"page_url",
	"chapter_number",
	"chapter_title",
	"section_id",
	"section_number",
	"section_title",
	"language",
)

COHERE = "cohere/rerank-v5.0-fast"


def _row(**extra):
	"""Build a search row with the section fields set to ``None``."""
	row = {
		"chunk_id": "c1",
		"text": "body text",
		"title": "A Title",
		"url": "https://example.org/page.html",
		"source": "book",
		"score": 0.5,
	}
	row.update(dict.fromkeys(SECTION_FIELDS))
	row.update(extra)

	return row


def test_get_function_returns_sections_and_arguments(monkeypatch):
	"""Test that get_function carries the version, sections and arguments."""
	# arrange
	doc = {
		"name": "sits_regularize",
		"title": "Build a regular data cube",
		"url": "https://example.org/f",
		"text": "doc:sits_regularize",
		"version": "1.5.4",
		"sections": [
			{
				"id": "value",
				"name": "Value",
				"url": "https://example.org/f#value",
				"text": "A cube.",
			}
		],
		"arguments": [
			{
				"name": "data",
				"url": "https://example.org/f#arg-data",
				"text": "A cube.",
			}
		],
	}
	monkeypatch.setattr(store, "get_function", lambda _engine, _name: doc)

	# act
	result = operations.get_function("sits_regularize")

	# assert
	assert result.function is not None
	assert result.function.arguments[0].url.endswith("#arg-data")
	assert result.function.sections[0].id == "value"
	assert result.function.version == "1.5.4"
	assert operations.format_function(result) == doc["text"]


def test_get_function_not_found_message(monkeypatch):
	"""Test that a function that is not found has no doc and a message."""
	# arrange
	monkeypatch.setattr(store, "get_function", lambda _engine, _name: None)

	# act
	result = operations.get_function("nope")

	# assert
	assert result.function is None
	assert "No SITS function named 'nope'" in operations.format_function(result)


def test_search_docs_formats_hits(monkeypatch):
	"""Test that search_docs returns hits that format correctly."""
	# arrange
	monkeypatch.setattr(store, "search_hybrid", lambda *args, **kwargs: [_row()])

	# act
	result = operations.search_docs("query")
	text = operations.format_hit(result.hits[0])

	# assert
	assert text.startswith("# A Title")
	assert "body text" in text


#
# Tests - the reranker of each client
#
class _Encoder:
	"""A local cross-encoder stand-in that puts the last text first."""

	def rerank(self, query, documents):
		yield from (float(index) for index in range(len(documents)))


def _cohere_answers(order):
	"""Stub ``CohereReranker.rerank`` to return ``order``."""
	return lambda self, query, documents: order


def _reranker_fails(self, query, documents):
	"""Stub a reranker's ``rerank`` to fail like an outage."""
	raise RerankError("Cohere rerank failed: HTTP 503")


def _cohere_must_not_be_called(self, query, documents):
	"""Stub ``CohereReranker.rerank`` to fail the test."""
	raise AssertionError("open use called Cohere")


def _search_as(monkeypatch, client, cohere, default_model="fastembed/m"):
	"""Search the book as ``client``, with ``cohere`` for keys and a local default.

	Args:
		monkeypatch: The pytest monkeypatch.

		client: The client ``access.client_of_request`` gives.

		cohere: The stub of ``CohereReranker.rerank``.

		default_model: The default reranker setting.

	Returns:
		SearchResult: The result of the search.
	"""
	settings = Settings(
		key_reranker_model=COHERE,
		reranker_api_key="fake-cohere-key",
		default_reranker_model=default_model,
		_env_file=None,
	)

	# arrange
	rows = [_row(chunk_id="c0", text="a"), _row(chunk_id="c1", text="b")]

	monkeypatch.setattr(operations, "_settings", lambda: settings)
	monkeypatch.setattr(operations, "_engine", lambda: None)
	monkeypatch.setattr(store, "search_hybrid", lambda *args, **kwargs: rows)
	monkeypatch.setattr(access, "client_of_request", lambda: client)
	monkeypatch.setattr(rerank.CohereReranker, "rerank", cohere)
	monkeypatch.setattr(rerank, "_load_cross_encoder", lambda model: _Encoder())

	# return!
	return operations.search_docs("q")


def test_open_use_never_calls_cohere(monkeypatch):
	# act
	result = _search_as(monkeypatch, "ip:1.2.3.4", _cohere_must_not_be_called)

	# assert
	assert result.reranker == "fastembed/m"
	assert [hit.id for hit in result.hits] == ["c1", "c0"]
	assert [hit.rerank_score for hit in result.hits] == [1.0, 0.0]


def test_key_holder_gets_the_key_reranker(monkeypatch):
	# act
	result = _search_as(monkeypatch, "key:chat", _cohere_answers([(1, 0.9), (0, 0.2)]))

	# assert
	assert result.reranker == COHERE
	assert [hit.rerank_score for hit in result.hits] == [0.9, 0.2]


def test_key_holder_falls_back_to_the_default(monkeypatch, caplog):
	# act
	with caplog.at_level(logging.WARNING):
		result = _search_as(monkeypatch, "key:chat", _reranker_fails)

	# assert
	assert result.reranker == "fastembed/m"
	assert [hit.rerank_score for hit in result.hits] == [1.0, 0.0]
	assert len(caplog.records) == 1
	assert caplog.records[0].levelno == logging.WARNING
	assert "HTTP 503" in caplog.records[0].getMessage()


def test_all_rerankers_fail_gives_fused_order(monkeypatch, caplog):
	# arrange
	monkeypatch.setattr(rerank.FastembedReranker, "rerank", _reranker_fails)

	# act
	with caplog.at_level(logging.WARNING):
		result = _search_as(monkeypatch, "key:chat", _reranker_fails)

	# assert
	assert result.reranker is None
	assert [hit.id for hit in result.hits] == ["c0", "c1"]
	assert [hit.rerank_score for hit in result.hits] == [None, None]
	assert len(caplog.records) == 2


def test_search_functions_searches_the_function_source(monkeypatch):
	"""Test that search_functions asks for the function source."""
	# arrange
	captured = {}

	def fake_search(*args, **kwargs):
		captured.update(kwargs)

		return [_row(source="function", function_name="sits_regularize")]

	monkeypatch.setattr(store, "search_hybrid", fake_search)

	# act
	result = operations.search_functions("make a regular cube")

	# assert
	assert captured["source"] == "function"
	assert result.hits[0].function_name == "sits_regularize"
