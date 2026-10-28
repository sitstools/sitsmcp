#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Unit tests for the SQLite storage and hybrid search layer."""

import pytest
from conftest import EMBEDDING_DIM, _fake_embed

from sitsmcp import store


@pytest.fixture
def engine(tmp_path, monkeypatch):
	"""Fixture to create a real temp DB with stubbed embeddings."""
	monkeypatch.setattr(store, "embed_texts", _fake_embed)

	# initialize the database
	db_engine = store.create_db_engine(tmp_path / "sitsmcp-test.db")
	store.init_db(db_engine, EMBEDDING_DIM)

	# return the database engine
	return db_engine


def test_store_and_hybrid_search_finds_seeded_chunk(engine):
	"""Test that store_and_hybrid_search finds a seeded chunk."""
	# arrange
	chunks = [
		{
			"id": "book-1",
			"text": "Random forest classification of satellite image time series.",
			"source": "book",
			"title": "Classification",
			"url": "https://example.org/rf",
			"function_name": None,
		},
		{
			"id": "book-2",
			"text": "Building data cubes from Sentinel-2 imagery.",
			"source": "book",
			"title": "Data cubes",
			"url": "https://example.org/cube",
			"function_name": None,
		},
	]

	# act
	stored = store.store_chunks(engine, "stub", chunks)
	hits = store.search_hybrid(engine, "stub", "random forest", "book", 5, 10, 60)

	# assert
	assert stored == 2
	assert hits
	assert "Random forest" in hits[0]["text"]


def test_functions_round_trip(engine):
	"""Test that a function record returns version, sections, arguments."""
	# arrange
	url = "https://example.org/sits_classify.html"
	sections = [
		{"id": "value", "name": "Value", "url": f"{url}#value", "text": "A cube."}
	]
	arguments = [{"name": "data", "url": f"{url}#arg-data", "text": "Input."}]
	records = [
		{
			"function_name": "sits_classify",
			"title": "sits_classify",
			"url": url,
			"text": "Full documentation for sits_classify.",
			"version": "1.5.4",
			"sections": sections,
			"arguments": arguments,
		}
	]

	# store the functions
	stored = store.store_functions(engine, records)
	found = store.get_function(engine, "sits_classify")
	missing = store.get_function(engine, "sits_unknown")

	# assert
	assert stored == 1
	assert found == {
		"name": "sits_classify",
		"title": "sits_classify",
		"url": url,
		"text": "Full documentation for sits_classify.",
		"version": "1.5.4",
		"sections": sections,
		"arguments": arguments,
	}

	# assert the missing function is None
	assert missing is None


def _section_chunk(chunk_id: str, body: str) -> dict:
	"""Build a book chunk with all the section fields set to ``None``.

	Args:
		chunk_id (str): The chunk identifier.

		body (str): The chunk text.

	Returns:
		dict: A chunk dictionary.
	"""
	return {
		"id": chunk_id,
		"text": body,
		"source": "book",
		"title": "Random forest",
		"url": "https://example.org/ml.html#rf",
		"function_name": None,
		"page_url": "https://example.org/ml.html",
		"chapter_number": "9",
		"chapter_title": "Machine learning",
		"section_id": "rf",
		"section_number": "9.2",
		"section_title": "Random forest",
		"language": "python",
	}


def test_section_fields_round_trip(engine):
	"""Test that the section fields come back unchanged from a search."""
	# arrange
	chunk = _section_chunk("book-1", "Random forest classification.")

	# store the chunk
	store.store_chunks(engine, "stub", [chunk])
	hits = store.search_hybrid(engine, "stub", "random forest", "book", 5, 10, 60)

	# assert the hits
	assert hits
	for column in store.CHUNK_COLUMNS:
		assert hits[0][column] == chunk[column]


def test_hits_carry_descending_scores(engine):
	"""Test that hits carry positive scores, sorted from best to worst."""
	# arrange
	chunks = [
		_section_chunk("book-1", "Random forest classification of images."),
		_section_chunk("book-2", "Building data cubes from forest imagery."),
	]

	# act
	store.store_chunks(engine, "stub", chunks)
	hits = store.search_hybrid(engine, "stub", "random forest", "book", 5, 10, 60)
	scores = [hit["score"] for hit in hits]

	# assert
	assert len(hits) == 2
	assert all(isinstance(score, float) and score > 0 for score in scores)
	assert scores == sorted(scores, reverse=True)
