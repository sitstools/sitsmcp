#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Unit tests for the SQLite storage and hybrid search layer."""

import hashlib
from pathlib import Path

import pytest

from sitsmcp import store

#
# Constants
#
TEST_DIR = Path("test-output")
EMBEDDING_DIM = 8


def _fake_embed(model_name: str, texts: list[str]) -> list[list[float]]:
	"""Stub embedding, independent of any model download.

	Args:
		model_name (str): The embedding model identifier.

		texts (list[str]): The text to embed.

	Returns:
		list[list[float]]: The embedded text.
	"""
	vectors = []

	# embed the text
	for text in texts:
		digest = hashlib.sha256(text.encode()).digest()

		vectors.append([digest[i] / 255.0 for i in range(EMBEDDING_DIM)])

	# return vectors
	return vectors


@pytest.fixture
def engine(tmp_path, monkeypatch):
	"""Fixture to create a real temp DB with stubbed embeddings."""
	TEST_DIR.mkdir(exist_ok=True)
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
	"""Test that functions round trip correctly."""
	# arrange
	records = [
		{
			"function_name": "sits_classify",
			"title": "sits_classify",
			"url": "https://example.org/sits_classify",
			"text": "Full documentation for sits_classify.",
		}
	]

	# act
	stored = store.store_functions(engine, records)
	found = store.get_function(engine, "sits_classify")
	missing = store.get_function(engine, "sits_unknown")

	# assert
	assert stored == 1
	assert found == "Full documentation for sits_classify."
	assert missing is None
