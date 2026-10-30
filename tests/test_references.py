#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Unit tests for the reference citation registry."""

from pathlib import Path

import orjson

from sitsmcp import data

#
# Constants
#
REFERENCES_PATH = Path(__file__).resolve().parents[1] / "data" / "references.json"
"""Path to the citation registry shipped with the package."""

ENTRY = {
	"id": "maciel2020trajectories",
	"title": "Identifying Land Use Change Trajectories in Brazil's Agricultural Frontier",
	"authors": ["Adeline M. Maciel", "Lubia Vinhas"],
	"year": 2020,
	"venue": "Land",
	"doi": "10.3390/land9120506",
	"url": "https://doi.org/10.3390/land9120506",
	"domain": ["land-use", "agriculture"],
	"abstract": "Many of the world's agricultural frontiers are in the tropics.",
}
"""A complete reference entry used across the rendering tests."""


#
# Internal functions
#
def _write(tmp_path: Path, entries: list[dict]) -> Path:
	"""Write entries to a temporary registry file.

	Args:
		tmp_path (Path): The pytest temporary directory.

		entries (list[dict]): The reference entries to write.

	Returns:
		Path: Path to the written registry.
	"""
	path = tmp_path / "references.json"
	path.write_bytes(orjson.dumps(entries))

	return path


#
# Tests - parsing
#
def test_each_entry_becomes_exactly_one_chunk(tmp_path):
	"""Test that every entry yields exactly one chunk."""
	# arrange
	path = _write(tmp_path, [ENTRY, ENTRY | {"id": "other", "url": "https://e.org/2"}])

	# act
	chunks = data.parse_references(path)

	# assert
	assert len(chunks) == 2
	assert all(chunk["source"] == "reference_article" for chunk in chunks)
	assert all(chunk["function_name"] is None for chunk in chunks)


def test_chunk_carries_the_citable_metadata(tmp_path):
	"""Test that a chunk carries the title and URL used to cite the article."""
	# arrange
	path = _write(tmp_path, [ENTRY])

	# act
	chunk = data.parse_references(path)[0]

	# assert
	assert chunk["title"] == ENTRY["title"]
	assert chunk["url"] == ENTRY["url"]


def test_entries_without_a_url_are_skipped(tmp_path):
	"""Test that entries that cannot be cited are skipped."""
	# arrange
	path = _write(tmp_path, [ENTRY, {"id": "broken", "title": "No link"}])

	# act
	chunks = data.parse_references(path)

	# assert
	assert len(chunks) == 1


#
# Tests - shipped registry
#
def test_shipped_registry_indexes_every_publication():
	"""Test that the shipped registry yields one chunk per publication."""
	# arrange
	entries = orjson.loads(REFERENCES_PATH.read_bytes())

	# act
	chunks = data.parse_references(REFERENCES_PATH)

	# assert
	assert len(entries) == 18
	assert len(chunks) == len(entries)
	assert len({chunk["id"] for chunk in chunks}) == len(entries)
