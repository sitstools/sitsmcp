#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Unit tests for the operations layer."""

from sitsmcp import operations, store


def test_get_function_returns_doc(monkeypatch):
	"""Test that get_function returns the documentation for a function."""
	# arrange
	monkeypatch.setattr(store, "get_function", lambda _engine, name: f"doc:{name}")

	# act
	result = operations.get_function("sits_classify")

	# assert
	assert result == "doc:sits_classify"


def test_get_function_not_found_message(monkeypatch):
	"""Test that get_function returns a message when a function is not found."""
	# arrange
	monkeypatch.setattr(store, "get_function", lambda _engine, _name: None)

	# act
	result = operations.get_function("nope")

	# assert
	assert "No SITS function named 'nope'" in result


def test_search_docs_formats_hits(monkeypatch):
	"""Test that search_docs formats hits correctly."""
	# arrange
	hit = {
		"text": "body text",
		"title": "A Title",
		"url": "https://example.org",
		"source": "book",
	}

	monkeypatch.setattr(store, "search_hybrid", lambda *args, **kwargs: [hit])

	# act
	results = operations.search_docs("query")

	# assert
	assert isinstance(results, list)
	assert results[0].startswith("# A Title")
	assert "body text" in results[0]
