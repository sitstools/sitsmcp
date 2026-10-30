#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Unit tests for the book and function reference page parsers."""

from pathlib import Path

from sitsmcp import data

#
# Constants
#
FIXTURES = Path(__file__).parent / "fixtures"
"""Saved pages of the book and of the function reference."""

BOOK_URL = "https://e-sensing.github.io/sitsbook/dc_regularize.html"
"""Address of the saved book page."""

BOOK_CONTEXT = "Chapter 5 Building regular data cubes › 5.2 Regularizing"
"""Context line of the chunks of the ``Regularizing`` section."""

REFERENCE_URL = "https://e-sensing.github.io/sits/reference/sits_regularize.html"
"""Address of the saved function reference page."""


#
# Internal functions
#
def _book_section(section_id: str, language: str | None = None) -> list[dict]:
	"""Parse the saved book page and return the chunks of one section.

	Args:
		section_id (str): The anchor of the section.

		language (str | None): ``"python"`` for its Python panes, else the main text.

	Returns:
		list[dict]: The chunks of the section.
	"""
	html = (FIXTURES / "book_page.html").read_text(encoding="utf-8")

	# parse the page
	chunks = data._parse_book_page(BOOK_URL, html, 1000, 200)

	# return!
	return [
		chunk
		for chunk in chunks
		if chunk["section_id"] == section_id
		and (chunk["language"] == "python") == (language == "python")
	]


def _reference_record() -> dict:
	"""Parse the saved function reference page."""
	html = (FIXTURES / "reference_page.html").read_text(encoding="utf-8")

	# parse the page
	record = data._parse_reference_page(REFERENCE_URL, html)

	# assert
	assert record is not None

	return record


#
# Book
#
def test_book_section_gets_its_anchor() -> None:
	# act
	regularizing = _book_section("regularizing")

	# assert
	assert regularizing[0]["url"] == f"{BOOK_URL}#regularizing"
	assert regularizing[0]["page_url"] == BOOK_URL
	assert regularizing[0]["section_number"] == "5.2"
	assert regularizing[0]["title"] == "5.2 Regularizing"
	assert regularizing[0]["text"].startswith(f"{BOOK_CONTEXT}\n\n")


def test_book_python_panes_become_their_own_chunks() -> None:
	# act
	main = _book_section("regularizing")
	python = _book_section("regularizing", "python")

	# assert
	assert main
	assert all(chunk["language"] == "r" for chunk in main)
	assert not any("```python" in chunk["text"] for chunk in main)
	assert python
	assert all("```python" in chunk["text"] for chunk in python)
	assert not {chunk["id"] for chunk in main} & {chunk["id"] for chunk in python}


#
# Function reference
#
def test_reference_page_keeps_title_and_version() -> None:
	# act
	record = _reference_record()

	# assert
	assert record["function_name"] == "sits_regularize"
	assert record["title"] == "Build a regular data cube from an irregular one"
	assert record["version"] == "1.5.4"
	assert record["source"] == "function"


def test_each_argument_is_a_chunk_with_its_anchor() -> None:
	# act
	chunks = data.function_chunks([_reference_record()], 1000)

	# assert
	period = next(chunk for chunk in chunks if chunk["section_id"] == "arg-period")
	assert period["url"] == f"{REFERENCE_URL}#arg-period"
	assert period["page_url"] == REFERENCE_URL
	assert period["function_name"] == "sits_regularize"
	assert period["text"].startswith(
		"sits_regularize: Build a regular data cube from an irregular one"
		" › Argument period\n\nISO8601"
	)
