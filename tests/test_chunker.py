#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Unit tests for the recursive text chunker."""

from sitsmcp import data


def test_empty_input_returns_no_chunks():
	"""Test that an empty input returns no chunks."""
	# arrange
	text = "   \n  "

	# act
	chunks = data.chunk_text(text)

	# assert
	assert chunks == []


def test_short_text_is_a_single_chunk():
	"""Test that a short text is a single chunk."""
	# arrange
	text = "SITS classifies satellite image time series."

	# act
	chunks = data.chunk_text(text, chunk_size=1000, chunk_overlap=200)

	# assert
	assert chunks == [text]


def _long_text() -> str:
	"""Build a long text with a long paragraph and short paragraphs."""
	sentence = "Sentence {} of a long paragraph."

	first = " ".join(sentence.format(index) for index in range(300))
	second = "\n\n".join(sentence.format(index) for index in range(30))

	return first + "\n\n" + second


def test_chunks_never_exceed_chunk_size():
	"""Test that no chunk exceeds the chunk size, overlap included."""
	# arrange
	text = _long_text()

	# act
	chunks = data.chunk_text(text, chunk_size=1000, chunk_overlap=200)

	# assert
	assert max(len(chunk) for chunk in chunks) <= 1000


def test_overlap_is_carried_once_between_neighbours():
	"""Test that the end of a chunk is carried into the next chunk."""
	# arrange
	text = _long_text()

	# act
	chunks = data.chunk_text(text, chunk_size=1000, chunk_overlap=200)

	# assert
	assert len(chunks) > 1

	for index in range(len(chunks) - 1):
		assert chunks[index].strip()[-50:] in chunks[index + 1]
