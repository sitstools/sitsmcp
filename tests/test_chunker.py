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


def test_long_text_splits_and_respects_chunk_size():
	"""Test that a long text splits and respects the chunk size."""
	# arrange
	paragraph = ". ".join(f"sentence number {i}" for i in range(200))

	# act
	chunks = data.chunk_text(paragraph, chunk_size=100, chunk_overlap=20)

	# assert
	assert len(chunks) > 1
	assert all(len(chunk) <= 100 + 20 + 1 for chunk in chunks)
