#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Shared test helpers."""

import hashlib

import pytest

from sitsmcp import operations, rerank
from sitsmcp.config import Settings

#
# Constants
#
EMBEDDING_DIM = 8
"""Dimensionality of the stub embeddings."""


#
# Public functions
#
def _fake_embed(
	model_name: str, texts: list[str], batch_size: int = 16
) -> list[list[float]]:
	"""Stub embedding.

	Args:
		model_name (str): The embedding model identifier.

		texts (list[str]): The text to embed.

		batch_size (int): Ignored.

	Returns:
		list[list[float]]: The embedded texts.
	"""
	vectors = []

	# embed the text
	for text in texts:
		digest = hashlib.sha256(text.encode()).digest()

		vectors.append([digest[index] / 255.0 for index in range(EMBEDDING_DIM)])

	# return vectors
	return vectors


@pytest.fixture(autouse=True)
def _no_reranker(monkeypatch: pytest.MonkeyPatch):
	"""Keep the real Cohere key and any real model out of unit tests."""
	# forget the settings and the rerankers
	operations._settings.cache_clear()
	rerank._shared_rerankers.clear()

	# turn reranking off
	monkeypatch.setenv("SITSMCP_KEY_RERANKER_MODEL", "")
	monkeypatch.setenv("SITSMCP_DEFAULT_RERANKER_MODEL", "")
	monkeypatch.delenv("SITSMCP_RERANKER_API_KEY", raising=False)
	monkeypatch.delenv("SITSMCP_RERANKER_MODEL", raising=False)

	# read no .env
	monkeypatch.setitem(Settings.model_config, "env_file", None)

	# keep the access settings out
	for name in ("API_KEYS", "RATE_LIMIT_OPEN", "RATE_LIMIT_KEY"):
		monkeypatch.delenv(f"SITSMCP_{name}", raising=False)


@pytest.fixture
def restore_mcp():
	"""Give back the middleware list of the shared server."""
	from sitsmcp import operations, server

	# remember the list
	saved = list(server.mcp.middleware)
	operations._settings.cache_clear()
	operations._engine.cache_clear()

	yield

	# restore the list
	server.mcp.middleware[:] = saved
	operations._settings.cache_clear()
	operations._engine.cache_clear()
