#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Application configuration via pydantic-settings."""

from pathlib import Path
from typing import Self

from pydantic import NonNegativeInt, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	"""sitsmcp configuration."""

	model_config = SettingsConfigDict(
		env_prefix="SITSMCP_",
		env_file=".env",
		env_file_encoding="utf-8",
		extra="ignore",
		hide_input_in_errors=True,
	)

	#
	# Server
	#
	host: str = "127.0.0.1"
	"""Host interface the MCP HTTP server binds to."""

	port: int = 8000
	"""Port the MCP HTTP server listens on."""

	#
	# Database
	#
	db_path: Path = Path("sitsmcp.db")
	"""Path to the SQLite database file."""

	embedding_model: str = "BAAI/bge-large-en-v1.5"
	"""fastembed model name used to embed chunks and queries."""

	embedding_dim: int = 1024
	"""Dimensionality of the embedding vectors (must match the model)."""

	embedding_batch_size: int = 16
	"""Number of texts embedded at once while indexing (larger uses more memory)."""

	#
	# Content
	#
	sitsbook_url: str = "https://e-sensing.github.io/sitsbook"
	"""Base URL of the SITS book (Quarto site)."""

	sits_reference_url: str = "https://e-sensing.github.io/sits"
	"""Base URL of the SITS function reference (pkgdown site)."""

	references_path: Path = Path("data/references.json")
	"""Path to the JSON list of reference articles to index."""

	#
	# Database - Chunking
	#
	chunk_size: int = 1000
	"""Maximum size of each text chunk, in characters."""

	chunk_overlap: int = 200
	"""Overlap between consecutive chunks, in characters.

	Book pages do not use it: they are split by section and never overlap. It
	applies only to callers of ``data.chunk_text`` that pass it.
	"""

	#
	# Database - Search
	#
	top_k: int = 5
	"""Number of results returned by a search tool."""

	candidate_k: int = 20
	"""Per-retriever candidate pool size before fusion."""

	rrf_k: int = 60
	"""Reciprocal Rank Fusion constant."""

	#
	# Database - Reranking
	#
	key_reranker_model: str = "cohere/rerank-v5.0-fast"
	"""Reranker of key holders, as ``<provider>/<model>`` (``cohere`` or ``fastembed``).

	Clients with an API key, and calls in process, try it first and fall back to
	``default_reranker_model`` when it fails. Empty is off.
	"""

	default_reranker_model: str = "fastembed/Xenova/ms-marco-MiniLM-L-6-v2"
	"""Reranker of everyone, as ``fastembed/<model>``. It is off if empty.

	Open use (no API key) gets only this one, and key holders get it when the
	key reranker fails or is off. This open use mode never calls a paid service.
	"""

	reranker_model: str | None = None
	"""The old name of ``key_reranker_model``. Any value, even empty, fails at start.

	It is read only to catch a ``SITSMCP_RERANKER_MODEL`` left in the
	environment or in ``.env``.
	"""

	reranker_api_key: SecretStr | None = None
	"""API key of the ``cohere`` reranker. Without it the key reranker is off."""

	reranker_candidates: int = 20
	"""Number of fused hits the reranker sees. It must be at least ``top_k``."""

	reranker_timeout: float = 2.0
	"""Seconds allowed for each network phase of a Cohere call.

	Connecting takes at most 1 s. Reading and writing each take up to this
	many. It is not a limit on the whole call.
	"""

	#
	# Access
	#
	api_keys: SecretStr | None = None
	"""API keys as ``name:key`` pairs, comma-separated. None means open use only."""

	rate_limit_open: NonNegativeInt = 30
	"""Tool calls a minute of a client IP without a key. 0 is no limit."""

	rate_limit_key: NonNegativeInt = 300
	"""Tool calls a minute of an API key. 0 is no limit."""

	#
	# Validation
	#
	@model_validator(mode="after")
	def _check_reranker(self) -> Self:
		"""Check the reranker configuration."""
		from sitsmcp.rerank import check_default_reranker, parse_reranker_model

		# raise an error if the reranker model is not None
		if self.reranker_model is not None:
			# raise an error if the reranker model is not None
			raise ValueError(
				"SITSMCP_RERANKER_MODEL was renamed: use SITSMCP_KEY_RERANKER_MODEL "
				"(key holders) and SITSMCP_DEFAULT_RERANKER_MODEL (everyone)"
			)

		# check the key reranker model
		if self.key_reranker_model:
			parse_reranker_model(self.key_reranker_model)

		# check the default reranker model
		check_default_reranker(self.default_reranker_model)

		# check the reranker candidates
		if self.reranker_candidates < self.top_k:
			raise ValueError(
				f"reranker_candidates ({self.reranker_candidates}) "
				f"must be at least top_k ({self.top_k})"
			)

		# check the reranker timeout
		if self.reranker_timeout <= 0:
			raise ValueError(
				f"reranker_timeout ({self.reranker_timeout}) must be above 0"
			)

		# parse the API keys
		from sitsmcp.access import parse_api_keys

		if self.api_keys is not None:
			parse_api_keys(self.api_keys.get_secret_value())

		# return settings
		return self
