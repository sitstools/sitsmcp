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

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	"""sitsmcp configuration."""

	model_config = SettingsConfigDict(
		env_prefix="SITSMCP_",
		env_file=".env",
		env_file_encoding="utf-8",
		extra="ignore",
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
	"""Overlap between consecutive chunks, in characters."""

	#
	# Database - Search
	#
	top_k: int = 5
	"""Number of results returned by a search tool."""

	candidate_k: int = 20
	"""Per-retriever candidate pool size before fusion."""

	rrf_k: int = 60
	"""Reciprocal Rank Fusion constant."""
