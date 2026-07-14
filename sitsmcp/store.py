#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Storage and retrieval module."""

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import sqlite_vec
from fastembed import TextEmbedding
from sqlalchemy import Engine, create_engine, event, text
from tqdm import tqdm

#
# Constants
#
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
"""Regular expression to match word tokens used to build safe FTS5 queries."""

SCHEMA_STATEMENTS = (
	"""
	CREATE TABLE IF NOT EXISTS chunks (
		id INTEGER PRIMARY KEY,
		chunk_id TEXT UNIQUE,
		source TEXT NOT NULL,
		title TEXT,
		url TEXT,
		function_name TEXT,
		text TEXT NOT NULL
	)
	""",
	"CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks(source)",
	"""
	CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
		text,
		content='chunks',
		content_rowid='id'
	)
	""",
	"""
	CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
		INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
	END
	""",
	"""
	CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
		INSERT INTO chunks_fts(chunks_fts, rowid, text)
		VALUES ('delete', old.id, old.text);
	END
	""",
	"""
	CREATE TABLE IF NOT EXISTS functions (
		name TEXT PRIMARY KEY,
		title TEXT,
		url TEXT,
		text TEXT NOT NULL
	)
	""",
)
"""Ordered DDL statements for the non-vector schema."""


#
# Internal functions
#
@lru_cache(maxsize=4)
def _get_model(model_name: str) -> TextEmbedding:
	"""Get an embedding model instance.

	Args:
		model_name (str): The fastembed model identifier.

	Returns:
		TextEmbedding: Embedding model instance.
	"""
	return TextEmbedding(model_name=model_name)


def _fts_query(query: str) -> str | None:
	"""Build FTS5 query.

	Args:
		query (str): Query text.

	Returns:
		str: FTS5 query string, or None if the query has no usable tokens.
	"""
	tokens = TOKEN_RE.findall(query.lower())

	# return query or none
	return " OR ".join(tokens) if tokens else None


def _vec_ranks(
	engine: Engine,
	model_name: str,
	query: str,
	source: str,
	candidate_k: int,
) -> list[int]:
	"""Rank chunk row ids by vector similarity (best first).

	Args:
		engine (Engine): The database engine.

		model_name (str): The embedding model identifier.

		query (str): Query text.

		source (str): The corpus to restrict the search to.

		candidate_k (int): The number of candidates to retrieve.

	Returns:
		list[int]: Chunk row ids ordered from most to least similar.
	"""
	# embed the query
	embedding = embed_texts(model_name=model_name, texts=[query])[0]

	# build the SQL query
	sql = text(
		"""
		SELECT chunk_rowid FROM vec_chunks
		WHERE embedding MATCH :emb AND k = :k AND source = :source
		ORDER BY distance
		"""
	)

	# execute the query
	with engine.connect() as conn:
		rows = conn.execute(
			statement=sql,
			parameters={
				"emb": sqlite_vec.serialize_float32(embedding),
				"k": candidate_k,
				"source": source,
			},
		).fetchall()

	# return the row ids
	return [row[0] for row in rows]


def _fts_ranks(engine: Engine, query: str, source: str, candidate_k: int) -> list[int]:
	"""Rank chunk row ids by FTS5 relevance (best first).

	Args:
		engine (Engine): The database engine.

		query (str): Query text.

		source (str): The corpus to restrict the search to.

		candidate_k (int): The number of candidates to retrieve.

	Returns:
		list[int]: Chunk row ids ordered from most to least relevant.
	"""
	fts = _fts_query(query)

	# return empty list if no usable tokens
	if not fts:
		return []

	# build the SQL query
	sql = text(
		"""
		SELECT c.id FROM chunks_fts
		JOIN chunks c ON c.id = chunks_fts.rowid
		WHERE chunks_fts MATCH :q AND c.source = :source
		ORDER BY bm25(chunks_fts)
		LIMIT :k
		"""
	)

	# execute the query
	with engine.connect() as conn:
		rows = conn.execute(
			statement=sql,
			parameters={
				"q": fts,
				"source": source,
				"k": candidate_k,
			},
		).fetchall()

	# return the row ids
	return [row[0] for row in rows]


def _fuse_rrf(ranked_lists: list[list[int]], rrf_k: int) -> list[int]:
	"""Fuse several ranked id lists with Reciprocal Rank Fusion.

	Args:
		ranked_lists (list[list[int]]): One ordered id list per retriever (best first).

		rrf_k (int): The RRF damping constant.

	Returns:
		Row ids ordered by fused score, best first.
	"""
	scores = defaultdict(float)

	# iterate over the ranked lists
	for ranked in ranked_lists:
		# iterate over the ranked row ids
		for rank, row_id in enumerate(ranked):
			# update the scores
			scores[row_id] += 1.0 / (rrf_k + rank)

	# return the sorted row ids
	return sorted(
		scores,
		key=lambda row_id: scores[row_id],
		reverse=True,
	)


def _fetch_rows(engine: Engine, row_ids: list[int]) -> list[dict]:
	"""Fetch chunk rows by id.

	Args:
		engine (Engine): The database engine.

		row_ids (list[int]): The ordered chunk row ids to fetch.

	Returns:
		list[dict]: Chunk dictionaries.
	"""
	if not row_ids:
		return []

	# build sql row id placeholders
	placeholders = ",".join(f":id_{i}" for i in range(len(row_ids)))

	# build the SQL query
	sql = text(
		f"SELECT id, text, title, url, source FROM chunks WHERE id IN ({placeholders})"
	)

	# build the parameters
	params = {f"id_{i}": row_id for i, row_id in enumerate(row_ids)}

	# execute the query
	with engine.connect() as conn:
		rows = conn.execute(
			statement=sql,
			parameters=params,
		).fetchall()

	# build the by_id dictionary
	by_id = {
		row.id: {
			"text": row.text,
			"title": row.title,
			"url": row.url,
			"source": row.source,
		}
		for row in rows
	}

	# return
	return [by_id[row_id] for row_id in row_ids if row_id in by_id]


#
# Public functions
#
def create_db_engine(db_path: Path) -> Engine:
	"""Create a SQLite engine with the `sqlite-vec` extension loaded.

	The extension is loaded on every new DB-API connection so vector virtual
	tables are available across the connection pool.

	Args:
		db_path (Path): Path to the SQLite database file.

	Returns:
		Engine: A configured SQLAlchemy engine.
	"""
	# create the engine
	engine = create_engine(url=f"sqlite+pysqlite:///{db_path}")

	# listen for new connections
	@event.listens_for(engine, "connect")
	def _load_extension(dbapi_conn, _record):
		# enable the extension
		dbapi_conn.enable_load_extension(True)

		# load the extension
		sqlite_vec.load(dbapi_conn)

		# disable the extension
		dbapi_conn.enable_load_extension(False)

	# return the engine
	return engine


def init_db(engine: Engine, embedding_dim: int) -> None:
	"""Initialize the database with the schema.

	Args:
		engine (Engine): The database engine.

		embedding_dim (int): Dimensionality of the stored embedding vectors.
	"""
	# execute the schema statements
	with engine.begin() as conn:
		for statement in SCHEMA_STATEMENTS:
			conn.execute(text(statement))

		conn.execute(
			text(
				f"""
				CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
					chunk_rowid INTEGER PRIMARY KEY,
					source TEXT,
					embedding float[{embedding_dim}]
				)
				"""
			)
		)


def embed_texts(model_name: str, texts: list[str]) -> list[list[float]]:
	"""Embed texts locally with fastembed.

	Args:
		model_name (str): The fastembed model identifier.

		texts (list[str]): The texts to embed.

	Returns:
		list[list[float]]: One embedding vector per input text.
	"""
	# get the model
	model = _get_model(model_name)

	# return the embeddings
	return [vector.tolist() for vector in model.embed(texts)]


def delete_source(engine: Engine, source: str) -> None:
	"""Delete all chunks and their embeddings for a collection.

	Args:
		engine (Engine): The database engine.

		source (str): The collection tag to clear (e.g., ``book``).
	"""
	with engine.begin() as conn:
		# delete the embeddings
		conn.execute(
			statement=text(
				"DELETE FROM vec_chunks WHERE chunk_rowid IN "
				"(SELECT id FROM chunks WHERE source = :s)"
			),
			parameters={"s": source},
		)

		# delete the chunks
		conn.execute(
			statement=text("DELETE FROM chunks WHERE source = :s"),
			parameters={"s": source},
		)


def store_chunks(engine: Engine, model_name: str, chunks: list[dict]) -> int:
	"""Embed and store chunks for a collection.

	Args:
		engine (Engine): The database engine.

		model_name (str): The embedding model identifier.

		chunks (list[dict]): Chunk dictionaries from :mod:`sitsmcp.data`.

	Returns:
		int: The number of chunks stored.
	"""
	if not chunks:
		return 0

	# embed the chunks
	embeddings = embed_texts(
		model_name=model_name,
		texts=[chunk["text"] for chunk in chunks],
	)

	# build the SQL query
	insert_chunk = text(
		"""
		INSERT OR IGNORE INTO chunks
			(chunk_id, source, title, url, function_name, text)
		VALUES (:chunk_id, :source, :title, :url, :function_name, :text)
		"""
	)

	# build the SQL query
	insert_vec = text(
		"INSERT INTO vec_chunks (chunk_rowid, source, embedding) "
		"VALUES (:rowid, :source, :emb)"
	)

	# execute the queries
	stored = 0

	with engine.begin() as conn:
		for chunk, embedding in tqdm(
			iterable=zip(chunks, embeddings, strict=True),
			total=len(chunks),
			desc="Storing chunks",
			unit="chunk",
		):
			result = conn.execute(
				insert_chunk,
				{
					"chunk_id": chunk["id"],
					"source": chunk["source"],
					"title": chunk.get("title"),
					"url": chunk.get("url"),
					"function_name": chunk.get("function_name"),
					"text": chunk["text"],
				},
			)

			# check if the chunk was stored
			if result.rowcount == 0:
				continue

			# store the embedding
			conn.execute(
				insert_vec,
				{
					"rowid": result.lastrowid,
					"source": chunk["source"],
					"emb": sqlite_vec.serialize_float32(embedding),
				},
			)

			# increment the counter
			stored += 1

	# return the number of chunks stored
	return stored


def store_functions(engine: Engine, functions: list[dict]) -> int:
	"""Store function reference records for exact-name lookup.

	Args:
		engine (Engine): The database engine.

		functions (list[dict]): List of function dictionaries.

	Returns:
		int: The number of function records stored.
	"""
	if not functions:
		return 0

	# build the SQL query
	sql = text(
		"""
		INSERT INTO functions (name, title, url, text)
		VALUES (:name, :title, :url, :text)
		ON CONFLICT(name) DO UPDATE SET
			title = excluded.title,
			url = excluded.url,
			text = excluded.text
		"""
	)

	# execute the query
	with engine.begin() as conn:
		# iterate over the functions
		for record in functions:
			# execute the query
			conn.execute(
				statement=sql,
				parameters={
					"name": record["function_name"],
					"title": record.get("title"),
					"url": record.get("url"),
					"text": record["text"],
				},
			)

	# return!
	return len(functions)


def search_hybrid(
	engine: Engine,
	model_name: str,
	query: str,
	source: str,
	top_k: int,
	candidate_k: int,
	rrf_k: int,
) -> list[dict]:
	"""Hybrid vector + full-text search over a collection.

	Runs a vector KNN search and an FTS5 search, then fuses their rankings
	with Reciprocal Rank Fusion (RRF).

	Args:
		engine (Engine): The database engine.

		model_name (str): The embedding model identifier.

		query (str): The natural-language query.

		source (str): The collection to search (e.g., ``book``).

		top_k (int): The number of fused results to return.

		candidate_k (int): The per-retriever candidate pool size.

		rrf_k (int): The RRF damping constant.

	Returns:
		list[dict]: ``top_k`` chunk dictionaries, best first.
	"""
	# rank the chunks by vector similarity
	vec = _vec_ranks(
		engine=engine,
		model_name=model_name,
		query=query,
		source=source,
		candidate_k=candidate_k,
	)

	# rank the chunks by FTS5 relevance
	fts = _fts_ranks(
		engine=engine,
		query=query,
		source=source,
		candidate_k=candidate_k,
	)

	# fuse the rankings with RRF
	fused = _fuse_rrf(
		ranked_lists=[vec, fts],
		rrf_k=rrf_k,
	)

	# fetch the rows
	return _fetch_rows(
		engine=engine,
		row_ids=fused[:top_k],
	)


def get_function(engine: Engine, name: str) -> str | None:
	"""Return the full documentation for a function, if present.

	Args:
		engine (Engine): The database engine.

		name (str): The exact function name (e.g., ``sits_classify``).

	Returns:
		str: The function documentation, or None if unknown.
	"""
	with engine.connect() as conn:
		# execute the query
		row = conn.execute(
			statement=text("SELECT text FROM functions WHERE name = :name"),
			parameters={"name": name},
		).fetchone()

	# return the documentation
	return row[0] if row else None
