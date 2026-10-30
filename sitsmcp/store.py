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
import sqlite3
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import orjson
import pydash as py_
import sqlite_vec
from fastembed import TextEmbedding
from sqlalchemy import Connection, Engine, Row, RowMapping, create_engine, event, text
from sqlalchemy.pool import ConnectionPoolEntry
from tqdm import tqdm

#
# Constants
#
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
"""Regular expression to match word tokens used to build safe FTS5 queries."""

CHUNK_COLUMNS: tuple[str, ...] = (
	"page_url",
	"chapter_number",
	"chapter_title",
	"section_id",
	"section_number",
	"section_title",
	"language",
)
"""Nullable TEXT columns of ``chunks`` that locate a chunk in the book."""

FUNCTION_COLUMNS: tuple[str, ...] = ("version", "sections", "arguments")
"""Nullable TEXT columns of ``functions``."""

SCHEMA_STATEMENTS = (
	"""
	CREATE TABLE IF NOT EXISTS chunks (
		id INTEGER PRIMARY KEY,
		chunk_id TEXT UNIQUE,
		source TEXT NOT NULL,
		title TEXT,
		url TEXT,
		function_name TEXT,
		page_url TEXT,
		chapter_number TEXT,
		chapter_title TEXT,
		section_id TEXT,
		section_number TEXT,
		section_title TEXT,
		language TEXT,
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
		text TEXT NOT NULL,
		version TEXT,
		sections TEXT,
		arguments TEXT
	)
	""",
	"""
	CREATE TABLE IF NOT EXISTS index_info (
		source TEXT PRIMARY KEY,
		indexed_at TEXT NOT NULL,
		built_at TEXT,
		version TEXT,
		chunks INTEGER NOT NULL
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


@lru_cache(maxsize=16)
def _table_columns(
	engine: Engine, table: str, wanted: tuple[str, ...]
) -> tuple[str, ...]:
	"""Get the columns a table of a database has.

	Args:
		engine (Engine): The database engine.

		table (str): The table to inspect.

		wanted (tuple[str, ...]): The columns to look for.

	Returns:
		tuple[str, ...]: The wanted columns present, in their order.
	"""
	# read the columns
	with engine.connect() as conn:
		existing = _existing_columns(conn, table)

	# return!
	return tuple(column for column in wanted if column in existing)


def _existing_columns(conn: Connection, table: str) -> set[str]:
	"""Get the names of the columns a table has.

	Args:
		conn (Connection): An open database connection.

		table (str): The table to inspect.

	Returns:
		set[str]: The column names.
	"""
	return {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}


def _add_missing_columns(conn: Connection, table: str, wanted: tuple[str, ...]) -> None:
	"""Add columns to a table, as nullable TEXT.

	Args:
		conn (Connection): An open database connection, inside a transaction.

		table (str): The table to change.

		wanted (tuple[str, ...]): The columns the table should have; always a
			constant.
	"""
	existing = _existing_columns(conn, table)

	# add each missing column
	for column in wanted:
		if column not in existing:
			conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} TEXT"))


def _column_suffix(columns: tuple[str, ...]) -> str:
	"""Build a comma-separated list of column names.

	Args:
		columns (tuple[str, ...]): The column names.

	Returns:
		str: For example ``", version, sections"``.
	"""
	return "".join(f", {column}" for column in columns)


def _load_list(raw: str | None) -> list[dict]:
	"""Parse a JSON list column.

	Args:
		raw (str | None): The stored JSON text, or None.

	Returns:
		list[dict]: The decoded list, empty when the column is ``NULL``.
	"""
	# return the list or an empty one
	return orjson.loads(raw) if raw else []


def _dump_list(values: list[dict] | None) -> str:
	"""Serialize a list as the text of a JSON list column.

	Args:
		values (list[dict] | None): The list, or None.

	Returns:
		str: The JSON text.
	"""
	# return the JSON text
	return orjson.dumps(values or []).decode()


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


def _fuse_rrf(ranked_lists: list[list[int]], rrf_k: int) -> list[tuple[int, float]]:
	"""Fuse several ranked id lists with Reciprocal Rank Fusion.

	Args:
		ranked_lists (list[list[int]]): One ordered id list per retriever (best first).

		rrf_k (int): The RRF damping constant.

	Returns:
		list[tuple[int, float]]: Row ids with their fused scores, best first.
	"""
	# start every id seen by any retriever at zero
	scores = {row_id: 0.0 for ranked in ranked_lists for row_id in ranked}

	# iterate over the ranked lists
	for ranked in ranked_lists:
		# iterate over the ranked row ids
		for rank, row_id in enumerate(ranked):
			# update the scores
			scores[row_id] += 1.0 / (rrf_k + rank)

	# return the sorted (row id, score) pairs
	return sorted(
		scores.items(),
		key=lambda item: item[1],
		reverse=True,
	)


def _fetch_rows(engine: Engine, ranked: list[tuple[int, float]]) -> list[dict]:
	"""Fetch chunk rows by id.

	Args:
		engine (Engine): The database engine.

		ranked (list[tuple[int, float]]): The ordered (row id, score) pairs.

	Returns:
		list[dict]: Chunk dictionaries, each with its ``score``.
	"""
	row_ids = [row_id for row_id, _ in ranked]

	if not row_ids:
		return []

	# build sql row id placeholders
	placeholders = ",".join(f":id_{index}" for index in range(len(row_ids)))

	# build the SQL query, with the section columns the database has
	# column names come from the constant, never from input
	present = _table_columns(engine, "chunks", CHUNK_COLUMNS)
	sql = text(
		"SELECT id, chunk_id, text, title, url, source, function_name"
		f"{_column_suffix(present)} "
		f"FROM chunks WHERE id IN ({placeholders})"
	)

	# build the parameters
	params = {f"id_{index}": row_id for index, row_id in enumerate(row_ids)}

	# execute the query
	with engine.connect() as conn:
		rows = conn.execute(
			statement=sql,
			parameters=params,
		).fetchall()

	# build the by_id dictionary
	by_id = {row.id: _chunk_record(row, present) for row in rows}

	# return
	return [
		{**by_id[row_id], "score": float(score)}
		for row_id, score in ranked
		if row_id in by_id
	]


def _chunk_record(row: Row, present: tuple[str, ...]) -> dict:
	"""Turn a chunk row into a chunk dictionary.

	Args:
		row (Row): A row selected by ``_fetch_rows``.

		present (tuple[str, ...]): The ``CHUNK_COLUMNS`` the row holds.

	Returns:
		dict: The chunk, without its score.
	"""
	return {
		"chunk_id": row.chunk_id,
		"text": row.text,
		"title": row.title,
		"url": row.url,
		"source": row.source,
		"function_name": row.function_name,
		**{
			column: getattr(row, column) if column in present else None
			for column in CHUNK_COLUMNS
		},
	}


def _function_record(row: RowMapping, present: tuple[str, ...]) -> dict:
	"""Convert a function row into a function dictionary.

	Args:
		row (RowMapping): A row selected by ``get_function``.

		present (tuple[str, ...]): The ``FUNCTION_COLUMNS`` the row holds.

	Returns:
		dict: The function record.
	"""
	return {
		"name": row["name"],
		"title": row["title"],
		"url": row["url"],
		"text": row["text"],
		"version": row["version"] if "version" in present else None,
		"sections": _load_list(row["sections"]) if "sections" in present else [],
		"arguments": _load_list(row["arguments"]) if "arguments" in present else [],
	}


def _has_table(engine: Engine, name: str) -> bool:
	"""Check whether the database has a table.

	Args:
		engine (Engine): The database engine.

		name (str): The table name.

	Returns:
		bool: Whether the table exists.
	"""
	with engine.connect() as conn:
		found = conn.execute(
			text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = :name"),
			{"name": name},
		).first()

	# return!
	return found is not None


def _recorded_sources(engine: Engine) -> dict[str, dict]:
	"""Get the rows of ``index_info``, by source.

	Args:
		engine (Engine): The database engine.

	Returns:
		dict[str, dict]: The rows of each recorded source.
	"""
	if not _has_table(engine, "index_info"):
		return {}

	# read the rows
	with engine.connect() as conn:
		result = conn.execute(
			text("SELECT source, indexed_at, built_at, version, chunks FROM index_info")
		).mappings()
		records = [dict(row) for row in result]

	# return!
	return py_.key_by(records, "source")


def _unrecorded_source(source: str) -> dict:
	"""Create the ``index_info`` row of a source indexed before the table existed.

	Args:
		source (str): The collection tag.

	Returns:
		dict: The row, with no dates and no version.
	"""
	# return!
	return {
		"source": source,
		"indexed_at": None,
		"built_at": None,
		"version": None,
	}


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
	def _load_extension(
		dbapi_conn: sqlite3.Connection, _record: ConnectionPoolEntry
	) -> None:
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

		# add the columns an older database lacks
		_add_missing_columns(conn, "chunks", CHUNK_COLUMNS)
		_add_missing_columns(conn, "functions", FUNCTION_COLUMNS)

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

	# the columns may have changed
	_table_columns.cache_clear()


def chunk_counts(engine: Engine) -> dict[str, int]:
	"""Get the number of indexed chunks of each source.

	Args:
		engine (Engine): The database engine.

	Returns:
		dict[str, int]: The number of chunks of each source.
	"""
	# count the chunks
	with engine.connect() as conn:
		rows = conn.execute(
			text("SELECT source, count(*) FROM chunks GROUP BY source")
		).fetchall()

	# return!
	return {row[0]: row[1] for row in rows}


def record_index(
	engine: Engine,
	source: str,
	chunks: int,
	built_at: str | None,
	version: str | None,
	now: datetime | None = None,
) -> None:
	"""Record that a source was indexed, replacing its previous row.

	Args:
		engine (Engine): The database engine.

		source (str): The collection tag (e.g., ``book``).

		chunks (int): The number of chunks of the source.

		built_at (str | None): When the upstream site was built.

		version (str | None): The package version the source documents.

		now (datetime | None): The time of the run.
	"""
	# stamp the run in UTC
	indexed_at = (now or datetime.now(UTC)).astimezone(UTC)
	with engine.begin() as conn:
		conn.execute(
			statement=text(
				"""
				INSERT INTO index_info (source, indexed_at, built_at, version, chunks)
				VALUES (:source, :indexed_at, :built_at, :version, :chunks)
				ON CONFLICT(source) DO UPDATE SET
					indexed_at = excluded.indexed_at,
					built_at = excluded.built_at,
					version = excluded.version,
					chunks = excluded.chunks
				"""
			),
			parameters={
				"source": source,
				"indexed_at": indexed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
				"built_at": built_at,
				"version": version,
				"chunks": chunks,
			},
		)


def index_info(engine: Engine) -> list[dict]:
	"""Get when each source was indexed.

	Args:
		engine (Engine): The database engine.

	Returns:
		list[dict]: One dictionary per source, sorted by source.
	"""
	# read the recorded rows and the live chunk counts
	recorded = _recorded_sources(engine)
	counts = chunk_counts(engine)

	# list every source that has a row or chunks
	sources = sorted(set(recorded) | set(counts))

	# return each row with the live count
	return [
		{
			**recorded.get(source, _unrecorded_source(source)),
			"chunks": counts.get(source, 0),
		}
		for source in sources
	]


def embed_texts(
	model_name: str, texts: list[str], batch_size: int = 16
) -> list[list[float]]:
	"""Embed texts with fastembed.

	Args:
		model_name (str): The fastembed model identifier.

		texts (list[str]): The texts to embed.

		batch_size (int): The number of texts embedded at once.

	Returns:
		list[list[float]]: One embedding vector per input text.
	"""
	# get the model
	model = _get_model(model_name)

	# return the embeddings
	return [vector.tolist() for vector in model.embed(texts, batch_size=batch_size)]


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


def store_chunks(
	engine: Engine, model_name: str, chunks: list[dict], batch_size: int = 16
) -> int:
	"""Embed and store chunks.

	Args:
		engine (Engine): The database engine.

		model_name (str): The embedding model identifier.

		chunks (list[dict]): Chunk dictionaries from ``sitsmcp.data``.

		batch_size (int): The number of texts embedded at once.

	Returns:
		int: The number of chunks stored.
	"""
	if not chunks:
		return 0

	# embed the chunks
	embeddings = embed_texts(
		model_name=model_name,
		texts=[chunk["text"] for chunk in chunks],
		batch_size=batch_size,
	)

	# build the SQL query
	# column names come from the constant, never from input
	columns = ", ".join(CHUNK_COLUMNS)
	values = ", ".join(f":{column}" for column in CHUNK_COLUMNS)

	insert_chunk = text(
		f"""
		INSERT OR IGNORE INTO chunks
			(chunk_id, source, title, url, function_name, {columns}, text)
		VALUES (:chunk_id, :source, :title, :url, :function_name, {values}, :text)
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
					**{column: chunk.get(column) for column in CHUNK_COLUMNS},
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
		INSERT INTO functions (name, title, url, text, version, sections, arguments)
		VALUES (:name, :title, :url, :text, :version, :sections, :arguments)
		ON CONFLICT(name) DO UPDATE SET
			title = excluded.title,
			url = excluded.url,
			text = excluded.text,
			version = excluded.version,
			sections = excluded.sections,
			arguments = excluded.arguments
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
					"version": record.get("version"),
					"sections": _dump_list(record.get("sections")),
					"arguments": _dump_list(record.get("arguments")),
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
		list[dict]: ``top_k`` chunk dictionaries with their fused scores,
			best first.
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
		ranked=fused[:top_k],
	)


def get_function(engine: Engine, name: str) -> dict | None:
	"""Get the full documentation for a function, if present.

	Args:
		engine (Engine): The database engine.

		name (str): The exact function name (e.g., ``sits_classify``).

	Returns:
		dict: The function documentation, or None if unknown.
	"""
	present = _table_columns(engine, "functions", FUNCTION_COLUMNS)

	with engine.connect() as conn:
		row = (
			conn.execute(
				statement=text(
					f"SELECT name, title, url, text{_column_suffix(present)} "
					"FROM functions WHERE name = :name"
				),
				parameters={"name": name},
			)
			.mappings()
			.fetchone()
		)

	if row is None:
		return None

	# return the function record
	return _function_record(row, present)
