#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - CLI module."""

from collections.abc import Callable

import httpx
import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import Engine

from sitsmcp import access, data, store
from sitsmcp.config import Settings
from sitsmcp.models import IndexInfo, SourceInfo
from sitsmcp.rerank import active_rerankers, reranker_status

#
# Constants
#
console = Console()
"""Shared Rich console for CLI output."""

WORKFLOW = """[bold cyan]sitsmcp setup workflow[/]

  [bold white]1.[/] [blue]Create the database[/]
     [green]$ sitsmcp db init[/]

  [bold white]2.[/] [blue]Index the content[/] [dim](downloads the embedding model once)[/]
     [green]$ sitsmcp index reference[/]     [dim]# function reference (lookup + search)[/]
     [green]$ sitsmcp index docs[/]          [dim]# SITS book (semantic + keyword)[/]
     [green]$ sitsmcp index references[/]    [dim]# citations from data/references.json (local)[/]
     [dim]or all at once:[/] [green]$ sitsmcp index all[/]

  [bold white]3.[/] [blue]Serve over MCP[/]
     [green]$ sitsmcp serve[/]
"""
"""Rich-formatted setup guide shown by the ``workflow`` command."""

#
# Typer application
#
app = typer.Typer(
	name="sitsmcp",
	help="MCP server for the SITS R package documentation and references.",
	no_args_is_help=True,
)
"""Top-level Typer application."""

db_app = typer.Typer(help="Database management commands.", no_args_is_help=True)
"""``sitsmcp db`` command group."""

index_app = typer.Typer(help="Index content into the database.", no_args_is_help=True)
"""``sitsmcp index`` command group."""

app.add_typer(db_app, name="db")
app.add_typer(index_app, name="index")


#
# Internal functions
#
def _prepare(settings: Settings) -> Engine:
	"""Create the engine and ensure the schema exists.

	Args:
		settings (Settings): The active settings.

	Returns:
		engine (Engine): A ready-to-use database engine.
	"""
	# create engine
	engine = store.create_db_engine(settings.db_path)

	# initialize database
	store.init_db(engine, settings.embedding_dim)

	# return engine
	return engine


def _access_status(settings: Settings) -> str:
	"""Describe the API keys and the rate limits.

	Args:
		settings (Settings): The active settings.

	Returns:
		str: The line ``serve`` prints.
	"""
	raw = settings.api_keys.get_secret_value() if settings.api_keys else ""

	# build the per-minute string
	def per_minute(limit: int) -> str:
		return f"{limit}/min" if limit else "no limit"

	# build the access status string
	return (
		f"Access: {len(access.parse_api_keys(raw))} API keys; limits "
		f"{per_minute(settings.rate_limit_open)} open, "
		f"{per_minute(settings.rate_limit_key)} per key"
	)


def _build_date(
	source: str, fetch: Callable[[httpx.Client, str], str | None], base_url: str
) -> str | None:
	"""Read the build date of a site.

	Args:
		source (str): The collection tag.

		fetch (Callable[[httpx.Client, str], str | None]): The reader of the date.

		base_url (str): The base URL of the site.

	Returns:
		str | None: The build date, or ``None`` if unknown.
	"""
	reason = "no date found"

	# try to fetch the date
	try:
		with httpx.Client(timeout=data.HTTP_TIMEOUT, follow_redirects=True) as client:
			built_at = fetch(client, base_url)

		if built_at:
			return built_at

	except httpx.HTTPError as error:
		reason = str(error) or type(error).__name__

	# inform user
	console.print(f"[yellow]Build date of {source} unknown: {reason}[/]")

	# return!
	return None


def _index_docs(engine: Engine, settings: Settings) -> int:
	"""Parse and store the SITS book.

	Args:
		engine (Engine): The database engine.

		settings (Settings): The active settings.

	Returns:
		int: The number of chunks stored.
	"""
	# read the build date first
	built_at = _build_date(
		source="book", fetch=data.fetch_book_build, base_url=settings.sitsbook_url
	)

	# parse documentation
	chunks = data.parse_documentation(
		base_url=settings.sitsbook_url,
		chunk_size=settings.chunk_size,
		chunk_overlap=settings.chunk_overlap,
	)

	# delete existing chunks
	store.delete_source(engine, "book")

	# store chunks
	stored = store.store_chunks(
		engine=engine,
		model_name=settings.embedding_model,
		chunks=chunks,
		batch_size=settings.embedding_batch_size,
	)

	# record the run
	store.record_index(engine, "book", stored, built_at, None)

	# return!
	return stored


def _index_reference(engine: Engine, settings: Settings) -> tuple[int, int]:
	"""Parse and store the SITS function reference and its search chunks.

	Args:
		engine (Engine): The database engine.

		settings (Settings): The active settings.

	Returns:
		tuple[int, int]: The number of functions and of search chunks stored.
	"""
	# read the build date first
	built_at = _build_date(
		source="function",
		fetch=data.fetch_reference_build,
		base_url=settings.sits_reference_url,
	)

	# parse reference
	records = data.parse_reference(settings.sits_reference_url)

	# store functions
	functions = store.store_functions(engine, records)

	# delete existing chunks
	store.delete_source(engine, "function")

	# store chunks
	chunks = store.store_chunks(
		engine=engine,
		model_name=settings.embedding_model,
		chunks=data.function_chunks(records, settings.chunk_size),
		batch_size=settings.embedding_batch_size,
	)

	# record the run
	store.record_index(
		engine=engine,
		source="function",
		chunks=chunks,
		built_at=built_at,
		version=data.reference_version(records),
	)

	# return!
	return functions, chunks


def _index_references(engine: Engine, settings: Settings) -> int:
	"""Parse and store the reference articles.

	Args:
		engine (Engine): The database engine.

		settings (Settings): The active settings.

	Returns:
		int: The number of chunks stored.
	"""
	# parse references
	chunks = data.parse_references(settings.references_path)

	# delete existing chunks
	store.delete_source(engine, "reference_article")

	# store chunks
	stored = store.store_chunks(
		engine=engine,
		model_name=settings.embedding_model,
		chunks=chunks,
		batch_size=settings.embedding_batch_size,
	)

	# record the run
	store.record_index(engine, "reference_article", stored, None, None)

	# return!
	return stored


#
# Public functions — CLI commands
#
@app.command()
def workflow() -> None:
	"""Show the usage workflow - what to run and in what order."""
	console.print(Panel(WORKFLOW, border_style="blue", title="Usage Workflow"))


@db_app.command("init")
def db_init() -> None:
	"""Create the database schema (idempotent)."""
	# prepare settings
	settings = Settings()

	# prepare database
	_prepare(settings)

	# inform user
	console.print(
		f"[bold green]Done![/] Database ready at [cyan]{settings.db_path}[/]."
	)


@index_app.command("docs")
def index_docs() -> None:
	"""Index the SITS book (semantic + keyword search)."""
	# prepare settings
	settings = Settings()

	# index docs
	count = _index_docs(_prepare(settings), settings)

	# inform user
	console.print(f"[bold green]Done![/] Indexed [cyan]{count}[/] book chunks.")


@index_app.command("reference")
def index_reference() -> None:
	"""Index the SITS function reference (lookup + search)."""
	# prepare settings
	settings = Settings()

	# index reference
	functions, chunks = _index_reference(_prepare(settings), settings)

	# inform user
	console.print(
		f"[bold green]Done![/] Indexed [cyan]{functions}[/] functions "
		f"([cyan]{chunks}[/] search chunks)."
	)


@index_app.command("references")
def index_references() -> None:
	"""Index the reference articles listed in data/references.json."""
	# prepare settings
	settings = Settings()

	# index references
	count = _index_references(_prepare(settings), settings)

	# inform user
	console.print(f"[bold green]Done![/] Indexed [cyan]{count}[/] articles.")


@index_app.command("all")
def index_all() -> None:
	"""Index the function reference, the SITS book and the articles."""
	# prepare settings
	settings = Settings()

	# prepare database
	engine = _prepare(settings)

	# index
	functions, function_chunks = _index_reference(engine, settings)
	docs = _index_docs(engine, settings)
	articles = _index_references(engine, settings)

	# inform user
	console.print(
		f"[bold green]Done![/] {functions} functions ({function_chunks} search "
		f"chunks), {docs} book chunks, "
		f"{articles} articles."
	)


def _live_status(
	source: str,
	stored: dict | None,
	fetch: Callable[[httpx.Client, str], str | None],
	base_url: str,
) -> tuple[str, int]:
	"""Compare the stored build date of a source with the site's.

	Args:
		source (str): The source name.

		stored (dict | None): The source's row, or ``None`` if it has no chunks.

		fetch (Callable[[httpx.Client, str], str | None]): The reader of the live date.

		base_url (str): The base URL of the site.

	Returns:
		tuple[str, int]: The words to print and the exit code of the source.
	"""
	# a source without chunks is stale
	if stored is None or not stored["chunks"]:
		return "not indexed", 1

	# a source without a stored date cannot be compared
	if not stored["built_at"]:
		return "unknown (index has no build date)", 1

	# fetch the live date
	try:
		with httpx.Client(timeout=data.HTTP_TIMEOUT, follow_redirects=True) as client:
			live = fetch(client, base_url)
	except httpx.HTTPError as error:
		return f"unknown (fetch failed: {str(error) or type(error).__name__})", 2
	except Exception as error:
		return f"unknown (check failed: {type(error).__name__})", 2

	# the site gives no date
	if not live:
		return "unknown (site gives no build date)", 1

	# any change means a rebuild
	if live != stored["built_at"]:
		return f"stale (index {stored['built_at']}, site {live})", 1

	# return!
	return "fresh", 0


@app.command()
def info(
	check: bool = typer.Option(
		False,
		"--check",
		help="Compare the stored build dates with the sites' (exit 0 fresh, "
		"1 stale, 2 a fetch failed).",
	),
) -> None:
	"""Show when each source was indexed, and with --check whether it is fresh."""
	# prepare settings
	settings = Settings()

	# if the database does not exist
	if not settings.db_path.exists():
		# inform user
		console.print(
			f"[red]No database at {settings.db_path}: run sitsmcp index all[/]"
		)

		# exit with error
		raise typer.Exit(code=1)

	# read the rows
	rows = store.index_info(store.create_db_engine(settings.db_path))
	report = IndexInfo(
		sources=[SourceInfo(**row) for row in rows],
		embedding_model=settings.embedding_model,
		**active_rerankers(settings),
	)

	# show the rows
	table = Table(title="Index")
	table.add_column("source", overflow="fold")
	table.add_column("chunks")
	table.add_column("indexed", min_width=20, no_wrap=True)
	table.add_column("built", min_width=20, no_wrap=True)
	table.add_column("sits")

	# add the rows to the table
	for source_row in report.sources:
		table.add_row(
			source_row.source,
			str(source_row.chunks),
			source_row.indexed_at or "unknown",
			source_row.built_at or "unknown",
			source_row.version or "",
		)

	# show the table
	console.print(table)
	console.print(f"embedding model: {report.embedding_model}")
	console.print(f"key reranker: {report.key_reranker or 'none'}")
	console.print(f"default reranker: {report.default_reranker or 'none'}")

	# if not checking, exit
	if not check:
		return

	# compare with the sites
	codes = []
	sites = (
		("book", data.fetch_book_build, settings.sitsbook_url),
		("function", data.fetch_reference_build, settings.sits_reference_url),
	)
	by_source = {row["source"]: row for row in rows}

	# compare with the sites
	for source, fetch, base_url in sites:
		# get the words and code
		words, code = _live_status(source, by_source.get(source), fetch, base_url)

		# add the code to the list
		codes.append(code)

		# show the words
		colour = {0: "green", 1: "yellow", 2: "red"}[code]

		# show the words
		console.print(f"{source}: [{colour}]{escape(words)}[/]")

	# a failed fetch hides the verdict, so it wins over stale
	raise typer.Exit(code=2 if 2 in codes else max(codes))


@app.command()
def serve() -> None:
	"""Start the MCP server (HTTP transport)."""
	from sitsmcp.server import main, mcp

	# prepare settings
	settings = Settings()
	access.install(mcp, settings)

	# inform user
	console.print(
		f"[bold green]Serving[/] MCP over HTTP at "
		f"[cyan]http://{settings.host}:{settings.port}/mcp[/]."
	)
	console.print(reranker_status(settings))
	console.print(_access_status(settings))

	# run the server
	main(settings.host, settings.port, access.http_middleware(settings))
