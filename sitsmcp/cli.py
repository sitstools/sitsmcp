#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - CLI module."""

import typer
from rich.console import Console
from rich.panel import Panel
from sqlalchemy import Engine

from sitsmcp import data, store
from sitsmcp.config import Settings

#
# Constants
#
console = Console()
"""Shared Rich console for CLI output."""

WORKFLOW = """[bold cyan]sitsmcp setup workflow[/]

  [bold white]1.[/] [blue]Create the database[/]
     [green]$ sitsmcp db init[/]

  [bold white]2.[/] [blue]Index the content[/] [dim](downloads the embedding model once)[/]
     [green]$ sitsmcp index reference[/]     [dim]# function reference (exact lookup)[/]
     [green]$ sitsmcp index docs[/]          [dim]# SITS book (semantic + keyword)[/]
     [green]$ sitsmcp index references[/]    [dim]# articles from data/references.json[/]
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


def _index_docs(engine: Engine, settings: Settings) -> int:
	"""Parse and store the SITS book.

	Args:
		engine (Engine): The database engine.

		settings (Settings): The active settings.

	Returns:
		int: The number of chunks stored.
	"""
	# parse documentation
	chunks = data.parse_documentation(
		settings.sitsbook_url, settings.chunk_size, settings.chunk_overlap
	)

	# delete existing chunks
	store.delete_source(engine, "book")

	# store chunks
	return store.store_chunks(
		engine=engine, model_name=settings.embedding_model, chunks=chunks
	)


def _index_reference(engine: Engine, settings: Settings) -> int:
	"""Parse and store the SITS function reference.

	Args:
		engine (Engine): The database engine.

		settings (Settings): The active settings.

	Returns:
		int: The number of function records stored.
	"""
	# parse reference
	records = data.parse_reference(settings.sits_reference_url)

	# store functions
	return store.store_functions(engine, records)


def _index_references(engine: Engine, settings: Settings) -> int:
	"""Parse and store the reference articles.

	Args:
		engine (Engine): The database engine.

		settings (Settings): The active settings.

	Returns:
		int: The number of chunks stored.
	"""
	# parse references
	chunks = data.parse_references(
		settings.references_path, settings.chunk_size, settings.chunk_overlap
	)

	# delete existing chunks
	store.delete_source(engine, "reference_article")

	# store chunks
	return store.store_chunks(
		engine=engine, model_name=settings.embedding_model, chunks=chunks
	)


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
	"""Index the SITS function reference (exact-name lookup)."""
	# prepare settings
	settings = Settings()

	# index reference
	count = _index_reference(_prepare(settings), settings)

	# inform user
	console.print(f"[bold green]Done![/] Indexed [cyan]{count}[/] functions.")


@index_app.command("references")
def index_references() -> None:
	"""Index the reference articles listed in data/references.json."""
	# prepare settings
	settings = Settings()

	# index references
	count = _index_references(_prepare(settings), settings)

	# inform user
	console.print(f"[bold green]Done![/] Indexed [cyan]{count}[/] article chunks.")


@index_app.command("all")
def index_all() -> None:
	"""Index the function reference, the SITS book and the articles."""
	# prepare settings
	settings = Settings()

	# prepare database
	engine = _prepare(settings)

	# index
	functions = _index_reference(engine, settings)
	docs = _index_docs(engine, settings)
	articles = _index_references(engine, settings)

	# inform user
	console.print(
		f"[bold green]Done![/] {functions} functions, {docs} book chunks, "
		f"{articles} article chunks."
	)


@app.command()
def serve() -> None:
	"""Start the MCP server (HTTP transport)."""
	from sitsmcp.server import main

	# prepare settings
	settings = Settings()

	# inform user
	console.print(
		f"[bold green]Serving[/] MCP over HTTP at "
		f"[cyan]http://{settings.host}:{settings.port}/mcp[/]."
	)

	# run the server
	main(settings.host, settings.port)
