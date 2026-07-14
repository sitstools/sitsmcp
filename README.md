## sitsmcp

[MCP](https://modelcontextprotocol.io) server exposing SITS R package documentation, function reference, and references

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/m3nin0-labs/sitsmcp.git
cd sitsmcp
uv sync
```

## Usage

Build the database, index the content, then serve it over MCP:

```bash
uv run sitsmcp db init         # 1. create the database schema
uv run sitsmcp index all       # 2. index the reference, book and articles
uv run sitsmcp serve           # 3. start the MCP server over HTTP
```

The server listens on `http://127.0.0.1:8000/mcp` by default. Run `uv run sitsmcp workflow` for the guided setup, or use `index reference`, `index docs` and `index references` to index sources individually.

## Configuration

Point your MCP client at the running server URL:

```json
{
  "mcpServers": {
    "sitsmcp": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Settings are optional and read from environment variables (prefix `SITSMCP_`) or a `.env` file. The most common are the bind address, `SITSMCP_HOST` (default `127.0.0.1`) and `SITSMCP_PORT` (default `8000`). See [`.env.example`](.env.example) for the full list and defaults.

## MCP Tools

The `sitsmcp` currently implements the following tools:

| Tool | Purpose |
| --- | --- |
| `sits_get_function` | Full reference documentation for a named SITS function. |
| `sits_search_docs` | Semantic search over the SITS book. |
| `sits_search_references` | Search academic articles that use SITS. |

## Development

```bash
uv sync --extra dev
uv run pytest          # run the test suite
uv run ruff check .    # lint
uv run mypy sitsmcp    # type-check
```

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes, and ensure tests, linting and type checks pass before submitting a pull request.

## License

`sitsmcp` is distributed under the GPL-2.0-only license. See [LICENSE](./LICENSE) for the full text.
