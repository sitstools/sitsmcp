#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - ASGI entrypoint for production (gunicorn/uvicorn)."""

from sitsmcp import access
from sitsmcp.config import Settings
from sitsmcp.server import mcp

#
# ASGI application served by gunicorn
#
# fail at boot on a bad configuration
settings = Settings()

# limit the tool calls, and check the keys
access.install(mcp, settings)

# create the ASGI application
app = mcp.http_app(stateless_http=True, middleware=access.http_middleware(settings))
