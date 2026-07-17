#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - ASGI entrypoint for production (gunicorn/uvicorn)."""

from sitsmcp.server import mcp

#
# ASGI application served by gunicorn
#
app = mcp.http_app(stateless_http=True)
