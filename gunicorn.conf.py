#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Gunicorn configuration for the production server."""

import os

# Bind on the container network
bind = "0.0.0.0:8000"

# ASGI worker so gunicorn can serve the FastMCP app.
worker_class = "uvicorn_worker.UvicornWorker"

# Each worker loads its own embedding model (~1.3 GB), so keep the count modest.
workers = int(os.getenv("WEB_CONCURRENCY", "2"))

# Allow long-lived streaming responses to complete.
timeout = 120
graceful_timeout = 30
keepalive = 5

# Log to stdout/stderr for Docker.
accesslog = "-"
errorlog = "-"
