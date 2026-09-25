#!/usr/bin/env python3
"""Container health check: exits 0 only when the ledger is not poisoned."""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

port = os.environ.get("LEDGER_PORT", "8080")
url = f"http://127.0.0.1:{port}/healthz"
try:
    with urllib.request.urlopen(url, timeout=3) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except (urllib.error.URLError, urllib.error.HTTPError, OSError):
    # HTTP 503 (poisoned) or connection refused -> unhealthy.
    sys.exit(1)
