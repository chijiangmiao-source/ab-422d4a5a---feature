#!/usr/bin/env python3
"""HTTP smoke test against a running ledger instance (LEDGER_BASE_URL).

Posts one small batch at the currently-advertised next sequence and reads it
back via the cursor API.  Retries briefly on 409 so parallel smoke runs do
not fail spuriously.  Exits non-zero on any violation.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode

BASE = os.environ.get("LEDGER_BASE_URL", "http://127.0.0.1:8080")


def call(method: str, path: str, body: object = None) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main() -> int:
    status, health = call("GET", "/healthz")
    assert status == 200, f"healthz: {status} {health}"
    next_seq = health["next_seq"]
    print(f"smoke: service healthy at {BASE}, next_seq={next_seq}")

    records = [int(time.time()) % 100_000, -17, 42]
    for attempt in range(10):
        status, body = call("POST", "/api/batches", {
            "expected_seq": next_seq, "records": records})
        if status == 201:
            break
        assert status == 409, f"unexpected POST status {status}: {body}"
        next_seq = body["current_seq"]
        time.sleep(0.2)
    else:  # pragma: no cover
        raise AssertionError("could not commit smoke batch after retries")

    assert body["seq"] == next_seq, body
    assert body["next_seq"] == next_seq + len(records), body
    print(f"smoke: committed batch seq={next_seq} -> {next_seq + len(records)}")

    query = urlencode({"cursor": next_seq - 1})
    status, page = call("GET", f"/api/records?{query}")
    assert status == 200, page
    got = [(r["seq"], r["dose"]) for r in page["records"]]
    assert got == [(next_seq + i, d) for i, d in enumerate(records)], got
    print(f"smoke: read back {got}")
    print("HTTP SMOKE OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"HTTP SMOKE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
