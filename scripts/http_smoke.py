#!/usr/bin/env python3
"""HTTP smoke test against a running ledger instance (LEDGER_BASE_URL).

Posts one small batch at the currently-advertised next sequence and reads it
back via the cursor API; then seals the current prefix and independently
folds a membership proof back to the sealed root.  Retries briefly on 409
so parallel smoke runs do not fail spuriously.  Exits non-zero on any
violation.
"""

from __future__ import annotations

import hashlib
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

    # Seal the whole prefix at the new head and independently verify a
    # membership proof for the first record just committed.
    head = next_seq + len(records)
    status, seal = call("POST", "/api/seals", {"expected_seq": head})
    assert status == 201, f"seal: {status} {seal}"
    assert seal["count"] == head - 1 and seal["next_seq"] == head, seal
    print(f"smoke: sealed prefix {seal['seal_id']} count={seal['count']} "
          f"root={seal['root'][:16]}...")

    status, proof = call("GET", f"/api/seals/{seal['seal_id']}/proof"
                                f"?seq={next_seq}")
    assert status == 200, proof
    status, proof_again = call("GET", f"/api/seals/{seal['seal_id']}/proof"
                                      f"?seq={next_seq}")
    assert proof == proof_again, "repeated proof query must be identical"

    # Independent recomputation from the wire fields only.
    current = hashlib.sha256(
        b"\x00"
        + json.dumps([proof["seq"], proof["dose"]],
                     separators=(",", ":")).encode("utf-8")
    ).digest()
    for sibling_hex, side in zip(proof["path"], proof["direction"]):
        sibling = bytes.fromhex(sibling_hex)
        assert side in ("left", "right"), side
        if side == "left":
            current = hashlib.sha256(b"\x01" + current + sibling).digest()
        else:
            current = hashlib.sha256(b"\x01" + sibling + current).digest()
    assert current.hex() == seal["root"], (
        f"proof folds to {current.hex()}, sealed root {seal['root']}")
    assert proof["root"] == seal["root"], proof
    print(f"smoke: proof for seq {next_seq} independently folds to root")
    print("HTTP SMOKE OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"HTTP SMOKE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
