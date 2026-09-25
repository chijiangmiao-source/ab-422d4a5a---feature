"""HTTP front-end for the dose ledger (Python standard library only).

Endpoints
---------
POST /api/batches
    Body: ``{"expected_seq": <next seq the caller saw>, "records": [int, ...]}``
    201 committed batch / 409 optimistic-concurrency loser (nothing written)
    / 400 malformed / 503 ledger poisoned.

POST /api/seals
    Body: ``{"expected_seq": <next seq the caller saw>}``
    Freezes the current confirmed prefix into a citable seal (canonical
    Merkle root, record count, seal id). Returns only after the seal frame
    is durably persisted. 409 on a stale sequence, 400 when there is nothing
    to seal, 503 when poisoned.

GET /api/proofs?seal_id=<hex>&seq=<n>
    Returns the leaf value, its digest, the sibling path and the left/right
    directions for record ``seq`` under ``seal_id``. An independent fold of
    the returned data must reproduce ``root``. 404 for an unknown seal id or
    a record outside the sealed prefix, 503 when poisoned.

GET /api/records?cursor=<seq>&limit=<n>
    Returns records with serial numbers strictly greater than ``cursor``,
    oldest first, plus ``next_cursor`` for the following page. 503 when
    poisoned.

GET /healthz
    200 healthy (possibly after a torn-tail truncation), 503 poisoned.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

from .ledger import Ledger, MAX_LIMIT, StaleSequence
from .merkle import Proof, encode_sibling_path
from .wal import PoisonedError, WAL

MAX_BODY_BYTES = 256 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "DoseLedger/1.0"
    # Every response carries Content-Length, so keep-alive is safe with the
    # threaded server (and makes Docker health checks cheaper).
    protocol_version = "HTTP/1.1"

    # Injected onto the server instance:
    ledger: Ledger

    # -- helpers ----------------------------------------------------------
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _read_json_body(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        length_hdr = self.headers.get("Content-Length")
        if length_hdr is None:
            return None, "missing Content-Length"
        try:
            length = int(length_hdr)
        except ValueError:
            return None, "invalid Content-Length"
        if length < 0 or length > MAX_BODY_BYTES:
            return None, "request body too large"
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "body must be a JSON object"
        if not isinstance(value, dict):
            return None, "body must be a JSON object"
        return value, None

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # Keep stderr concise; access logs go to stderr for docker logs.
        import sys

        sys.stderr.write(
            "%s - - %s\n" % (self.address_string(), fmt % args)
        )

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parts = urlsplit(self.path)
        if parts.path == "/healthz":
            self._handle_health()
        elif parts.path == "/api/records":
            self._handle_records(parts.query)
        elif parts.path == "/api/proofs":
            self._handle_proofs(parts.query)
        elif parts.path == "/":
            self._send_json(
                200,
                {"service": "dose-ledger", "endpoints": [
                    "POST /api/batches", "POST /api/seals",
                    "GET /api/proofs", "GET /api/records",
                    "GET /healthz"]},
            )
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        if parts.path == "/api/batches":
            self._handle_batch_post()
        elif parts.path == "/api/seals":
            self._handle_seal_post()
        else:
            self._send_json(404, {"error": "not_found"})

    def _handle_batch_post(self) -> None:
        body, err = self._read_json_body()
        if err is not None:
            self._send_json(400, {"error": "bad_request", "detail": err})
            return
        assert body is not None
        expected = body.get("expected_seq")
        records = body.get("records")
        if (
            isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected < 1
        ):
            self._send_json(
                400,
                {"error": "bad_request",
                 "detail": "expected_seq must be a positive integer"},
            )
            return
        if not isinstance(records, list):
            self._send_json(
                400,
                {"error": "bad_request",
                 "detail": "records must be a list of 1-32 integers"},
            )
            return
        ledger: Ledger = self.server.ledger  # type: ignore[attr-defined]
        try:
            frame = ledger.submit(expected, records)
        except StaleSequence as exc:
            self._send_json(
                409,
                {"error": "stale_sequence",
                 "detail": str(exc),
                 "current_seq": ledger.head()},
            )
            return
        except ValueError as exc:
            self._send_json(
                400, {"error": "bad_request", "detail": str(exc)}
            )
            return
        except PoisonedError as exc:
            self._send_json(
                503, {"error": "ledger_poisoned", "detail": str(exc)}
            )
            return
        self._send_json(
            201,
            {"status": "committed",
             "seq": frame.seq,
             "count": len(frame.records),
             "next_seq": frame.seq + len(frame.records)},
        )

    def _handle_seal_post(self) -> None:
        body, err = self._read_json_body()
        if err is not None:
            self._send_json(400, {"error": "bad_request", "detail": err})
            return
        assert body is not None
        expected = body.get("expected_seq")
        if (
            isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected < 1
        ):
            self._send_json(
                400,
                {"error": "bad_request",
                 "detail": "expected_seq must be a positive integer"},
            )
            return
        ledger: Ledger = self.server.ledger  # type: ignore[attr-defined]
        try:
            seal = ledger.seal(expected)
        except StaleSequence as exc:
            self._send_json(
                409,
                {"error": "stale_sequence",
                 "detail": str(exc),
                 "current_seq": ledger.head()},
            )
            return
        except ValueError as exc:
            self._send_json(
                400, {"error": "bad_request", "detail": str(exc)}
            )
            return
        except PoisonedError as exc:
            self._send_json(
                503, {"error": "ledger_poisoned", "detail": str(exc)}
            )
            return
        self._send_json(
            201,
            {"status": "sealed",
             "seal_id": seal.seal_id,
             "count": seal.count,
             "root": seal.root.hex(),
             "seq": seal.seq},
        )

    # -- handlers ---------------------------------------------------------
    def _handle_health(self) -> None:
        ledger: Ledger = self.server.ledger  # type: ignore[attr-defined]
        if ledger.poisoned:
            self._send_json(
                503,
                {"status": "poisoned", "reason": ledger.poison_reason},
            )
        else:
            self._send_json(200, {"status": "ok", "next_seq": ledger.head()})

    def _handle_records(self, query: str) -> None:
        ledger: Ledger = self.server.ledger  # type: ignore[attr-defined]
        params = parse_qs(query)
        try:
            cursor = int(params.get("cursor", ["0"])[0])
            limit = int(params.get("limit", [str(MAX_LIMIT)])[0])
        except ValueError:
            self._send_json(
                400, {"error": "bad_request",
                      "detail": "cursor and limit must be integers"}
            )
            return
        try:
            page = ledger.read(cursor, limit)
        except ValueError as exc:
            self._send_json(
                400, {"error": "bad_request", "detail": str(exc)}
            )
            return
        except PoisonedError as exc:
            self._send_json(
                503, {"error": "ledger_poisoned", "detail": str(exc)}
            )
            return
        self._send_json(
            200,
            {"records": [{"seq": r.seq, "dose": r.dose} for r in page.records],
             "next_cursor": page.next_cursor},
        )

    def _handle_proofs(self, query: str) -> None:
        ledger: Ledger = self.server.ledger  # type: ignore[attr-defined]
        params = parse_qs(query)
        seal_id = params.get("seal_id", [""])[0]
        seq_raw = params.get("seq", [""])[0]
        try:
            seq = int(seq_raw)
        except ValueError:
            self._send_json(
                400, {"error": "bad_request",
                      "detail": "seq must be an integer"}
            )
            return
        if not seal_id or seq < 1:
            self._send_json(
                400,
                {"error": "bad_request",
                 "detail": "seal_id is required and seq must be positive"},
            )
            return
        try:
            record = ledger.sealed_record(seal_id, seq)
            proof: Proof = ledger.proof(seal_id, seq)
        except KeyError:
            self._send_json(
                404, {"error": "not_found", "detail": "unknown seal_id"}
            )
            return
        except LookupError as exc:
            self._send_json(
                404, {"error": "not_found", "detail": str(exc)}
            )
            return
        except ValueError as exc:
            self._send_json(
                400, {"error": "bad_request", "detail": str(exc)}
            )
            return
        except PoisonedError as exc:
            self._send_json(
                503, {"error": "ledger_poisoned", "detail": str(exc)}
            )
            return
        directions, siblings = encode_sibling_path(proof.steps)
        # Sibling digests as a fixed-order list of lowercase hex strings;
        # directions is one char per bottom-up step, naming the sibling's
        # side ("L" = sibling is the left child, "R" = right child). Both
        # orders are fixed, never derived from a dict/memory layout.
        self._send_json(
            200,
            {"seal_id": seal_id,
             "seq": record.seq,
             "dose": record.dose,
             "leaf": proof.leaf.hex(),
             "count": proof.count,
             "root": proof.root.hex(),
             "directions": directions,
             "siblings": [
                 siblings[i : i + 32].hex()
                 for i in range(0, len(siblings), 32)
             ]},
        )


def build_server(host: str, port: int, wal_path: str) -> ThreadingHTTPServer:
    wal = WAL(wal_path)
    ledger = Ledger(wal)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.ledger = ledger  # type: ignore[attr-defined]
    httpd.daemon_threads = True
    return httpd


def main() -> None:
    host = os.environ.get("LEDGER_HOST", "0.0.0.0")
    port = int(os.environ.get("LEDGER_PORT", "8080"))
    wal_path = os.environ.get("LEDGER_WAL_PATH", "/data/wal.bin")
    os.makedirs(os.path.dirname(wal_path) or ".", exist_ok=True)
    httpd = build_server(host, port, wal_path)
    ledger: Ledger = httpd.ledger  # type: ignore[attr-defined]
    if ledger.poisoned:
        # Still start serving so health checks report poisoned (503), but do
        # not accept appends.
        print(f"ledger is POISONED: {ledger.poison_reason}", flush=True)
    else:
        wal = ledger._wal
        print(
            f"ledger recovered: next_seq={ledger.head()} "
            f"torn_tail_bytes_removed={wal.truncated_bytes_on_boot}",
            flush=True,
        )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        ledger._wal.close()


if __name__ == "__main__":
    main()
