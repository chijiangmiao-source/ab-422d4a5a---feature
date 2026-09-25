"""End-to-end tests against a real HTTP server in a background thread."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from app.merkle import INNER_PREFIX, LEAF_PREFIX
from app.server import build_server


def _request(method: str, url: str, body: object = None) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class HttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.wal_path = os.path.join(self.dir, "wal.bin")
        self.httpd: ThreadingHTTPServer = build_server(
            "127.0.0.1", 0, self.wal_path
        )
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def test_health_and_info(self) -> None:
        status, body = _request("GET", f"{self.base}/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok", "next_seq": 1})

    def test_post_batch_then_cursor_read(self) -> None:
        status, body = _request(
            "POST", f"{self.base}/api/batches",
            {"expected_seq": 1, "records": [10, 20, 30]},
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["seq"], 1)
        self.assertEqual(body["next_seq"], 4)

        status, body = _request(
            "GET", f"{self.base}/api/records?cursor=0&limit=2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["records"],
                         [{"seq": 1, "dose": 10}, {"seq": 2, "dose": 20}])
        self.assertEqual(body["next_cursor"], 2)

        status, body = _request(
            "GET", f"{self.base}/api/records?cursor=2"
        )
        self.assertEqual(body["records"], [{"seq": 3, "dose": 30}])
        self.assertEqual(body["next_cursor"], 3)

    def test_conflict_loser_keeps_log_clean(self) -> None:
        s1, _ = _request("POST", f"{self.base}/api/batches",
                         {"expected_seq": 1, "records": [1]})
        self.assertEqual(s1, 201)
        s2, body = _request("POST", f"{self.base}/api/batches",
                            {"expected_seq": 1, "records": [2]})
        self.assertEqual(s2, 409)
        self.assertEqual(body["current_seq"], 2)

        _, body = _request("GET", f"{self.base}/api/records")
        self.assertEqual(body["records"], [{"seq": 1, "dose": 1}])

    def test_bad_requests(self) -> None:
        for payload in (
            {"expected_seq": 1, "records": []},
            {"expected_seq": 1, "records": [1] * 33},
            {"expected_seq": 0, "records": [1]},
            {"expected_seq": 1, "records": "nope"},
        ):
            status, body = _request(
                "POST", f"{self.base}/api/batches", payload
            )
            self.assertEqual(status, 400, payload)
            self.assertEqual(body["error"], "bad_request")

        status, body = _request("GET", f"{self.base}/api/records?cursor=x")
        self.assertEqual(status, 400)

    def test_rebuilds_sequence_after_restart(self) -> None:
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [5, 6]})
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 3, "records": [7]})
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

        self.httpd = build_server("127.0.0.1", 0, self.wal_path)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

        status, health = _request("GET", f"{self.base}/healthz")
        self.assertEqual(health["next_seq"], 4)
        _, body = _request("GET", f"{self.base}/api/records")
        self.assertEqual(
            body["records"],
            [{"seq": 1, "dose": 5}, {"seq": 2, "dose": 6},
             {"seq": 3, "dose": 7}],
        )
        # A stale expected_seq is rejected; correct one succeeds.
        self.assertEqual(
            _request("POST", f"{self.base}/api/batches",
                     {"expected_seq": 3, "records": [8]})[0],
            409,
        )
        self.assertEqual(
            _request("POST", f"{self.base}/api/batches",
                     {"expected_seq": 4, "records": [8]})[0],
            201,
        )

    def test_seal_and_independent_proof_recomputation(self) -> None:
        # Commit two batches (5 records), then seal at the observed head.
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [10, 20, 30]})
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 4, "records": [40, 50]})
        status, seal = _request("POST", f"{self.base}/api/seals",
                                {"expected_seq": 6})
        self.assertEqual(status, 201, seal)
        self.assertEqual(seal["status"], "sealed")
        self.assertEqual(seal["count"], 5)
        self.assertEqual(seal["seq"], 6)
        self.assertEqual(len(seal["root"]), 64)
        seal_id = seal["root"][:32]
        self.assertEqual(seal["seal_id"], seal_id)

        # Fetch a proof for seq 3 and independently fold it back to root.
        status, body = _request(
            "GET", f"{self.base}/api/proofs?seal_id={seal_id}&seq=3")
        self.assertEqual(status, 200, body)
        self.assertEqual(body["seq"], 3)
        self.assertEqual(body["dose"], 30)
        self.assertEqual(body["count"], 5)
        self.assertEqual(body["root"], seal["root"])
        self._assert_folds_to_root(body)
        self._assert_leaf_encoding(body)

        # Repeated queries return byte-identical proofs.
        _, again = _request(
            "GET", f"{self.base}/api/proofs?seal_id={seal_id}&seq=3")
        self.assertEqual(again, body)

        # Every record in the prefix verifies.
        for seq in range(1, 6):
            _, p = _request(
                "GET", f"{self.base}/api/proofs?seal_id={seal_id}&seq={seq}")
            self._assert_folds_to_root(p)

    def _assert_folds_to_root(self, body: dict) -> None:
        """Recompute the root from the JSON proof using only stdlib hashlib."""
        node = bytes.fromhex(body["leaf"])
        self.assertEqual(len(node), 32)
        directions = body["directions"]
        siblings = body["siblings"]
        self.assertEqual(len(directions), len(siblings))
        for ch, sib_hex in zip(directions, siblings):
            sib = bytes.fromhex(sib_hex)
            if ch == "L":
                node = hashlib.sha256(
                    INNER_PREFIX + sib + node).digest()
            elif ch == "R":
                node = hashlib.sha256(
                    INNER_PREFIX + node + sib).digest()
            else:
                self.fail(f"bad direction {ch!r}")
        self.assertEqual(node.hex(), body["root"])

    def _assert_leaf_encoding(self, body: dict) -> None:
        leaf_input = json.dumps(
            {"dose": body["dose"], "seq": body["seq"]},
            separators=(",", ":"), sort_keys=True).encode()
        self.assertEqual(
            body["leaf"],
            hashlib.sha256(LEAF_PREFIX + leaf_input).hexdigest(),
        )

    def test_seal_errors(self) -> None:
        # Empty prefix cannot be sealed.
        status, body = _request("POST", f"{self.base}/api/seals",
                                {"expected_seq": 1})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "bad_request")

        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [1, 2]})
        # Stale seal point: head moved to 3.
        status, body = _request("POST", f"{self.base}/api/seals",
                                {"expected_seq": 2})
        self.assertEqual(status, 409)
        self.assertEqual(body["current_seq"], 3)
        # Malformed payload.
        status, _ = _request("POST", f"{self.base}/api/seals",
                             {"expected_seq": 0})
        self.assertEqual(status, 400)

    def test_proof_errors(self) -> None:
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [1, 2, 3]})
        _, seal = _request("POST", f"{self.base}/api/seals",
                           {"expected_seq": 4})
        seal_id = seal["seal_id"]

        status, body = _request(
            "GET", f"{self.base}/api/proofs?seal_id={seal_id}&seq=9")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

        status, body = _request(
            "GET", f"{self.base}/api/proofs?seal_id={'00' * 16}&seq=1")
        self.assertEqual(status, 404)

        status, _ = _request(
            "GET", f"{self.base}/api/proofs?seal_id={seal_id}&seq=x")
        self.assertEqual(status, 400)
        status, _ = _request(
            "GET", f"{self.base}/api/proofs?seal_id=&seq=1")
        self.assertEqual(status, 400)

    def test_proof_survives_restart_and_older_seal_stays_citable(self) -> None:
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [10, 20]})
        _, first = _request("POST", f"{self.base}/api/seals",
                            {"expected_seq": 3})
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 3, "records": [30]})
        _, second = _request("POST", f"{self.base}/api/seals",
                             {"expected_seq": 4})
        self.restart()

        _, p1 = _request(
            "GET",
            f"{self.base}/api/proofs?seal_id={first['seal_id']}&seq=1")
        self._assert_folds_to_root(p1)
        self.assertEqual(p1["root"], first["root"])
        self.assertEqual(p1["count"], 2)
        _, p2 = _request(
            "GET",
            f"{self.base}/api/proofs?seal_id={second['seal_id']}&seq=3")
        self._assert_folds_to_root(p2)
        # The newer record is not provable under the older seal.
        status, _ = _request(
            "GET",
            f"{self.base}/api/proofs?seal_id={first['seal_id']}&seq=3")
        self.assertEqual(status, 404)

    def restart(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        self.httpd = build_server("127.0.0.1", 0, self.wal_path)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def test_poisoned_log_returns_503_for_everything(self) -> None:
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [1, 2, 3]})
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

        data = bytearray(open(self.wal_path, "rb").read())
        data[24] ^= 0xFF  # flip a payload byte
        with open(self.wal_path, "wb") as fh:
            fh.write(data)

        self.httpd = build_server("127.0.0.1", 0, self.wal_path)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

        self.assertEqual(_request("GET", f"{self.base}/healthz")[0], 503)
        self.assertEqual(
            _request("GET", f"{self.base}/api/records")[0], 503
        )
        self.assertEqual(
            _request("POST", f"{self.base}/api/batches",
                     {"expected_seq": 1, "records": [9]})[0],
            503,
        )


if __name__ == "__main__":
    unittest.main()
