#!/bin/sh
# One-shot verification gate used by the compose "verify" service.
# The script's exit code is the evidence: any failing stage aborts (set -e)
# and the container exits non-zero.
set -eu

if command -v python >/dev/null 2>&1; then
    PY=python
else
    PY=python3
fi

if [ -d /app ]; then
    cd /app
else
    cd "$(dirname "$0")/.."
fi

echo "--- [1/3] build check: byte-compile all sources ---"
$PY -m compileall -q app tests scripts
echo "compile OK"

echo "--- [2/3] unit/integration tests ---"
$PY -m unittest discover -v -s tests -p 'test_*.py'

echo "--- [3/3] lifecycle proof + HTTP smoke ---"
# Without LEDGER_BASE_URL this runs local subprocess scenarios for torn-tail
# recovery, corruption isolation and HTTP concurrent preemption.
# With LEDGER_BASE_URL (compose verify service) it additionally performs an
# HTTP smoke against the running ledger container.
$PY tests/e2e_smoke.py

if [ -n "${LEDGER_BASE_URL:-}" ]; then
    $PY scripts/http_smoke.py
fi

echo
echo "VERIFY PASSED"
