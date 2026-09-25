FROM python:3.11-slim

# Zero third-party dependencies: the service, tests and health check all use
# the Python standard library, so no network access is needed at build time.
WORKDIR /app

COPY app ./app
COPY tests ./tests
COPY scripts ./scripts

RUN mkdir -p /data && chmod +x scripts/verify.sh \
    && python -m compileall -q app tests

ENV LEDGER_HOST=0.0.0.0 \
    LEDGER_PORT=8080 \
    LEDGER_WAL_PATH=/data/wal.bin \
    PYTHONUNBUFFERED=1

EXPOSE 8080
VOLUME ["/data"]

HEALTHCHECK --interval=5s --timeout=3s --start-period=3s --retries=3 \
    CMD ["python", "scripts/healthcheck.py"]

CMD ["python", "-m", "app.server"]
