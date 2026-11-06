"""One-JSON-object-per-line logging to stdout, so `docker compose logs` is searchable."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Fields passed through `extra=` (job_id, stage and so on) become top-level keys.
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    try:
        root.setLevel(level.upper())
    except ValueError:
        # A typo in LOG_LEVEL must not stop the service from starting.
        root.setLevel(logging.INFO)
        root.warning("invalid LOG_LEVEL, using INFO", extra={"log_level": level})
    # Route uvicorn's logs through the same formatter. Leave uvicorn.access alone:
    # --no-access-log works by turning its propagation off, and re-enabling it
    # would log every /health and job poll.
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
