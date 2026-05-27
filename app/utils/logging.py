"""
CyberGuard AI - Structured Logging Utility

JSON tabanlı yapılandırılmış loglama.
Logları logs/app/ klasörüne yazar ve konsola basar.

Kullanım:
    from app.utils.logging import get_logger, setup_logging

    logger = get_logger(__name__)
    logger.info("İşlem başladı", extra={"user": "admin", "ip": "1.2.3.4"})
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.paths import PROJECT_ROOT

# Logs klasörü
LOGS_DIR = os.path.join(str(PROJECT_ROOT), "logs", "app")
os.makedirs(LOGS_DIR, exist_ok=True)


class JSONFormatter(logging.Formatter):
    """
    Her log kaydını JSON satırı olarak biçimlendirir.
    Kolayca Elasticsearch/Loki'ye gönderilebilir.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Exception bilgisi
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Ek alanlar (extra= ile geçilenleri dahil et)
        for key, value in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            ):
                try:
                    json.dumps(value)   # serileştirilebilir mi?
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    log_file: str = "cyberguard.log",
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
    json_console: bool = False,
) -> None:
    """
    Uygulama genelinde loglama yapılandırması.
    main.py startup'ında bir kez çağrılmalı.

    Args:
        level:        Log seviyesi (DEBUG/INFO/WARNING/ERROR)
        log_file:     Dosya adı (logs/app/ altında açılır)
        max_bytes:    Rotating dosya max boyutu
        backup_count: Rotasyonda saklanacak yedek dosya sayısı
        json_console: True → konsola da JSON yaz; False → okunabilir format
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Mevcut handler'ları kaldır (tekrar çağrılma durumuna karşı)
    root_logger.handlers.clear()

    # --- Rotating File Handler (JSON) ---
    file_path = os.path.join(LOGS_DIR, log_file)
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    if json_console:
        console_handler.setFormatter(JSONFormatter())
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        console_handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root_logger.addHandler(console_handler)

    # Gürültülü kütüphaneleri sustur
    for noisy in ("uvicorn.access", "httpx", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Modüle özgü logger döndürür."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# FastAPI Request-ID middleware yardımcısı
# ---------------------------------------------------------------------------

class RequestIDMiddleware:
    """
    Her HTTP isteğine rastgele bir X-Request-ID atar ve
    bunu yapılandırılmış loglara ekler.

    Kullanım (main.py):
        from app.utils.logging import RequestIDMiddleware
        app.add_middleware(RequestIDMiddleware)
    """

    def __init__(self, app):
        self.app = app
        self._logger = get_logger("request")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = str(uuid.uuid4())[:8]
            start = time.monotonic()

            async def send_with_header(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"x-request-id", request_id.encode()))
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, send_with_header)

            duration_ms = round((time.monotonic() - start) * 1000, 2)
            path = scope.get("path", "")
            method = scope.get("method", "")
            self._logger.info(
                f"{method} {path}",
                extra={"request_id": request_id, "duration_ms": duration_ms},
            )
        else:
            await self.app(scope, receive, send)
