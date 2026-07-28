"""Logging setup shared by the CLI, the API server and the desktop shell."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED = False

#: Third-party loggers that are chatty at DEBUG and never useful to a user.
_NOISY_LOGGERS = (
    "watchdog",
    "watchdog.observers",
    "httpx",
    "httpcore",
    "asyncio",
    "multipart",
)


def configure_logging(
    level: str = "INFO",
    *,
    log_file: Path | None = None,
    rich_console: bool = True,
    force: bool = False,
) -> None:
    """Install handlers on the root logger.

    Idempotent: repeated calls are ignored unless ``force`` is set, so importing a
    module that configures logging cannot clobber an already-running server's handlers.

    Args:
        level: Root log level name, e.g. ``"INFO"``.
        log_file: Optional path for a size-rotating file handler.
        rich_console: Use Rich's handler for colourised console output.
        force: Reconfigure even if logging was already set up.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    if rich_console:
        console_handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            omit_repeated_times=False,
            markup=False,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
    root.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=8 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s [%(threadName)s]: %(message)s"
            )
        )
        root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(logging.INFO, root.level))

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    Prefer this over :func:`logging.getLogger` so every module ends up under the
    ``ai_asset_manager`` namespace and can be silenced as a unit by embedders.
    """
    if not name.startswith("ai_asset_manager"):
        name = f"ai_asset_manager.{name}"
    return logging.getLogger(name)
