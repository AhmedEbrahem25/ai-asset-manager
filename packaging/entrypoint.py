"""Entry point for the frozen ``aam`` executable.

Separate from :mod:`ai_asset_manager.cli` so that everything specific to running as a
bundled binary lives in one place and the CLI module stays a plain importable module.

The scanner uses a thread pool rather than processes, so ``freeze_support`` is belt and
braces — but it costs nothing and a frozen program that spawns processes without it
re-runs its own entry point in each child, which presents as the CLI mysteriously
restarting.
"""

from __future__ import annotations

import multiprocessing
import sys


def run() -> None:
    """Start the CLI."""
    multiprocessing.freeze_support()

    from ai_asset_manager.cli import main

    main()


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        # A frozen binary has no traceback the user can act on and no console left open
        # when launched from Explorer. Say what happened in one line and exit non-zero.
        print(f"aam: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
