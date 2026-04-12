"""
server/app.py
─────────────
OpenEnv multi-mode deployment entry point.

The OpenEnv validator requires:
    [project.scripts]
    server = "server.app:app"

This module satisfies that requirement by re-exporting the canonical
FastAPI `app` object from the root app.py.

All business logic lives in root app.py — do NOT duplicate it here.
"""

import sys
import os

# Ensure the project root is on sys.path so `from app import app` works
# regardless of how this module is imported (script, uvicorn, pytest, etc.)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app import app  # noqa: E402  re-exported for OpenEnv entry point

__all__ = ["app"]


def main() -> None:
    """
    CLI entry point — called when running `server` from the command line
    after `pip install -e .` or `uv sync`.

    Example:
        server              # default port 7860
        PORT=8080 server    # custom port
    """
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()