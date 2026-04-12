"""
server/app.py
─────────────
OpenEnv multi-mode deployment entry point.

The OpenEnv validator requires [project.scripts] to point to a callable:
    server = "server.app:main"

The `main()` function below starts uvicorn with the LogEnv FastAPI app.
All business logic lives in the root app.py — nothing is duplicated here.
"""

import os
import sys

# Ensure the project root is importable regardless of working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app import app  # noqa: F401 — re-exported so `server.app:app` also resolves

__all__ = ["app", "main"]


def main() -> None:
    """
    CLI entry point — called by the `server` script installed via pip.

    This is the function the OpenEnv validator expects at:
        [project.scripts]
        server = "server.app:main"

    Usage:
        server                  # starts on port 7860 (default)
        PORT=8080 server        # custom port
    """
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()