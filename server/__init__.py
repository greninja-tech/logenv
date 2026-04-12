# server package — required for OpenEnv multi-mode deployment
# Exposes the FastAPI app via the [project.scripts] server entry point
from .app import app

__all__ = ["app"]