"""Durable, credit-safe consumption layer for long-form film generation.

This package intentionally talks to the supply layer only through the frozen HTTP
contracts.  It does not import or inspect the provider/router implementation.
"""

from .clients import ChatPoolClient, VideoRouterClient
from .orchestrator import FilmOrchestrator, FilmRequest
from .store import FilmJobStore

__all__ = ["ChatPoolClient", "VideoRouterClient", "FilmJobStore", "FilmOrchestrator", "FilmRequest"]
