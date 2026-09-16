"""API module for llm-orc web server."""

from llm_orc.services.orchestra_service import OrchestraService

_orchestra_service: OrchestraService | None = None


def get_orchestra_service() -> OrchestraService:
    """Get or create the shared OrchestraService instance."""
    global _orchestra_service
    if _orchestra_service is None:
        _orchestra_service = OrchestraService()
    return _orchestra_service
