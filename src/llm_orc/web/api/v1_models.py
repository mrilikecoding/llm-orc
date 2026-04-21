"""Serving Layer ``/v1/models`` endpoint.

OpenAI-compatible listing of the orchestrator Model Profile IDs an
operator has exposed (per ``docs/agentic-serving/system-design.md``
§Serving Layer; roadmap WP-B Group 3). The body shape mirrors
``https://api.openai.com/v1/models`` so agentic coding tools can
populate their model picker from this endpoint without bespoke client
code.

Filtering follows ``OrchestratorConfigResolver.list_allowed_model_profile_ids``:
the operator-configured allowlist intersected with the Model Profile
library. Absent profiles silently drop out of the list; session start
is where missing-profile errors surface
(``OrchestratorConfigResolver.resolve_validated``).
"""

from typing import Any

from fastapi import APIRouter

from llm_orc.agentic.orchestrator_config import OrchestratorConfigResolver
from llm_orc.core.config.config_manager import ConfigurationManager

router = APIRouter(prefix="/v1", tags=["openai-compat"])

_MODEL_OWNER = "llm-orc"


def get_orchestrator_config_resolver() -> OrchestratorConfigResolver:
    """Return an OrchestratorConfigResolver for the current request.

    Matches the existing router pattern (see ``ensembles.py``
    ``get_orchestra_service``). Per-request construction keeps the
    endpoint stateless and picks up ``config.yaml`` changes without
    process restart. Tests monkeypatch this function to inject scoped
    resolvers.
    """
    return OrchestratorConfigResolver(ConfigurationManager())


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """Return the OpenAI-compatible model list for the orchestrator."""
    resolver = get_orchestrator_config_resolver()
    ids = resolver.list_allowed_model_profile_ids()
    return {
        "object": "list",
        "data": [
            {
                "id": profile_id,
                "object": "model",
                "created": 0,
                "owned_by": _MODEL_OWNER,
            }
            for profile_id in ids
        ],
    }
