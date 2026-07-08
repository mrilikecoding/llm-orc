"""Serving Layer ``/v1/models`` endpoint.

OpenAI-compatible listing of the model-profile IDs an operator has
exposed (per ``docs/serving.md`` §Serving Layer).
The body shape mirrors ``https://api.openai.com/v1/models`` so agentic
coding tools can populate their model picker from this endpoint without
bespoke client code.

Filtering follows ``ModelProfileAllowlist.list_allowed_model_profile_ids``:
the operator-configured allowlist intersected with the Model Profile
library. Absent profiles silently drop out of the list.
"""

from typing import Any

from fastapi import APIRouter

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.model_profile_allowlist import ModelProfileAllowlist

router = APIRouter(prefix="/v1", tags=["openai-compat"])

_MODEL_OWNER = "llm-orc"


def get_model_profile_allowlist() -> ModelProfileAllowlist:
    """Return a ModelProfileAllowlist for the current request.

    Matches the existing router pattern (see ``ensembles.py``
    ``get_orchestra_service``). Per-request construction keeps the
    endpoint stateless and picks up ``config.yaml`` changes without
    process restart. Tests monkeypatch this function to inject scoped
    allowlists.
    """
    return ModelProfileAllowlist(ConfigurationManager())


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """Return the OpenAI-compatible model list for the serving layer."""
    allowlist = get_model_profile_allowlist()
    ids = allowlist.list_allowed_model_profile_ids()
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
