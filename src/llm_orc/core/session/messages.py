"""``ChatMessage`` value type shared across the session substrate and serving layer.

Relocated from ``llm_orc.agentic.session_start`` at Cycle-8 WP-B8. The
original placement kept the dissolved Orchestrator Runtime's FC-4 import
surface clean; post-collapse the type lives with the substrate both its
consumers (Session Registry identity derivation; the serving-layer
``SessionContext`` contract) sit above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChatMessage:
    """An OpenAI-compatible chat message flowing through the Serving Layer.

    Tool-round-trip fields (``tool_call_id``, ``tool_calls``) are populated
    when the client echoes back the prior turn's delegations under
    Option C (Client Tool Surface Commitment) — ``role: assistant``
    messages carry ``tool_calls`` and ``role: tool`` messages carry
    ``tool_call_id``. ``content`` is nullable to match OpenAI's shape for
    an assistant message whose turn carried only tool calls.
    """

    role: str
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = field(default_factory=tuple)
