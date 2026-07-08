"""Relocation shim — moved to :mod:`llm_orc.web.serving.session_start`.

Cycle-8 WP-B8: the surviving serving-layer session-start contract
(``SessionContext``, ``PromptFragment``, ``SessionStartCache``, the
ADR-009 Phase-2 resolver reservation) relocated out of the ``agentic/``
deletion target. Deleted with the package at WP-F8.
"""

from llm_orc.web.serving.session_start import (
    ChatMessage as ChatMessage,
)
from llm_orc.web.serving.session_start import (
    PromptFragment as PromptFragment,
)
from llm_orc.web.serving.session_start import (
    SessionContext as SessionContext,
)
from llm_orc.web.serving.session_start import (
    SessionStartCache as SessionStartCache,
)
from llm_orc.web.serving.session_start import (
    resolve_session_start_context as resolve_session_start_context,
)
