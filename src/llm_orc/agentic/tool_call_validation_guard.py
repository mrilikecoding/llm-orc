"""Relocation shim — moved to :mod:`llm_orc.core.validation.tool_call_guard`.

Cycle-8 WP-B8: the surviving ADR-017 guard-half (deterministic
phantom-tool-call detection) relocated out of the ``agentic/`` deletion
target. Deleted with the package at WP-F8.
"""

from llm_orc.core.validation.tool_call_guard import (
    DEFAULT_ASSERTION_PATTERNS as DEFAULT_ASSERTION_PATTERNS,
)
from llm_orc.core.validation.tool_call_guard import (
    PhantomToolCallError as PhantomToolCallError,
)
from llm_orc.core.validation.tool_call_guard import (
    scan_response_for_phantom_claims as scan_response_for_phantom_claims,
)
