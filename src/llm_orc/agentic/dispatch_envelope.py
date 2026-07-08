"""Relocation shim — moved to :mod:`llm_orc.models.dispatch_envelope`.

Cycle-8 WP-B8: the surviving ADR-024 I/O Envelope contract relocated out
of the ``agentic/`` deletion target, beside ``LlmOrcStructuralError``
(the same cross-cutting shared-type status its own docstring claims).
Deleted with the package at WP-F8.
"""

from llm_orc.models.dispatch_envelope import (
    DispatchEnvelope as DispatchEnvelope,
)
from llm_orc.models.dispatch_envelope import (
    EnvelopeStatus as EnvelopeStatus,
)
