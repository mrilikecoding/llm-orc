"""Relocation shim — moved to :mod:`llm_orc.core.validation.composition_validator`.

Cycle-8 WP-B8: the surviving AS-2 admission gate (WP-C8 reuses it for
registry/shape-catalog admission) relocated out of the ``agentic/``
deletion target. Deleted with the package at WP-F8.
"""

from llm_orc.core.validation.composition_validator import (
    CompositionAccepted as CompositionAccepted,
)
from llm_orc.core.validation.composition_validator import (
    CompositionOutcome as CompositionOutcome,
)
from llm_orc.core.validation.composition_validator import (
    CompositionRejected as CompositionRejected,
)
from llm_orc.core.validation.composition_validator import (
    CompositionRequest as CompositionRequest,
)
from llm_orc.core.validation.composition_validator import (
    CompositionValidator as CompositionValidator,
)
from llm_orc.core.validation.composition_validator import (
    ConfigManagerEnsembleWriter as ConfigManagerEnsembleWriter,
)
from llm_orc.core.validation.composition_validator import (
    ConfigManagerPrimitiveRegistry as ConfigManagerPrimitiveRegistry,
)
from llm_orc.core.validation.composition_validator import (
    EnsembleWriteError as EnsembleWriteError,
)
from llm_orc.core.validation.composition_validator import (
    LocalEnsembleWriter as LocalEnsembleWriter,
)
from llm_orc.core.validation.composition_validator import (
    PrimitiveRegistry as PrimitiveRegistry,
)
from llm_orc.core.validation.composition_validator import (
    RejectionKind as RejectionKind,
)
