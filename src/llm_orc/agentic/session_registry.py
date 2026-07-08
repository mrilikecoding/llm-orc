"""Relocation shim — Session Registry moved to :mod:`llm_orc.core.session.registry`.

Cycle-8 WP-B8: the surviving ADR-013 substrate relocated out of the
``agentic/`` deletion target. This shim keeps the dissolved loop-driver
chain and its tests importing unchanged; it is deleted with the package
at WP-F8.
"""

from llm_orc.core.session.registry import (
    Cluster as Cluster,
)
from llm_orc.core.session.registry import (
    IdentityMethod as IdentityMethod,
)
from llm_orc.core.session.registry import (
    SessionCloseCallback as SessionCloseCallback,
)
from llm_orc.core.session.registry import (
    SessionIdentity as SessionIdentity,
)
from llm_orc.core.session.registry import (
    SessionRegistry as SessionRegistry,
)
from llm_orc.core.session.registry import (
    SessionState as SessionState,
)
from llm_orc.core.session.registry import (
    requires_structured_handoff_artifacts as requires_structured_handoff_artifacts,
)
from llm_orc.core.session.registry import (
    resolve_cluster as resolve_cluster,
)
