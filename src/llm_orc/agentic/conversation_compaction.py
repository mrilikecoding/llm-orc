"""Relocation shim — compaction moved to :mod:`llm_orc.core.session.compaction`.

Cycle-8 WP-B8: the surviving ADR-012 session-substrate service relocated
out of the ``agentic/`` deletion target. Deleted with the package at WP-F8.
"""

from llm_orc.core.session.compaction import (
    DEFAULT_COMPACTION_IDLE_WINDOW_MINUTES as DEFAULT_COMPACTION_IDLE_WINDOW_MINUTES,
)
from llm_orc.core.session.compaction import (
    DEFAULT_COMPACTION_LAYER_4_CIRCUIT_BREAKER_THRESHOLD as DEFAULT_COMPACTION_LAYER_4_CIRCUIT_BREAKER_THRESHOLD,  # noqa: E501
)
from llm_orc.core.session.compaction import (
    DEFAULT_COMPACTION_PERSIST_THRESHOLD_CHARS as DEFAULT_COMPACTION_PERSIST_THRESHOLD_CHARS,  # noqa: E501
)
from llm_orc.core.session.compaction import (
    DEFAULT_COMPACTION_SESSION_NOTES_TOKEN_CAP as DEFAULT_COMPACTION_SESSION_NOTES_TOKEN_CAP,  # noqa: E501
)
from llm_orc.core.session.compaction import (
    DEFAULT_COMPACTION_SUMMARIZER_ENSEMBLE as DEFAULT_COMPACTION_SUMMARIZER_ENSEMBLE,
)
from llm_orc.core.session.compaction import (
    DEFAULT_COMPACTION_TRIGGER_TOKEN_COUNT as DEFAULT_COMPACTION_TRIGGER_TOKEN_COUNT,
)
from llm_orc.core.session.compaction import (
    NINE_SECTIONS as NINE_SECTIONS,
)
from llm_orc.core.session.compaction import (
    Clock as Clock,
)
from llm_orc.core.session.compaction import (
    CompactedContext as CompactedContext,
)
from llm_orc.core.session.compaction import (
    CompactionDefaults as CompactionDefaults,
)
from llm_orc.core.session.compaction import (
    CompactionLayer4FailureError as CompactionLayer4FailureError,
)
from llm_orc.core.session.compaction import (
    ConversationCompaction as ConversationCompaction,
)
from llm_orc.core.session.compaction import (
    SessionNotes as SessionNotes,
)
from llm_orc.core.session.compaction import (
    Summarizer as Summarizer,
)
from llm_orc.core.session.compaction import (
    SummarizerInvocation as SummarizerInvocation,
)
