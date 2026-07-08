"""Session Registry structured-handoff artifacts (ADR-013, WP-D4).

Per ``docs/serving.md`` §Session Registry
(extended in Cycle 4). Owns the three artifact components plus the
write-gate validation surface:

* :class:`FeatureListStore` — feature-list-with-monotonic-passes (JSON
  schema with monotonic ``passes`` field; structural non-regression at
  schema level).
* :class:`ProgressLog` — append-only progress log (free-text narrative
  continuity; filesystem-level append-only constraint).
* :class:`InitScriptGate` — init-sh-style deterministic environment
  bootstrap (operator-authored shell script; hash-recorded at
  authoring time; signed-script tamper-detection scope per ADR-013
  argument-audit P3.2).

All three classes raise :class:`WriteGateRejectionError` —
``error_kind="write_gate_rejection"`` per FC-17 and the Cycle 4 error
pathway table — with the three validation classes sub-discriminated via
``dispatch_context["validation_class"]``. Schema and append-only
violations are orchestrator-recoverable (``reformulate``); init.sh
integrity violations require operator intervention (hash-rotation
workflow).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

from llm_orc.models.structural_errors import LlmOrcStructuralError, RecoveryAction

ValidationClass = Literal[
    "feature_list_schema",
    "progress_log_append_only",
    "init_sh_integrity",
]


class WriteGateRejectionError(LlmOrcStructuralError):
    """Raised when a structured-handoff artifact write violates the write-gate.

    Third concrete subclass of :class:`LlmOrcStructuralError` per FC-17
    (after ``ToolCallingNotSupportedError`` and ``PhantomToolCallError``).
    The discriminator ``error_kind="write_gate_rejection"`` is fixed by
    construction; the three validation classes are sub-discriminated via
    ``dispatch_context["validation_class"]`` per ADR-013 §Decision.

    The ``recovery_action_required`` is supplied per validation class:
    feature-list and progress-log violations are orchestrator-recoverable
    (``reformulate``); init.sh hash mismatch requires operator
    intervention (``operator_intervention_required``) because hash
    rotation is structurally outside the orchestrator's tool surface.
    """

    def __init__(
        self,
        message: str,
        *,
        validation_class: ValidationClass,
        recovery_action_required: RecoveryAction,
        dispatch_context: dict[str, Any] | None = None,
        operator_diagnostic: str | None = None,
    ) -> None:
        merged_context: dict[str, Any] = {"validation_class": validation_class}
        if dispatch_context:
            merged_context.update(dispatch_context)
        super().__init__(
            message,
            error_kind="write_gate_rejection",
            recovery_action_required=recovery_action_required,
            dispatch_context=merged_context,
            operator_diagnostic=operator_diagnostic,
        )


@dataclass(frozen=True)
class FeatureEntry:
    """A single feature-list entry: an identifier plus a monotonic ``passes`` flag.

    Persistence shape follows Anthropic's published ``feature_list.json``
    specification per essay 005 §"Long-Horizon Reliability
    Infrastructure" — a JSON array of objects with ``id`` and ``passes``
    keys. The ``passes`` field is monotonic at the write-gate boundary:
    a ``True → False`` transition is rejected unless an audit-logged
    operator override is supplied per ADR-013 §Decision (i).
    """

    id: str
    passes: bool


class FeatureListStore:
    """Persistent feature list with monotonic-``passes`` write-gate enforcement.

    Per ADR-013 §Decision (i) JSON schema validation: writes that
    transition any feature from ``passes: true`` to ``passes: false``
    without an audit-logged operator override are rejected with
    :class:`WriteGateRejectionError`. The rejection is structural —
    the rejected write does not touch disk (the validation runs
    against in-memory state and the persisted state is mutated only
    after validation passes).

    Persistence format is JSON (per Anthropic's specification):
    ``[{"id": "...", "passes": true}, ...]``.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def read(self) -> tuple[FeatureEntry, ...]:
        if not self._path.exists():
            return ()
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return tuple(
            FeatureEntry(id=entry["id"], passes=entry["passes"]) for entry in raw
        )

    def submit_write(
        self,
        proposed: tuple[FeatureEntry, ...],
        *,
        operator_override_ids: frozenset[str] = frozenset(),
    ) -> None:
        """Validate ``proposed`` against the current state; persist on pass.

        Raises :class:`WriteGateRejectionError` on the first monotonicity
        violation. Disk is untouched until the full proposed set has
        passed validation.
        """
        current = self.read()
        current_by_id = {entry.id: entry for entry in current}
        for entry in proposed:
            prior = current_by_id.get(entry.id)
            if prior is None:
                continue
            if prior.passes and not entry.passes:
                if entry.id in operator_override_ids:
                    continue
                raise WriteGateRejectionError(
                    f"monotonic `passes` constraint violated for "
                    f"feature {entry.id!r}: True → False requires an "
                    f"audit-logged operator override",
                    validation_class="feature_list_schema",
                    recovery_action_required="reformulate",
                    dispatch_context={"feature_id": entry.id},
                )

        payload = [asdict(entry) for entry in proposed]
        self._path.write_text(json.dumps(payload), encoding="utf-8")


_APPEND_ONLY_OPERATION: Final[str] = "append"


class ProgressLog:
    """Append-only progress log with structural rejection of non-append writes.

    Per ADR-013 §Decision (ii) append-only constraint enforcement: the
    write surface admits only ``operation="append"``; overwrite,
    truncate, and mid-file edit operations are rejected with
    :class:`WriteGateRejectionError`. The file on disk is unchanged
    on rejection (rejection happens before any filesystem call).

    Persistence shape is free text following Anthropic's
    ``claude-progress.txt`` convention.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def read(self) -> str:
        if not self._path.exists():
            return ""
        return self._path.read_text(encoding="utf-8")

    def submit_write(self, *, operation: str, text: str = "") -> None:
        """Admit only append; reject every other operation kind structurally."""
        if operation != _APPEND_ONLY_OPERATION:
            raise WriteGateRejectionError(
                f"non-append progress-log operation {operation!r} rejected — "
                f"only {_APPEND_ONLY_OPERATION!r} is admitted",
                validation_class="progress_log_append_only",
                recovery_action_required="reformulate",
                dispatch_context={"rejected_operation": operation},
            )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(text)


class InitScriptGate:
    """Hash-gated execution of the operator-authored init.sh.

    Per ADR-013 §Decision (iii) signed-script integrity verification
    (tamper-detection scope per argument-audit P3.2): execution is
    gated on a hash match between the operator-recorded hash and the
    file's current content hash. Mismatch produces
    :class:`WriteGateRejectionError` with
    ``recovery_action_required="operator_intervention_required"`` —
    hash rotation is the operator workflow, not orchestrator-reachable.

    The integrity check is *tamper-detection*: it detects modification
    of init.sh between operator-authoring time and session-execution
    time. It does not validate that the operator-authored script is
    itself safe; the operator's authoring step is the trust boundary.
    """

    def __init__(self, script_path: Path, *, recorded_hash: str) -> None:
        self._script_path = script_path
        self._recorded_hash = recorded_hash

    def verify_integrity(self) -> None:
        if not self._script_path.exists():
            raise WriteGateRejectionError(
                f"init.sh integrity check failed — script not found at "
                f"{self._script_path}",
                validation_class="init_sh_integrity",
                recovery_action_required="operator_intervention_required",
                dispatch_context={
                    "recorded_hash": self._recorded_hash,
                    "script_path": str(self._script_path),
                },
            )
        actual_hash = self._compute_hash()
        if actual_hash != self._recorded_hash:
            raise WriteGateRejectionError(
                "init.sh integrity check failed — content hash does not "
                "match the operator-recorded hash. Run the hash-rotation "
                "workflow to re-author the recorded hash.",
                validation_class="init_sh_integrity",
                recovery_action_required="operator_intervention_required",
                dispatch_context={
                    "recorded_hash": self._recorded_hash,
                    "actual_hash": actual_hash,
                    "script_path": str(self._script_path),
                },
            )

    def rotate_hash(self) -> str:
        """Return the current content hash for the operator to record.

        The rotation is a pure computation — the operator persists the
        new hash by re-creating the gate (or updating recorded config).
        The operation does NOT mutate ``self._recorded_hash`` to prevent
        an in-process bypass: a hostile orchestrator with access to the
        gate object cannot unlock execution by calling ``rotate_hash``.
        """
        return self._compute_hash()

    def _compute_hash(self) -> str:
        return hashlib.sha256(self._script_path.read_bytes()).hexdigest()
