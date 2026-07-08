"""Tests for the Session Registry structured-handoff artifacts (WP-D4).

Per ``docs/serving.md`` §Session Registry
(extended in Cycle 4 per ADR-013) and ``docs/agentic-serving/scenarios.md``
§"Feature: Session Registry Initializer-then-Resume (ADR-013)".

Eight scenarios + one preservation scenario:

1. Cluster 2 session activates structured-handoff artifact set
2. Cluster 1 session opts out of artifact set
3. Monotonic passes constraint enforced at schema level
4. Append-only progress log rejects non-append writes
5. init.sh hash mismatch produces typed error
6. Operator hash rotation re-authors integrity record
7. Cross-cluster session defaults to required artifact set
8. (preservation) Existing Session identification responsibility unchanged
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llm_orc.core.session.artifacts import (
    FeatureEntry,
    FeatureListStore,
    InitScriptGate,
    ProgressLog,
    WriteGateRejectionError,
)
from llm_orc.core.session.registry import (
    SessionRegistry,
    requires_structured_handoff_artifacts,
    resolve_cluster,
)
from llm_orc.models.structural_errors import LlmOrcStructuralError


class TestWriteGateRejectionErrorShape:
    """``WriteGateRejectionError`` is the typed-error producer for ADR-013.

    Third concrete subclass of ``LlmOrcStructuralError`` per FC-17 (after
    ``ToolCallingNotSupportedError`` and ``PhantomToolCallError``). The
    discriminator ``error_kind="write_gate_rejection"`` is fixed by
    construction. The three validation classes are sub-discriminated via
    ``dispatch_context["validation_class"]``.
    """

    def test_subclasses_llm_orc_structural_error(self) -> None:
        error = WriteGateRejectionError(
            "rejected",
            validation_class="feature_list_schema",
            recovery_action_required="reformulate",
        )
        assert isinstance(error, LlmOrcStructuralError)

    def test_error_kind_is_write_gate_rejection(self) -> None:
        error = WriteGateRejectionError(
            "rejected",
            validation_class="feature_list_schema",
            recovery_action_required="reformulate",
        )
        assert error.error_kind == "write_gate_rejection"

    def test_validation_class_lands_in_dispatch_context(self) -> None:
        error = WriteGateRejectionError(
            "rejected",
            validation_class="progress_log_append_only",
            recovery_action_required="reformulate",
        )
        assert error.dispatch_context["validation_class"] == (
            "progress_log_append_only"
        )

    def test_recovery_action_required_is_reformulate_for_schema_violations(
        self,
    ) -> None:
        error = WriteGateRejectionError(
            "rejected",
            validation_class="feature_list_schema",
            recovery_action_required="reformulate",
        )
        assert error.recovery_action_required == "reformulate"

    def test_recovery_action_can_be_operator_intervention_required(self) -> None:
        """init.sh hash mismatch is operator-recoverable, not orchestrator-recoverable.

        Per WP-D4 entry settled premise #1 (cycle-status.md) and the
        precedent that ADR-012 Layer 4 circuit-breaker uses the same
        disposition.
        """
        error = WriteGateRejectionError(
            "rejected",
            validation_class="init_sh_integrity",
            recovery_action_required="operator_intervention_required",
        )
        assert error.recovery_action_required == "operator_intervention_required"

    def test_extra_dispatch_context_is_preserved(self) -> None:
        error = WriteGateRejectionError(
            "rejected",
            validation_class="feature_list_schema",
            recovery_action_required="reformulate",
            dispatch_context={"feature_id": "auth-flow", "session_id": "sess-1"},
        )
        assert error.dispatch_context["feature_id"] == "auth-flow"
        assert error.dispatch_context["session_id"] == "sess-1"
        assert error.dispatch_context["validation_class"] == ("feature_list_schema")


class TestFeatureListMonotonicPasses:
    """Scenario 3: monotonic passes constraint enforced at schema level.

    Given a feature_list.json entry with ``passes: true`` for feature
    ``auth-flow``; when the orchestrator submits a write that would set
    ``auth-flow`` to ``passes: false`` without an audit-logged operator
    override; then the write-gate rejects the write with a typed
    ``write_gate_rejection`` error and the feature_list.json on disk is
    unchanged.
    """

    def test_admits_new_feature_entry(self, tmp_path: Path) -> None:
        store = FeatureListStore(tmp_path / "feature_list.json")

        store.submit_write((FeatureEntry(id="auth-flow", passes=False),))

        assert store.read() == (FeatureEntry(id="auth-flow", passes=False),)

    def test_admits_passes_transition_from_false_to_true(self, tmp_path: Path) -> None:
        store = FeatureListStore(tmp_path / "feature_list.json")
        store.submit_write((FeatureEntry(id="auth-flow", passes=False),))

        store.submit_write((FeatureEntry(id="auth-flow", passes=True),))

        assert store.read() == (FeatureEntry(id="auth-flow", passes=True),)

    def test_admits_passes_held_true(self, tmp_path: Path) -> None:
        store = FeatureListStore(tmp_path / "feature_list.json")
        store.submit_write((FeatureEntry(id="auth-flow", passes=True),))

        store.submit_write((FeatureEntry(id="auth-flow", passes=True),))

        assert store.read() == (FeatureEntry(id="auth-flow", passes=True),)

    def test_rejects_passes_regression_without_override(self, tmp_path: Path) -> None:
        store = FeatureListStore(tmp_path / "feature_list.json")
        store.submit_write((FeatureEntry(id="auth-flow", passes=True),))

        with pytest.raises(WriteGateRejectionError) as exc_info:
            store.submit_write((FeatureEntry(id="auth-flow", passes=False),))

        assert exc_info.value.error_kind == "write_gate_rejection"
        assert exc_info.value.dispatch_context["validation_class"] == (
            "feature_list_schema"
        )
        assert exc_info.value.dispatch_context["feature_id"] == "auth-flow"
        assert exc_info.value.recovery_action_required == "reformulate"

    def test_rejected_write_does_not_touch_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "feature_list.json"
        store = FeatureListStore(path)
        store.submit_write((FeatureEntry(id="auth-flow", passes=True),))
        original_bytes = path.read_bytes()

        with pytest.raises(WriteGateRejectionError):
            store.submit_write((FeatureEntry(id="auth-flow", passes=False),))

        assert path.read_bytes() == original_bytes

    def test_admits_passes_regression_with_audit_logged_override(
        self,
        tmp_path: Path,
    ) -> None:
        """The operator override is audit-logged at the application boundary.

        The write-gate's contract is that an override is *carried* on the
        write; whether the application logs the override (audit trail) is
        a Session Registry concern. The override is the structural unlock,
        not the audit step.
        """
        store = FeatureListStore(tmp_path / "feature_list.json")
        store.submit_write((FeatureEntry(id="auth-flow", passes=True),))

        store.submit_write(
            (FeatureEntry(id="auth-flow", passes=False),),
            operator_override_ids=frozenset({"auth-flow"}),
        )

        assert store.read() == (FeatureEntry(id="auth-flow", passes=False),)

    def test_override_only_unlocks_explicitly_named_features(
        self,
        tmp_path: Path,
    ) -> None:
        store = FeatureListStore(tmp_path / "feature_list.json")
        store.submit_write(
            (
                FeatureEntry(id="auth-flow", passes=True),
                FeatureEntry(id="payment", passes=True),
            )
        )

        with pytest.raises(WriteGateRejectionError) as exc_info:
            store.submit_write(
                (
                    FeatureEntry(id="auth-flow", passes=False),
                    FeatureEntry(id="payment", passes=False),
                ),
                operator_override_ids=frozenset({"auth-flow"}),
            )

        assert exc_info.value.dispatch_context["feature_id"] == "payment"

    def test_read_on_missing_file_returns_empty_tuple(self, tmp_path: Path) -> None:
        store = FeatureListStore(tmp_path / "does-not-exist.json")

        assert store.read() == ()

    def test_persisted_format_is_json_with_id_and_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "feature_list.json"
        store = FeatureListStore(path)

        store.submit_write(
            (
                FeatureEntry(id="auth-flow", passes=True),
                FeatureEntry(id="payment", passes=False),
            )
        )

        data = json.loads(path.read_text())
        assert data == [
            {"id": "auth-flow", "passes": True},
            {"id": "payment", "passes": False},
        ]


class TestProgressLogAppendOnly:
    """Scenario 4: append-only progress log rejects non-append writes.

    Given an active Session with an append-only progress log at any
    state; when the orchestrator submits a write that attempts to
    overwrite, truncate, or mid-file edit the log; then the write-gate
    rejects the operation with a typed ``write_gate_rejection`` error
    and the progress log on disk is unchanged.
    """

    def test_append_admits_writes(self, tmp_path: Path) -> None:
        log = ProgressLog(tmp_path / "progress.log")

        log.submit_write(operation="append", text="started work on auth\n")
        log.submit_write(operation="append", text="finished auth\n")

        assert log.read() == "started work on auth\nfinished auth\n"

    def test_overwrite_rejected_with_typed_error(self, tmp_path: Path) -> None:
        log = ProgressLog(tmp_path / "progress.log")
        log.submit_write(operation="append", text="initial entry\n")

        with pytest.raises(WriteGateRejectionError) as exc_info:
            log.submit_write(operation="overwrite", text="replacement\n")

        assert exc_info.value.error_kind == "write_gate_rejection"
        assert exc_info.value.dispatch_context["validation_class"] == (
            "progress_log_append_only"
        )
        assert exc_info.value.dispatch_context["rejected_operation"] == "overwrite"

    def test_truncate_rejected_with_typed_error(self, tmp_path: Path) -> None:
        log = ProgressLog(tmp_path / "progress.log")
        log.submit_write(operation="append", text="initial entry\n")

        with pytest.raises(WriteGateRejectionError) as exc_info:
            log.submit_write(operation="truncate")

        assert exc_info.value.dispatch_context["rejected_operation"] == "truncate"

    def test_mid_file_edit_rejected_with_typed_error(self, tmp_path: Path) -> None:
        log = ProgressLog(tmp_path / "progress.log")
        log.submit_write(operation="append", text="initial entry\n")

        with pytest.raises(WriteGateRejectionError) as exc_info:
            log.submit_write(operation="edit", text="replacement\n")

        assert exc_info.value.dispatch_context["rejected_operation"] == "edit"

    def test_rejected_write_leaves_file_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.log"
        log = ProgressLog(path)
        log.submit_write(operation="append", text="initial entry\n")
        original_bytes = path.read_bytes()

        with pytest.raises(WriteGateRejectionError):
            log.submit_write(operation="overwrite", text="replacement\n")

        assert path.read_bytes() == original_bytes

    def test_recovery_action_is_reformulate(self, tmp_path: Path) -> None:
        log = ProgressLog(tmp_path / "progress.log")

        with pytest.raises(WriteGateRejectionError) as exc_info:
            log.submit_write(operation="overwrite", text="x")

        assert exc_info.value.recovery_action_required == "reformulate"

    def test_read_on_missing_file_returns_empty_string(self, tmp_path: Path) -> None:
        log = ProgressLog(tmp_path / "does-not-exist.log")

        assert log.read() == ""


class TestInitScriptIntegrity:
    """Scenarios 5 + 6: init.sh hash mismatch and operator hash rotation.

    Scenario 5 — given a Session Registry configuration with init.sh
    integrity hash ``H1`` recorded at operator-authoring time; when the
    Session Registry would invoke init.sh whose actual content hashes
    to ``H2 != H1``; then init.sh execution is gated, a typed
    ``write_gate_rejection`` error fires naming the hash mismatch, and
    the Session does not proceed past initialization.

    Scenario 6 — given an operator legitimately modifies init.sh
    content; when the operator runs the hash-rotation workflow recording
    the new hash; then subsequent Sessions execute the modified init.sh
    successfully.
    """

    def test_admits_invocation_when_hash_matches(self, tmp_path: Path) -> None:
        script_path = tmp_path / "init.sh"
        content = b"#!/usr/bin/env bash\necho ready\n"
        script_path.write_bytes(content)
        recorded_hash = hashlib.sha256(content).hexdigest()
        gate = InitScriptGate(script_path, recorded_hash=recorded_hash)

        gate.verify_integrity()  # does not raise

    def test_rejects_when_actual_hash_differs(self, tmp_path: Path) -> None:
        script_path = tmp_path / "init.sh"
        script_path.write_bytes(b"#!/usr/bin/env bash\necho ready\n")
        recorded_hash = hashlib.sha256(b"old contents").hexdigest()
        gate = InitScriptGate(script_path, recorded_hash=recorded_hash)

        with pytest.raises(WriteGateRejectionError) as exc_info:
            gate.verify_integrity()

        assert exc_info.value.error_kind == "write_gate_rejection"
        assert exc_info.value.dispatch_context["validation_class"] == (
            "init_sh_integrity"
        )
        assert exc_info.value.dispatch_context["recorded_hash"] == recorded_hash
        actual_hash = hashlib.sha256(b"#!/usr/bin/env bash\necho ready\n").hexdigest()
        assert exc_info.value.dispatch_context["actual_hash"] == actual_hash

    def test_hash_mismatch_requires_operator_intervention(self, tmp_path: Path) -> None:
        script_path = tmp_path / "init.sh"
        script_path.write_bytes(b"new contents\n")
        gate = InitScriptGate(
            script_path,
            recorded_hash=hashlib.sha256(b"old contents").hexdigest(),
        )

        with pytest.raises(WriteGateRejectionError) as exc_info:
            gate.verify_integrity()

        assert exc_info.value.recovery_action_required == (
            "operator_intervention_required"
        )

    def test_rotate_hash_returns_new_hash_for_current_content(
        self,
        tmp_path: Path,
    ) -> None:
        script_path = tmp_path / "init.sh"
        new_content = b"#!/usr/bin/env bash\nexport PATH=$PATH:/extra\n"
        script_path.write_bytes(new_content)
        gate = InitScriptGate(
            script_path,
            recorded_hash=hashlib.sha256(b"old contents").hexdigest(),
        )

        new_hash = gate.rotate_hash()

        assert new_hash == hashlib.sha256(new_content).hexdigest()

    def test_after_rotation_a_fresh_gate_admits_invocation(
        self,
        tmp_path: Path,
    ) -> None:
        """Scenario 6 — operator rotates hash; subsequent Sessions succeed.

        The rotation workflow returns the new hash; the Session Registry's
        configuration is updated by the operator; a freshly-constructed
        gate with the rotated hash admits invocation.
        """
        script_path = tmp_path / "init.sh"
        new_content = b"#!/usr/bin/env bash\nexport PATH=$PATH:/extra\n"
        script_path.write_bytes(new_content)
        stale_gate = InitScriptGate(
            script_path,
            recorded_hash=hashlib.sha256(b"old contents").hexdigest(),
        )

        new_hash = stale_gate.rotate_hash()
        rotated_gate = InitScriptGate(script_path, recorded_hash=new_hash)
        rotated_gate.verify_integrity()  # does not raise

    def test_rotate_hash_does_not_silently_unlock_stale_gate(
        self,
        tmp_path: Path,
    ) -> None:
        """``rotate_hash`` is a pure computation — the operator must persist
        the rotation by re-creating the gate (or updating recorded config).

        Mutating the gate in place would let a hostile orchestrator
        invoke ``rotate_hash`` and bypass the integrity check; the
        rotation is structurally exposed to the operator's workflow, not
        to the orchestrator's tool surface.
        """
        script_path = tmp_path / "init.sh"
        script_path.write_bytes(b"new contents\n")
        gate = InitScriptGate(
            script_path,
            recorded_hash=hashlib.sha256(b"old contents").hexdigest(),
        )

        gate.rotate_hash()

        with pytest.raises(WriteGateRejectionError):
            gate.verify_integrity()

    def test_missing_script_file_is_a_typed_rejection(self, tmp_path: Path) -> None:
        gate = InitScriptGate(
            tmp_path / "missing.sh",
            recorded_hash="0" * 64,
        )

        with pytest.raises(WriteGateRejectionError) as exc_info:
            gate.verify_integrity()

        assert exc_info.value.dispatch_context["validation_class"] == (
            "init_sh_integrity"
        )
        assert exc_info.value.recovery_action_required == (
            "operator_intervention_required"
        )


class TestClusterDeterminationAtSessionStart:
    """Scenarios 1, 2, 7: cluster determination governs artifact set activation.

    Per ADR-013 §"Cluster-conditional applicability" and §"Cross-cluster
    sessions". Disposition (i) — default to required-artifact-set
    behavior — is the BUILD-time starting point for cross-cluster
    ambiguity.
    """

    def test_explicit_cluster_2_resolves_to_cluster_2(self) -> None:
        assert resolve_cluster("cluster_2") == "cluster_2"

    def test_explicit_cluster_1_resolves_to_cluster_1(self) -> None:
        assert resolve_cluster("cluster_1") == "cluster_1"

    def test_explicit_cluster_3_resolves_to_cluster_3(self) -> None:
        assert resolve_cluster("cluster_3") == "cluster_3"

    def test_none_declaration_defaults_to_cluster_2(self) -> None:
        """Disposition (i) — ambiguous declaration defaults to required.

        Scenario 7: a cross-cluster or absent declaration must produce
        cluster_2 so the artifact set is active for the session.
        """
        assert resolve_cluster(None) == "cluster_2"

    def test_multi_cluster_declaration_defaults_to_cluster_2(self) -> None:
        """Scenario 7: a North-Star-benchmark-style session straddling
        RESEARCH and BUILD names multiple clusters; disposition (i)
        defaults to cluster_2.
        """
        assert resolve_cluster(["cluster_1", "cluster_2"]) == "cluster_2"
        assert resolve_cluster(["cluster_2", "cluster_3"]) == "cluster_2"

    def test_single_element_list_resolves_to_that_cluster(self) -> None:
        assert resolve_cluster(["cluster_1"]) == "cluster_1"
        assert resolve_cluster(["cluster_3"]) == "cluster_3"

    def test_empty_list_defaults_to_cluster_2(self) -> None:
        """An empty declaration is ambiguous; disposition (i) applies."""
        assert resolve_cluster([]) == "cluster_2"

    def test_unrecognized_declaration_defaults_to_cluster_2(self) -> None:
        """A typo or unknown cluster name is treated as ambiguous.

        Disposition (i)'s reading: when in doubt, activate the artifact
        set — the cost is friction in Cluster 1/3 contexts; the benefit
        is no false-negative misclassification of Cluster 2 territory.
        """
        assert resolve_cluster("cluster_42") == "cluster_2"


class TestArtifactSetActivationByCluster:
    """Scenarios 1, 2: cluster determines whether artifact set is required.

    Cluster 2 → required (Scenario 1). Cluster 1 → supported but optional;
    operator opt-in via explicit configuration (Scenario 2). Cluster 3 →
    supported but optional. Scenario 7 funnels into "required" via
    disposition (i).
    """

    def test_cluster_2_requires_artifacts(self) -> None:
        """Scenario 1: Cluster 2 activates the artifact set."""
        assert requires_structured_handoff_artifacts("cluster_2") is True

    def test_cluster_1_does_not_require_artifacts_by_default(self) -> None:
        """Scenario 2: Cluster 1 is supported but not required."""
        assert requires_structured_handoff_artifacts("cluster_1") is False

    def test_cluster_3_does_not_require_artifacts_by_default(self) -> None:
        assert requires_structured_handoff_artifacts("cluster_3") is False

    def test_cluster_1_with_explicit_opt_in_requires_artifacts(self) -> None:
        """Scenario 2: operator can opt-in with explicit configuration."""
        assert requires_structured_handoff_artifacts("cluster_1", opt_in=True) is True

    def test_cluster_3_with_explicit_opt_in_requires_artifacts(self) -> None:
        assert requires_structured_handoff_artifacts("cluster_3", opt_in=True) is True

    def test_cluster_2_opt_in_flag_is_redundant_but_consistent(self) -> None:
        """Cluster 2 is always required — opt_in is redundant, not contradictory."""
        assert requires_structured_handoff_artifacts("cluster_2", opt_in=True) is True
        assert requires_structured_handoff_artifacts("cluster_2", opt_in=False) is True


class TestSessionRegistryClusterIntegration:
    """End-to-end: ``SessionRegistry`` integrates cluster determination.

    Scenarios 1, 2, 7 at the Session-Registry boundary: the registry's
    ``resolve_session_cluster`` method honors the operator's declaration
    and returns whether the artifact set should activate for the session.
    """

    def test_cluster_2_session_activates_artifact_set(self) -> None:
        """Scenario 1: explicit Cluster 2 declaration activates."""
        registry = SessionRegistry()

        cluster = registry.resolve_session_cluster(declaration="cluster_2")

        assert cluster == "cluster_2"
        assert registry.session_requires_artifact_set(cluster) is True

    def test_cluster_1_session_does_not_activate_artifact_set(self) -> None:
        """Scenario 2: Cluster 1 is opt-out by default."""
        registry = SessionRegistry()

        cluster = registry.resolve_session_cluster(declaration="cluster_1")

        assert cluster == "cluster_1"
        assert registry.session_requires_artifact_set(cluster) is False

    def test_cluster_1_with_opt_in_activates_artifact_set(self) -> None:
        """Scenario 2: explicit opt-in flips the activation."""
        registry = SessionRegistry()

        cluster = registry.resolve_session_cluster(declaration="cluster_1")

        assert registry.session_requires_artifact_set(cluster, opt_in=True) is True

    def test_cross_cluster_session_defaults_to_required(self) -> None:
        """Scenario 7: ambiguous declaration → disposition (i) → required."""
        registry = SessionRegistry()

        cluster = registry.resolve_session_cluster(
            declaration=["cluster_1", "cluster_2"]
        )

        assert cluster == "cluster_2"
        assert registry.session_requires_artifact_set(cluster) is True

    def test_absent_declaration_defaults_to_required(self) -> None:
        """Scenario 7 (no declaration): disposition (i) applies."""
        registry = SessionRegistry()

        cluster = registry.resolve_session_cluster(declaration=None)

        assert cluster == "cluster_2"
        assert registry.session_requires_artifact_set(cluster) is True
