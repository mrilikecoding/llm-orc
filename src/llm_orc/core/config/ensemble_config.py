"""Ensemble configuration loading and management."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_orc.schemas.agent_config import (
    AgentConfig,
    EnsembleAgentConfig,
    parse_agent_config,
)

logger = logging.getLogger(__name__)


def _find_agent_by_name(
    agents: list[AgentConfig], agent_name: str
) -> AgentConfig | None:
    """Find an agent by name in the agents list.

    Args:
        agents: List of agent configurations
        agent_name: Name of agent to find

    Returns:
        Agent configuration if found, None otherwise
    """
    return next((a for a in agents if a.name == agent_name), None)


def _find_cycle_from(
    agent_name: str,
    agents: list[AgentConfig],
    visited: set[str],
    in_stack: set[str],
    path: list[str],
) -> list[str] | None:
    """DFS from agent_name, returning cycle path if found.

    Handles both simple string and conditional dict dependencies.

    Args:
        agent_name: Starting agent name
        agents: List of agent configurations
        visited: Set of already-visited agent names
        in_stack: Set of agents in the current recursion stack
        path: Ordered list of agents in the current DFS path

    Returns:
        List of agent names forming the cycle, or None
    """
    if agent_name in in_stack:
        cycle_start = path.index(agent_name)
        return path[cycle_start:]
    if agent_name in visited:
        return None

    visited.add(agent_name)
    in_stack.add(agent_name)
    path.append(agent_name)

    agent_config = _find_agent_by_name(agents, agent_name)
    if agent_config:
        for dep in agent_config.depends_on:
            dep_name = dep if isinstance(dep, str) else dep.get("agent_name")
            if dep_name:
                cycle = _find_cycle_from(dep_name, agents, visited, in_stack, path)
                if cycle is not None:
                    return cycle

    path.pop()
    in_stack.remove(agent_name)
    return None


def detect_cycle(agents: list[AgentConfig]) -> list[str] | None:
    """Detect cycles in agent dependencies using DFS.

    Handles both string and dict-form dependencies.

    Args:
        agents: List of agent configurations

    Returns:
        List of agent names forming the cycle, or None if acyclic
    """
    visited: set[str] = set()

    for agent in agents:
        if agent.name not in visited:
            cycle = _find_cycle_from(agent.name, agents, visited, set(), [])
            if cycle is not None:
                return cycle
    return None


def _validated_literal(raw: Any, allowed: set[str]) -> str | None:
    """Return ``raw`` if it is one of ``allowed``; otherwise ``None``.

    Used by :meth:`EnsembleLoader.load_from_file` to load ADR-025's
    Cycle 6 WP-E optional YAML fields (``output_substrate``,
    ``output_retention``, ``calibration_substrate_access``) tolerantly:
    values inside the closed set load verbatim; values outside it (or
    of the wrong type) collapse to ``None`` so the dispatch site
    applies its documented default rather than the session failing to
    start on an operator typo. Matches the established loader posture
    on ``output_schema`` (non-dict → ``None`` rather than raise).
    """
    if isinstance(raw, str) and raw in allowed:
        return raw
    return None


def assert_no_cycles(agents: list[AgentConfig]) -> None:
    """Raise ValueError if any cycle exists in agent dependencies.

    Args:
        agents: List of agent configurations

    Raises:
        ValueError: With the cycle path (e.g. "a -> b -> a")
    """
    cycle = detect_cycle(agents)
    if cycle is not None:
        cycle_str = " -> ".join([*cycle, cycle[0]])
        raise ValueError(f"Circular dependency detected: {cycle_str}")


def _validate_fan_out_dependencies(agents: list[AgentConfig]) -> None:
    """Validate that fan_out agents have required dependencies.

    Args:
        agents: List of agent configurations

    Raises:
        ValueError: If any agent has fan_out: true without depends_on
    """
    for agent in agents:
        if agent.fan_out is True:
            if not agent.depends_on:
                raise ValueError(
                    f"Agent '{agent.name}' has fan_out: true but requires "
                    f"depends_on to specify the upstream agent providing "
                    f"the array"
                )


def _check_missing_dependencies(agents: list[AgentConfig]) -> None:
    """Check for missing dependencies in agent configurations.

    Args:
        agents: List of agent configurations

    Raises:
        ValueError: If any agent depends on a non-existent agent
    """
    agent_names = {agent.name for agent in agents}

    for agent in agents:
        for dep in agent.depends_on:
            dep_name = dep if isinstance(dep, str) else dep.get("agent_name")
            if dep_name and dep_name not in agent_names:
                raise ValueError(
                    f"Agent '{agent.name}' has missing dependency: '{dep_name}'"
                )


@dataclass
class EnsembleConfig:
    """Configuration for an ensemble of agents with dependency support."""

    name: str
    description: str
    agents: list[AgentConfig] = field(default_factory=list)
    default_task: str | None = None
    task: str | None = None  # Backward compatibility
    relative_path: str | None = None  # For hierarchical display
    validation: dict[str, Any] | None = None  # Validation configuration
    test_mode: dict[str, Any] | None = None  # Test mode configuration
    raw_output: bool = False
    """ADR-004 escape hatch. When true, the Result Summarizer Harness passes
    invoke_ensemble results through untouched. Reserved for ensembles whose
    output is already small and structured (classifiers, single-field lookups).
    Default is false — summarization is the operator's default contract."""

    topaz_skill: str | None = None
    """ADR-015 primary Topaz skill metadata for tier-escalation routing.

    Operator-authored YAML field naming the ensemble's primary Topaz
    skill — one of: ``code_generation``, ``tool_use``,
    ``mathematical_reasoning``, ``logical_reasoning``,
    ``factual_knowledge``, ``writing_quality``,
    ``instruction_following``, ``summarization``. The Tier-Escalation
    Router (WP-G4-1) reads this field to select the per-skill cheap-
    or escalated-tier Model Profile per the Calibration Gate's
    verdict. ``None`` (the field missing in YAML) causes the router
    to raise :class:`MissingSkillMetadataError` per ADR-015 §Per-
    skill role profiling. Stored as ``str`` rather than the
    :data:`TopazSkill` Literal because EnsembleConfig is loaded from
    operator YAML before validation; the router validates the value
    against the closed Topaz taxonomy at dispatch time.
    """

    output_schema: dict[str, Any] | None = None
    """ADR-024 optional JSON-Schema-shaped description of the typed
    payload the synthesizer agent (or post-dispatch processing) writes
    to ``DispatchEnvelope.structured`` (Cycle 6 WP-D).

    When declared, ``invoke_ensemble`` attempts a JSON-parse of the
    synthesizer's response and, on success, populates
    ``envelope.structured`` with the parsed payload. Schema validation
    is **advisory** at dispatch time per spike β's reframing of
    output-spec drift as ``input.data`` override (the synthesizer is
    not the drift source; enforcement at the synthesizer would catch
    the wrong thing). The schema's value is enabling downstream
    consumers (composing ensembles; the orchestrator's reasoning
    surface; calibration-gate critics under ``structured``-augmented
    evaluation) to parse the structured payload predictably when
    present. ``None`` (the field missing in YAML) means
    ``envelope.structured`` stays ``None``.
    """

    output_substrate: str | None = None
    """ADR-025 optional substrate-routing declaration (Cycle 6 WP-E).

    Accepted values ``"artifact"`` and ``"inline"``. The dispatch path
    reads this field to decide whether the dispatched ensemble's
    deliverable routes through SessionArtifactStore (``artifact`` —
    writes to ``.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/``;
    envelope.primary carries a summary line; envelope.artifacts[0]
    carries the typed ArtifactReference; agentic-result-summarizer
    is skipped per AS-7 amended) or remains inline (``inline`` —
    envelope.primary carries the summarized content; agentic-result-
    summarizer fires per ADR-004 within the inline scope).

    ``None`` (the field missing in YAML) means the dispatch path
    applies its category default — capability ensembles default to
    ``artifact``; system ensembles default to ``inline``. The loader
    stays a faithful YAML→dataclass translator; category-defaulting
    is the dispatch site's concern. Values outside the closed pair
    load as ``None`` rather than raising so a typo surfaces as
    ``defaults applied`` at dispatch time, not a session-start crash.
    """

    output_retention: str | None = None
    """ADR-025 optional retention semantics for substrate-routed
    deliverables (Cycle 6 WP-E).

    Accepted values ``"session"`` / ``"durable"`` / ``"ephemeral"``
    per ADR-025 §"Retention semantics":

    * ``session`` — retained for the session's lifetime; cleaned up
      when the session closes. The dispatch-site default for
      substrate-routed deliverables that are substantive but not
      promoted.
    * ``durable`` — retained indefinitely; survives session close.
      Used when the operator (or downstream skill framework) requests
      preservation.
    * ``ephemeral`` — retained only until the orchestrator's next
      turn; cleaned up after. Used for intermediate-stage deliverables
      that downstream stages consume immediately.

    ``None`` (the field missing in YAML) means the dispatch site
    applies ``session`` as the documented default for substrate-routed
    ensembles. Values outside the closed triple load as ``None``
    rather than raising — same tolerant posture as ``output_substrate``.
    """

    calibration_substrate_access: str | None = None
    """ADR-025 §"Calibration-gate evaluation surface" optional opt-in
    (Cycle 6 WP-E).

    Accepted values ``"summary"`` and ``"artifact"``:

    * ``summary`` (default at dispatch site when absent) — Calibration
      Gate critic agents evaluate envelope.primary + artifacts[0].summary
      only. Lowest-cost evaluation path; sufficient for ensembles whose
      quality is reasonably inferrable from a summary line
      (web-searcher URL-snippet records, text-summarizer paragraph
      summaries).
    * ``artifact`` — critic agents receive a tool-call surface to read
      ``artifacts[0].path`` and evaluate against actual deliverable
      content. Highest-cost evaluation path; required for ensembles
      whose quality cannot be evaluated from summary alone (notably
      ``code-generator`` — code correctness needs the code).

    ``None`` (the field missing in YAML) means the gate applies
    summary-only evaluation per the documented default. Values outside
    the closed pair load as ``None`` — typos surface as ``default
    applied`` rather than session-start crashes.
    """


@dataclass(frozen=True)
class EnsembleValidationResult:
    """One YAML that failed validation during ``EnsembleLoader.prime``.

    Cycle 6 WP-B piece 3 — validate-once-at-load (ADR-023 §"Noise-floor
    remediation"). The Operator-Terminal Event Sink consumes these
    results at serve startup and emits one ``WARN`` line per failure
    via :meth:`OperatorTerminalEventSink.emit_validation_warning`,
    eliminating the per-``list_ensembles()`` re-validation noise that
    Cycle 5 PLAY note 19 (sharpened by the Cycle 6 DISCOVER finding 7)
    surfaced as the operator-experience baseline.
    """

    yaml_path: str
    error: str


class EnsembleLoader:
    """Loads ensemble configurations from files.

    Cycle 6 adds a stateful validate-once-at-load cache populated by
    :meth:`prime`. Primed callers (the agentic-serving startup wiring)
    pay the validation cost once at startup or library reload; subsequent
    :meth:`list_ensembles` lookups hit the cache without re-walking the
    directory. Un-primed callers (CLI, MCP, ad-hoc instantiations) keep
    the existing on-demand validation behavior including the
    ``Skipping invalid ensemble`` warning log line, preserving the
    backward-compatible surface for non-serving entry points.
    """

    def __init__(self) -> None:
        # Per-directory caches keyed by the resolved absolute path so
        # ``a/b/`` and ``a/b`` (and chdir-relative variants) collapse onto
        # a single entry. The cache is empty until :meth:`prime` runs.
        self._cache: dict[str, list[EnsembleConfig]] = {}
        self._validation_results: dict[str, list[EnsembleValidationResult]] = {}

    def prime(self, directory: str) -> None:
        """Walk the directory once, validate every YAML, populate the cache.

        Successful loads land in the cache for subsequent
        :meth:`list_ensembles` lookups. Failures land in
        :meth:`validation_results` so the Operator-Terminal Event Sink
        can emit one ``WARN`` line per failure at startup rather than
        the loader re-emitting them on every enumeration.

        A missing directory is a no-op — operators with partial
        deployment shapes (e.g., no local ensembles directory yet) do
        not get a startup crash.
        """
        key = self._cache_key(directory)
        ensembles, results = self._walk_and_validate(directory, silent=True)
        self._cache[key] = ensembles
        self._validation_results[key] = results

    def reload(self, directory: str) -> None:
        """Re-prime a single directory, replacing its prior cached state.

        Per ADR-023 Direction-not-constraint on operator affordance —
        operators trigger reload via ``SIGHUP``, an admin endpoint, or a
        full restart. File-watch auto-reload was rejected in favor of
        explicit-reload semantics so silent background reloads do not
        emit ``WARN`` lines mid-session.
        """
        self.prime(directory)

    def validation_results(self) -> tuple[EnsembleValidationResult, ...]:
        """Return every validation failure across all primed directories."""
        results: list[EnsembleValidationResult] = []
        for directory_results in self._validation_results.values():
            results.extend(directory_results)
        return tuple(results)

    def load_from_file(
        self,
        file_path: str,
        search_dirs: list[str] | None = None,
    ) -> EnsembleConfig:
        """Load ensemble configuration from a YAML file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Ensemble file not found: {file_path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Support both default_task (preferred) and task (backward compat)
        default_task = data.get("default_task") or data.get("task")

        # Parse each agent dict into typed AgentConfig (ADR-012)
        agents = [parse_agent_config(a) for a in data["agents"]]

        topaz_skill_raw = data.get("topaz_skill")
        topaz_skill: str | None = (
            str(topaz_skill_raw) if topaz_skill_raw is not None else None
        )

        output_schema_raw = data.get("output_schema")
        output_schema: dict[str, Any] | None = (
            dict(output_schema_raw) if isinstance(output_schema_raw, dict) else None
        )

        output_substrate = _validated_literal(
            data.get("output_substrate"),
            {"artifact", "inline"},
        )
        output_retention = _validated_literal(
            data.get("output_retention"),
            {"session", "durable", "ephemeral"},
        )
        calibration_substrate_access = _validated_literal(
            data.get("calibration_substrate_access"),
            {"summary", "artifact"},
        )

        config = EnsembleConfig(
            name=data["name"],
            description=data["description"],
            agents=agents,
            default_task=default_task,
            task=data.get("task"),
            validation=data.get("validation"),
            test_mode=data.get("test_mode"),
            raw_output=bool(data.get("raw_output", False)),
            topaz_skill=topaz_skill,
            output_schema=output_schema,
            output_substrate=output_substrate,
            output_retention=output_retention,
            calibration_substrate_access=calibration_substrate_access,
        )

        # Validate agent dependencies
        self._validate_dependencies(config.agents)

        # Cross-ensemble cycle detection (ADR-013, Invariant 5)
        if search_dirs:
            validate_ensemble_reference_graph(config.name, agents, search_dirs)

        return config

    def list_ensembles(self, directory: str) -> list[EnsembleConfig]:
        """List all ensemble configurations in a directory and subdirectories.

        Primed callers hit the cache populated by :meth:`prime` and pay
        zero validation cost on the lookup. Un-primed callers walk the
        directory on demand and emit the existing
        ``Skipping invalid ensemble`` log line per failure — preserving
        backward compatibility for CLI and MCP entry points that have
        not adopted the startup-prime pattern.
        """
        key = self._cache_key(directory)
        if key in self._cache:
            return list(self._cache[key])

        ensembles, _results = self._walk_and_validate(directory, silent=False)
        return ensembles

    def _walk_and_validate(
        self, directory: str, *, silent: bool
    ) -> tuple[list[EnsembleConfig], list[EnsembleValidationResult]]:
        """Walk ``directory`` and validate every YAML found.

        When ``silent`` is ``True`` (the prime path), validation failures
        are captured in the returned list rather than emitted to the
        loader's ``logger.warning`` surface — the Operator-Terminal Event
        Sink owns the visible WARN format. When ``silent`` is ``False``
        (the fallback path for un-primed callers), the existing
        per-failure log line is preserved for CLI / MCP compatibility.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return [], []

        ensembles: list[EnsembleConfig] = []
        failures: list[EnsembleValidationResult] = []
        for yaml_file in self._find_yaml_files(dir_path):
            try:
                config = self.load_from_file(
                    str(yaml_file),
                    search_dirs=[directory],
                )
                relative_path = yaml_file.relative_to(dir_path)
                config.relative_path = (
                    str(relative_path.parent)
                    if relative_path.parent != Path(".")
                    else None
                )
                ensembles.append(config)
            except Exception as exc:
                failures.append(
                    EnsembleValidationResult(yaml_path=str(yaml_file), error=str(exc))
                )
                if not silent:
                    logger.warning("Skipping invalid ensemble %s: %s", yaml_file, exc)
                continue

        return ensembles, failures

    @staticmethod
    def _cache_key(directory: str) -> str:
        """Normalize a directory string to its resolved absolute path."""
        path = Path(directory)
        try:
            return str(path.resolve())
        except OSError:
            return str(path)

    @staticmethod
    def _find_yaml_files(dir_path: Path) -> list[Path]:
        """Find all .yaml and .yml files recursively."""
        files = list(dir_path.rglob("*.yaml"))
        files.extend(dir_path.rglob("*.yml"))
        return files

    def _validate_dependencies(self, agents: list[AgentConfig]) -> None:
        """Validate agent dependencies for cycles and missing deps."""
        _check_missing_dependencies(agents)
        _validate_fan_out_dependencies(agents)
        assert_no_cycles(agents)

    def _find_ensemble_in_dirs(
        self,
        name: str,
        search_dirs: list[str],
    ) -> EnsembleConfig | None:
        """Find an ensemble by name across search directories.

        Delegates to the module-level helper shared with the cross-ensemble
        cycle validator so both resolution paths stay identical.
        """
        return _find_ensemble_in_dirs(name, search_dirs)

    def find_ensemble(self, directory: str, name: str) -> EnsembleConfig | None:
        """Find an ensemble by name in a directory, supporting hierarchical names.

        Supports matching by:
        - Simple name: "my-ensemble"
        - Full hierarchical name: "examples/my-ensemble/my-ensemble"
        - Directory path (if name matches): "examples/my-ensemble"
        """
        ensembles = self.list_ensembles(directory)
        for ensemble in ensembles:
            # Build potential matching patterns
            display_name = (
                f"{ensemble.relative_path}/{ensemble.name}"
                if ensemble.relative_path
                else ensemble.name
            )

            # Also support matching by directory path (relative_path alone)
            # if the last component matches the ensemble name
            directory_path = ensemble.relative_path

            # Check all matching patterns
            if ensemble.name == name or display_name == name:
                return ensemble

            # Support "examples/neon-shadows" matching an ensemble at
            # examples/neon-shadows/ensemble.yaml with name: neon-shadows
            if (
                directory_path
                and directory_path == name
                and name.endswith(ensemble.name)
            ):
                return ensemble

        return None


def validate_ensemble_reference_graph(
    name: str,
    agents: list[AgentConfig],
    search_dirs: list[str],
) -> None:
    """Validate that the ensemble reference graph is acyclic (Invariant 5).

    Public validator shared by the load path (``EnsembleLoader.load_from_file``)
    and the MCP/web validate path (``ValidationHandler``). Composition-time
    callers (``compose_ensemble``) share this same routine so their behavior
    cannot diverge from the load path.

    Args:
        name: Root ensemble name.
        agents: Agent configurations of the root ensemble.
        search_dirs: Directories used to resolve ensemble-to-ensemble
            references while walking the reference graph.

    Raises:
        ValueError: If a cycle is detected; the message names the cycle path.
    """
    graph: dict[str, list[str]] = {}
    _build_reference_graph(name, agents, search_dirs, graph)

    visited: set[str] = set()
    in_stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> None:
        if node in in_stack:
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            msg = "cross-ensemble cycle detected: " + " -> ".join(cycle)
            raise ValueError(msg)
        if node in visited:
            return

        visited.add(node)
        in_stack.add(node)
        path.append(node)

        for ref in graph.get(node, []):
            dfs(ref)

        in_stack.remove(node)
        path.pop()

    dfs(name)


def compute_reference_graph_depth(
    name: str,
    agents: list[AgentConfig],
    search_dirs: list[str],
) -> int:
    """Return the maximum depth of the reference graph rooted at ``name``.

    Depth 0 is a leaf (no ensemble references). Depth N is a node whose
    deepest descendant is N edges away. Mirrors the runtime depth
    counter in :class:`EnsembleAgentRunner` — an N-edge chain would
    execute with child_depth == N, so callers compare the returned
    value against the configured depth limit and reject if
    ``depth > depth_limit``.

    Assumes the graph is acyclic; callers that need cycle rejection
    must invoke :func:`validate_ensemble_reference_graph` first.
    """
    graph: dict[str, list[str]] = {}
    _build_reference_graph(name, agents, search_dirs, graph)

    memo: dict[str, int] = {}

    def depth_of(node: str) -> int:
        if node in memo:
            return memo[node]
        refs = graph.get(node, [])
        if not refs:
            memo[node] = 0
            return 0
        memo[node] = 1 + max(depth_of(ref) for ref in refs)
        return memo[node]

    return depth_of(name)


def _build_reference_graph(
    ensemble_name: str,
    agents: list[AgentConfig],
    search_dirs: list[str],
    graph: dict[str, list[str]],
) -> None:
    """Recursively build the ensemble reference graph for cycle detection."""
    if ensemble_name in graph:
        return

    refs = [a.ensemble for a in agents if isinstance(a, EnsembleAgentConfig)]
    graph[ensemble_name] = refs

    for ref_name in refs:
        if ref_name in graph:
            continue
        ref_config = _find_ensemble_in_dirs(ref_name, search_dirs)
        if ref_config:
            _build_reference_graph(
                ref_name,
                ref_config.agents,
                search_dirs,
                graph,
            )


def _find_ensemble_in_dirs(
    name: str,
    search_dirs: list[str],
) -> EnsembleConfig | None:
    """Find an ensemble by name across search directories.

    Loads without cycle detection (no ``search_dirs`` is passed through) so
    the validator does not recurse into itself while building the reference
    graph.
    """
    loader = EnsembleLoader()
    for directory in search_dirs:
        dir_path = Path(directory)
        for ext in ("yaml", "yml"):
            candidate = dir_path / f"{name}.{ext}"
            if candidate.exists():
                return loader.load_from_file(str(candidate))
    return None
