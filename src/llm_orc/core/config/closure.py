"""The transitive closure of an ensemble: everything it needs to run.

Delegation is "ship the closure, preflight transitively, resolve
explicitly, run" (docs/plans/2026-09-28-remote-delegation.md). This
module is the first two words: from a root ensemble it collects every
child ensemble (``ensemble:``, ``loop.body``, a literal ``dispatch:``),
every script, every profile with its ``fallback_model_profile`` chain,
and every inline model, each with the path of agents that reaches it.
The preflight classifier (services/handlers/preflight.py) and, later,
the CLI's closure shipper both consume it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.schemas.agent_config import (
    DynamicDispatchAgentConfig,
    EnsembleAgentConfig,
    LlmAgentConfig,
    LoopAgentConfig,
    ScriptAgentConfig,
)

Kind = Literal["ensemble", "dispatch", "script", "profile", "model"]
Key = tuple[str, str]
FindEnsemble = Callable[[str], EnsembleConfig | None]


@dataclass(frozen=True)
class Dependency:
    """One thing an ensemble needs, and the agent path that names it."""

    kind: Kind
    name: str
    via: tuple[str, ...]
    provider: str | None = None
    found: bool = True


@dataclass(frozen=True)
class Closure:
    """The dependency set in walk order, plus which frame owns what.

    ``dependencies`` is deduplicated on ``(kind, name)``: the closure is
    a set, and the first sighting keeps its path. ``owned`` maps every
    frame (``"<ensemble>.<agent>"``) to every key reachable through it,
    duplicates included, so a per-agent verdict never misses a
    dependency that was first reported under a sibling.
    """

    dependencies: list[Dependency]
    owned: dict[str, frozenset[Key]]


def walk_closure(
    root: EnsembleConfig,
    find_ensemble: FindEnsemble,
    profiles: Mapping[str, Mapping[str, Any]],
) -> Closure:
    """Depth-first in agent order from ``root``.

    A child that does not resolve is recorded ``found=False`` and not
    walked. A ``${...}`` dispatch target cannot be followed statically
    (issue #94) and is recorded as ``dispatch``. Cycles cannot load
    (Invariant 5) but the walk keeps its own visited set regardless.
    """
    walker = _Walker(find_ensemble, profiles)
    walker.add(Dependency("ensemble", root.name, ()), ())
    walker.visit(root, ())
    return Closure(walker.deps, {f: frozenset(k) for f, k in walker.owned.items()})


class _Walker:
    def __init__(
        self, find_ensemble: FindEnsemble, profiles: Mapping[str, Mapping[str, Any]]
    ) -> None:
        self._find = find_ensemble
        self._profiles = profiles
        self.deps: list[Dependency] = []
        self.owned: dict[str, set[Key]] = {}
        self._seen: set[Key] = set()
        self._visited: set[str] = set()
        self._stack: list[str] = []  # ensembles being walked, root first
        self._members: dict[str, set[Key]] = {}  # ensemble -> keys under it

    def add(self, dep: Dependency, via: tuple[str, ...]) -> bool:
        """Record ``dep`` under every frame of ``via`` and every ensemble
        on the stack; True on first sight."""
        key: Key = (dep.kind, dep.name)
        self._own(key, via)
        if key in self._seen:
            return False
        self._seen.add(key)
        self.deps.append(dep)
        return True

    def _own(self, key: Key, via: tuple[str, ...]) -> None:
        for frame in via:
            self.owned.setdefault(frame, set()).add(key)
        for ensemble in self._stack:
            self._members.setdefault(ensemble, set()).add(key)

    def visit(self, config: EnsembleConfig, via: tuple[str, ...]) -> None:
        if config.name in self._visited:
            return
        self._visited.add(config.name)
        self._stack.append(config.name)
        try:
            for agent in config.agents:
                self._visit_agent(agent, (*via, f"{config.name}.{agent.name}"))
        finally:
            self._stack.pop()

    def _visit_agent(self, agent: Any, via: tuple[str, ...]) -> None:
        if isinstance(agent, ScriptAgentConfig):
            self.add(Dependency("script", agent.script, via), via)
        elif isinstance(agent, EnsembleAgentConfig):
            self._child(agent.ensemble, via)
        elif isinstance(agent, LoopAgentConfig):
            self._child(agent.loop.body, via)
        elif isinstance(agent, DynamicDispatchAgentConfig):
            if "${" in agent.dispatch:
                self.add(Dependency("dispatch", agent.dispatch, via), via)
            else:
                self._child(agent.dispatch, via)
        elif isinstance(agent, LlmAgentConfig):
            self._llm(agent, via)

    def _child(self, name: str, via: tuple[str, ...]) -> None:
        child = self._find(name)
        first = self.add(
            Dependency("ensemble", name, via, found=child is not None), via
        )
        if child is None:
            return
        if first:
            self.visit(child, via)
            return
        # Already walked under another frame: this frame owns its members too.
        for key in self._members.get(name, set()):
            self._own(key, via)

    def _llm(self, agent: LlmAgentConfig, via: tuple[str, ...]) -> None:
        if agent.model_profile is not None:
            self._profile_chain(agent.model_profile, via)
            if agent.fallback_model_profile:
                self._profile_chain(agent.fallback_model_profile, via)
        elif agent.model is not None:
            self.add(
                Dependency("model", agent.model, via, provider=agent.provider), via
            )

    def _profile_chain(self, name: str, via: tuple[str, ...]) -> None:
        current: str | None = name
        while current:
            profile = self._profiles.get(current)
            provider = profile.get("provider") if profile is not None else None
            dep = Dependency(
                "profile",
                current,
                via,
                provider=str(provider) if provider else None,
                found=profile is not None,
            )
            if not self.add(dep, via) or profile is None:
                return
            nxt = profile.get("fallback_model_profile")
            current = nxt if isinstance(nxt, str) and nxt else None
