"""The transitive closure of an ensemble: everything it needs to run.

Delegation is "ship the closure, preflight transitively, resolve
explicitly, run" (docs/plans/2026-09-28-remote-delegation.md). This
module is the first two words: from a root ensemble it collects every
child ensemble (``ensemble:``, ``loop.body``, a literal ``dispatch:``),
every script, every profile with its ``fallback_model_profile`` chain,
and every inline model, each with the path of agents that reaches it.
Everything is keyed by the reference string (the name the caller or the
parent agent wrote), never by a config's ``name:`` field: two files may
share a ``name:``, and the run resolves by reference.
The preflight classifier (services/handlers/preflight.py) and, later,
the CLI's closure shipper both consume it.
"""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.scripting.files_block import ListedFiles
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
    # A file a script lists in its llm-orc block (ruling 8): the owner's
    # name and the path as written. ``name`` is the path joined onto the
    # owner's directory, so two scripts listing one name do not collide.
    beside: str | None = None
    listed: str | None = None
    # Why the script's own block cannot be read; the script is unmet.
    problem: str | None = None


@dataclass(frozen=True)
class ScriptListing:
    """What the caller found for one script dependency: whether the file
    is there (a listed file is looked for beside its owner only) and
    what its block lists."""

    found: bool
    files: ListedFiles = ListedFiles()


ScriptFilesOf = Callable[[Dependency], ScriptListing]


@dataclass(frozen=True)
class Closure:
    """The dependency set in walk order, plus which frame owns what.

    ``dependencies`` is deduplicated on ``(kind, name)``: the closure is
    a set, and the first sighting keeps its path. ``owned`` maps every
    frame (``"<ensemble reference>.<agent>"``) to every key reachable
    through it,
    duplicates included, so a per-agent verdict never misses a
    dependency that was first reported under a sibling.
    """

    dependencies: list[Dependency]
    owned: dict[str, frozenset[Key]]


def walk_closure(
    root: EnsembleConfig,
    find_ensemble: FindEnsemble,
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    root_ref: str,
    script_files: ScriptFilesOf | None = None,
) -> Closure:
    """Depth-first in agent order from ``root``, which the caller named
    ``root_ref``.

    A child that does not resolve is recorded ``found=False`` and not
    walked. A ``${...}`` dispatch target cannot be followed statically
    (issue #94) and is recorded as ``dispatch``. Cycles cannot load
    (Invariant 5) but the walk keeps its own visited set regardless.

    ``script_files`` reads a script's llm-orc block (the caller resolves
    the script with the run's own resolver); each listed file becomes a
    ``script`` dependency the script's frames own, and a listed file's
    own block is followed.
    """
    walker = _Walker(find_ensemble, profiles, script_files)
    walker.add(Dependency("ensemble", root_ref, ()), ())
    walker.visit(root, (), root_ref)
    return Closure(walker.deps, {f: frozenset(k) for f, k in walker.owned.items()})


def _unmet(dep: Dependency) -> bool:
    return not dep.found or dep.problem is not None


class _Walker:
    def __init__(
        self,
        find_ensemble: FindEnsemble,
        profiles: Mapping[str, Mapping[str, Any]],
        script_files: ScriptFilesOf | None = None,
    ) -> None:
        self._find = find_ensemble
        self._profiles = profiles
        self._script_files = script_files
        self.deps: list[Dependency] = []
        self.owned: dict[str, set[Key]] = {}
        self._seen: set[Key] = set()
        self._visited: set[str] = set()
        self._stack: list[str] = []  # references being walked, root first
        self._members: dict[str, set[Key]] = {}  # reference -> keys under it

    def add(self, dep: Dependency, via: tuple[str, ...]) -> bool:
        """Record ``dep`` under every frame of ``via`` and every ensemble
        on the stack; True on first sight."""
        key: Key = (dep.kind, dep.name)
        self._own(key, via)
        if key in self._seen:
            self._prefer_unmet(key, dep)
            return False
        self._seen.add(key)
        self.deps.append(dep)
        return True

    def _prefer_unmet(self, key: Key, dep: Dependency) -> None:
        """One name can be sighted twice with different outcomes (a listed
        file is looked for beside its owner only). An unmet sighting
        replaces a met one in place, so the row cannot read ready."""
        if dep.kind != "script" or not _unmet(dep):
            return
        for index, recorded in enumerate(self.deps):
            if (recorded.kind, recorded.name) == key and not _unmet(recorded):
                self.deps[index] = dep

    def _own(self, key: Key, via: tuple[str, ...]) -> None:
        for frame in via:
            self.owned.setdefault(frame, set()).add(key)
        for ensemble in self._stack:
            self._members.setdefault(ensemble, set()).add(key)

    def visit(self, config: EnsembleConfig, via: tuple[str, ...], ref: str) -> None:
        if ref in self._visited:
            return
        self._visited.add(ref)
        self._stack.append(ref)
        try:
            for agent in config.agents:
                self._visit_agent(agent, (*via, f"{ref}.{agent.name}"))
        finally:
            self._stack.pop()

    def _visit_agent(self, agent: Any, via: tuple[str, ...]) -> None:
        if isinstance(agent, ScriptAgentConfig):
            self._script(Dependency("script", agent.script, via), via, frozenset())
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

    def _script(
        self, dep: Dependency, via: tuple[str, ...], following: frozenset[str]
    ) -> None:
        """Record ``dep`` and, through its block, the files it lists."""
        listing = self._script_files(dep) if self._script_files else ScriptListing(True)
        self.add(replace(dep, found=listing.found, problem=listing.files.error), via)
        if dep.name in following:
            return
        for path in listing.files.paths:
            member = Dependency(
                "script",
                posixpath.join(posixpath.dirname(dep.name), path),
                via,
                beside=dep.name,
                listed=path,
            )
            self._script(member, via, following | {dep.name})

    def _child(self, name: str, via: tuple[str, ...]) -> None:
        child = self._find(name)
        first = self.add(
            Dependency("ensemble", name, via, found=child is not None), via
        )
        if child is None:
            return
        if first:
            self.visit(child, via, name)
            return
        # Already walked under another frame: this frame owns its members too.
        for key in self._members.get(name, set()):
            self._own(key, via)

    def _llm(self, agent: LlmAgentConfig, via: tuple[str, ...]) -> None:
        # Mirrors model_factory._reachable_provider_options: the primary
        # profile's chain is walked transitively, the agent-level fallback
        # is one hop and its own chain is never followed.
        if agent.model_profile is not None:
            self._profile_chain(agent.model_profile, via)
        elif agent.model is not None:
            self.add(
                Dependency("model", agent.model, via, provider=agent.provider), via
            )
        if agent.fallback_model_profile:
            self._profile(agent.fallback_model_profile, via)

    def _profile(self, name: str, via: tuple[str, ...]) -> Mapping[str, Any] | None:
        """Record one profile hop (owned even on a repeat sighting)."""
        profile = self._profiles.get(name)
        provider = profile.get("provider") if profile is not None else None
        dep = Dependency(
            "profile",
            name,
            via,
            provider=str(provider) if provider else None,
            found=profile is not None,
        )
        self.add(dep, via)
        return profile

    def _profile_chain(self, name: str, via: tuple[str, ...]) -> None:
        """Follow ``fallback_model_profile`` hops, owning each for ``via``.

        Stops at a missing hop or one already seen in this chain, so a
        profile cycle terminates while a repeat sighting from another
        frame still owns the whole chain.
        """
        in_chain: set[str] = set()
        current: str | None = name
        while current and current not in in_chain:
            in_chain.add(current)
            profile = self._profile(current, via)
            if profile is None:
                return
            nxt = profile.get("fallback_model_profile")
            current = nxt if isinstance(nxt, str) and nxt else None
