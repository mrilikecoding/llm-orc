"""LLM Orchestra - Multi-agent LLM communication system."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-orchestra")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
