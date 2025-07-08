"""Ensemble configuration loading and management."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnsembleConfig:
    """Configuration for an ensemble of agents."""

    name: str
    description: str
    agents: list[dict[str, Any]]
    coordinator: dict[str, Any]


class EnsembleLoader:
    """Loads ensemble configurations from files."""

    def load_from_file(self, file_path: str) -> EnsembleConfig:
        """Load ensemble configuration from a YAML file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Ensemble file not found: {file_path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return EnsembleConfig(
            name=data["name"],
            description=data["description"],
            agents=data["agents"],
            coordinator=data["coordinator"],
        )

    def list_ensembles(self, directory: str) -> list[EnsembleConfig]:
        """List all ensemble configurations in a directory."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return []

        ensembles = []
        for yaml_file in dir_path.glob("*.yaml"):
            try:
                config = self.load_from_file(str(yaml_file))
                ensembles.append(config)
            except Exception:
                # Skip invalid files
                continue

        # Also check for .yml files
        for yml_file in dir_path.glob("*.yml"):
            try:
                config = self.load_from_file(str(yml_file))
                ensembles.append(config)
            except Exception:
                # Skip invalid files
                continue

        return ensembles

    def find_ensemble(self, directory: str, name: str) -> EnsembleConfig | None:
        """Find an ensemble by name in a directory."""
        ensembles = self.list_ensembles(directory)
        for ensemble in ensembles:
            if ensemble.name == name:
                return ensemble
        return None
