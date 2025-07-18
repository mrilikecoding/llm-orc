"""Tests for ensemble configuration loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from llm_orc.ensemble_config import EnsembleConfig, EnsembleLoader


class TestEnsembleConfig:
    """Test ensemble configuration."""

    def test_ensemble_config_creation(self) -> None:
        """Test creating an ensemble configuration."""
        config = EnsembleConfig(
            name="test_ensemble",
            description="A test ensemble",
            agents=[
                {"name": "agent1", "role": "tester", "model": "claude-3-sonnet"},
                {"name": "agent2", "role": "reviewer", "model": "claude-3-sonnet"},
                {
                    "name": "synthesizer",
                    "role": "synthesizer",
                    "model": "claude-3-sonnet",
                    "depends_on": ["agent1", "agent2"],
                    "synthesis_prompt": "Combine the results",
                    "output_format": "json",
                },
            ],
        )

        assert config.name == "test_ensemble"
        assert config.description == "A test ensemble"
        assert len(config.agents) == 3

        # Find synthesizer agent and verify its properties
        synthesizer = next(
            agent for agent in config.agents if agent["name"] == "synthesizer"
        )
        assert synthesizer["output_format"] == "json"


class TestEnsembleLoader:
    """Test ensemble configuration loading."""

    def test_load_ensemble_from_yaml(self) -> None:
        """Test loading ensemble configuration from YAML file."""
        # Create a temporary YAML file
        ensemble_yaml = {
            "name": "pr_review",
            "description": "Multi-perspective PR review ensemble",
            "agents": [
                {
                    "name": "security_reviewer",
                    "role": "security_analyst",
                    "model": "claude-3-sonnet",
                },
                {
                    "name": "performance_reviewer",
                    "role": "performance_analyst",
                    "model": "claude-3-sonnet",
                },
                {
                    "name": "synthesizer",
                    "role": "synthesizer",
                    "model": "claude-3-sonnet",
                    "depends_on": ["security_reviewer", "performance_reviewer"],
                    "synthesis_prompt": "Synthesize security and performance feedback",
                    "output_format": "structured",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            loader = EnsembleLoader()
            config = loader.load_from_file(yaml_path)

            assert config.name == "pr_review"
            assert len(config.agents) == 3
            assert config.agents[0]["name"] == "security_reviewer"

            # Find synthesizer and verify its properties
            synthesizer = next(
                agent for agent in config.agents if agent["name"] == "synthesizer"
            )
            assert synthesizer["output_format"] == "structured"
        finally:
            Path(yaml_path).unlink()

    def test_list_ensembles_in_directory(self) -> None:
        """Test listing available ensembles in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a couple of ensemble files
            ensemble1 = {
                "name": "ensemble1",
                "description": "First ensemble",
                "agents": [{"name": "agent1", "role": "role1", "model": "model1"}],
            }

            ensemble2 = {
                "name": "ensemble2",
                "description": "Second ensemble",
                "agents": [{"name": "agent2", "role": "role2", "model": "model2"}],
            }

            # Write ensemble files
            with open(f"{temp_dir}/ensemble1.yaml", "w") as f:
                yaml.dump(ensemble1, f)
            with open(f"{temp_dir}/ensemble2.yaml", "w") as f:
                yaml.dump(ensemble2, f)

            # Also create a non-yaml file that should be ignored
            with open(f"{temp_dir}/not_an_ensemble.txt", "w") as f:
                f.write("This should be ignored")

            loader = EnsembleLoader()
            ensembles = loader.list_ensembles(temp_dir)

            assert len(ensembles) == 2
            ensemble_names = [e.name for e in ensembles]
            assert "ensemble1" in ensemble_names
            assert "ensemble2" in ensemble_names

    def test_load_nonexistent_ensemble(self) -> None:
        """Test loading a nonexistent ensemble raises appropriate error."""
        loader = EnsembleLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_from_file("/nonexistent/path.yaml")

    def test_find_ensemble_by_name(self) -> None:
        """Test finding an ensemble by name in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create an ensemble file
            ensemble = {
                "name": "target_ensemble",
                "description": "Target ensemble",
                "agents": [{"name": "agent", "role": "role", "model": "model"}],
            }

            with open(f"{temp_dir}/target_ensemble.yaml", "w") as f:
                yaml.dump(ensemble, f)

            loader = EnsembleLoader()
            config = loader.find_ensemble(temp_dir, "target_ensemble")

            assert config is not None
            assert config.name == "target_ensemble"

            # Test finding nonexistent ensemble
            config = loader.find_ensemble(temp_dir, "nonexistent")
            assert config is None

    def test_dependency_based_ensemble_without_coordinator(self) -> None:
        """Test new dependency-based ensemble without coordinator field."""
        # RED: This test should fail initially since we haven't updated the code
        ensemble_yaml = {
            "name": "dependency_ensemble",
            "description": "Ensemble using agent dependencies",
            "agents": [
                {
                    "name": "researcher",
                    "model_profile": "fast-model",
                    "system_prompt": "Research the topic thoroughly",
                },
                {
                    "name": "analyst",
                    "model_profile": "quality-model",
                    "system_prompt": "Analyze the research findings",
                },
                {
                    "name": "synthesizer",
                    "model_profile": "quality-model",
                    "system_prompt": "Synthesize research and analysis",
                    "depends_on": ["researcher", "analyst"],
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            loader = EnsembleLoader()
            config = loader.load_from_file(yaml_path)

            assert config.name == "dependency_ensemble"
            assert len(config.agents) == 3

            # Find synthesizer agent and verify its dependencies
            synthesizer = next(
                agent for agent in config.agents if agent["name"] == "synthesizer"
            )
            assert synthesizer["depends_on"] == ["researcher", "analyst"]

        finally:
            Path(yaml_path).unlink()

    def test_dependency_validation_detects_cycles(self) -> None:
        """Test that dependency validation catches circular dependencies."""
        # RED: This should fail until we implement dependency validation
        ensemble_yaml = {
            "name": "circular_ensemble",
            "description": "Ensemble with circular dependencies",
            "agents": [
                {
                    "name": "agent_a",
                    "model_profile": "test-model",
                    "depends_on": ["agent_b"],
                },
                {
                    "name": "agent_b",
                    "model_profile": "test-model",
                    "depends_on": ["agent_a"],  # Creates cycle
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            loader = EnsembleLoader()
            with pytest.raises(ValueError, match="Circular dependency"):
                loader.load_from_file(yaml_path)
        finally:
            Path(yaml_path).unlink()

    def test_dependency_validation_detects_missing_deps(self) -> None:
        """Test that dependency validation catches missing dependencies."""
        # RED: This should fail until we implement dependency validation
        ensemble_yaml = {
            "name": "missing_dep_ensemble",
            "description": "Ensemble with missing dependencies",
            "agents": [
                {
                    "name": "dependent_agent",
                    "model_profile": "test-model",
                    "depends_on": ["nonexistent_agent"],  # Missing dep
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            loader = EnsembleLoader()
            with pytest.raises(ValueError, match="missing dependency"):
                loader.load_from_file(yaml_path)
        finally:
            Path(yaml_path).unlink()
