"""Tests for ensemble configuration loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from llm_orc.core.config.ensemble_config import (
    EnsembleConfig,
    EnsembleLoader,
    assert_no_cycles,
    detect_cycle,
)
from llm_orc.schemas.agent_config import AgentConfig, LlmAgentConfig, ScriptAgentConfig


class TestEnsembleConfig:
    """Test ensemble configuration."""

    def test_ensemble_config_creation(self) -> None:
        """Test creating an ensemble configuration."""
        config = EnsembleConfig(
            name="test_ensemble",
            description="A test ensemble",
            agents=[
                LlmAgentConfig(
                    name="agent1", model="claude-3-sonnet", provider="anthropic"
                ),
                LlmAgentConfig(
                    name="agent2", model="claude-3-sonnet", provider="anthropic"
                ),
                LlmAgentConfig(
                    name="synthesizer",
                    model="claude-3-sonnet",
                    provider="anthropic",
                    depends_on=["agent1", "agent2"],
                    output_format="json",
                ),
            ],
        )

        assert config.name == "test_ensemble"
        assert config.description == "A test ensemble"
        assert len(config.agents) == 3

        # Find synthesizer agent and verify its properties
        synthesizer = next(
            agent for agent in config.agents if agent.name == "synthesizer"
        )
        assert isinstance(synthesizer, LlmAgentConfig)
        assert synthesizer.output_format == "json"


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
                    "model": "claude-3-sonnet",
                    "provider": "anthropic",
                },
                {
                    "name": "performance_reviewer",
                    "model": "claude-3-sonnet",
                    "provider": "anthropic",
                },
                {
                    "name": "synthesizer",
                    "model": "claude-3-sonnet",
                    "provider": "anthropic",
                    "depends_on": ["security_reviewer", "performance_reviewer"],
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
            assert config.agents[0].name == "security_reviewer"

            # Find synthesizer and verify its properties
            synthesizer = next(
                agent for agent in config.agents if agent.name == "synthesizer"
            )
            assert isinstance(synthesizer, LlmAgentConfig)
            assert synthesizer.output_format == "structured"
        finally:
            Path(yaml_path).unlink()

    def test_list_ensembles_in_directory(self) -> None:
        """Test listing available ensembles in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a couple of ensemble files
            ensemble1 = {
                "name": "ensemble1",
                "description": "First ensemble",
                "agents": [
                    {"name": "agent1", "model": "model1", "provider": "anthropic"}
                ],
            }

            ensemble2 = {
                "name": "ensemble2",
                "description": "Second ensemble",
                "agents": [
                    {"name": "agent2", "model": "model2", "provider": "anthropic"}
                ],
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
                "agents": [
                    {"name": "agent", "model": "model", "provider": "anthropic"}
                ],
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
                agent for agent in config.agents if agent.name == "synthesizer"
            )
            assert synthesizer.depends_on == ["researcher", "analyst"]

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

    def test_list_ensembles_nonexistent_directory(self) -> None:
        """Test listing ensembles from nonexistent directory (line 53)."""
        loader = EnsembleLoader()

        # Test nonexistent directory
        result = loader.list_ensembles("/nonexistent/directory")

        assert result == []

    def test_list_ensembles_with_valid_files(self) -> None:
        """Test listing ensembles from directory with valid files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid .yaml file
            yaml_config = {
                "name": "test_ensemble_yaml",
                "description": "Test ensemble in YAML",
                "agents": [
                    {
                        "name": "agent1",
                        "model": "claude-3-sonnet",
                        "provider": "anthropic",
                    }
                ],
            }
            yaml_file = Path(temp_dir) / "test.yaml"
            with open(yaml_file, "w") as f:
                yaml.dump(yaml_config, f)

            # Create a valid .yml file
            yml_config = {
                "name": "test_ensemble_yml",
                "description": "Test ensemble in YML",
                "agents": [
                    {
                        "name": "agent2",
                        "model": "claude-3-sonnet",
                        "provider": "anthropic",
                    }
                ],
            }
            yml_file = Path(temp_dir) / "test.yml"
            with open(yml_file, "w") as f:
                yaml.dump(yml_config, f)

            loader = EnsembleLoader()
            result = loader.list_ensembles(temp_dir)

            assert len(result) == 2
            names = [config.name for config in result]
            assert "test_ensemble_yaml" in names
            assert "test_ensemble_yml" in names

    def test_list_ensembles_with_invalid_files(self) -> None:
        """Test listing ensembles with invalid files (lines 60-62, 66-71)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create an invalid .yaml file
            invalid_yaml = Path(temp_dir) / "invalid.yaml"
            with open(invalid_yaml, "w") as f:
                f.write("invalid: yaml: content: [")

            # Create an invalid .yml file
            invalid_yml = Path(temp_dir) / "invalid.yml"
            with open(invalid_yml, "w") as f:
                f.write("invalid: yml: content: {")

            # Create a valid file to ensure others still work
            valid_config = {
                "name": "valid_ensemble",
                "description": "Valid ensemble",
                "agents": [
                    {
                        "name": "agent1",
                        "model": "claude-3-sonnet",
                        "provider": "anthropic",
                    }
                ],
            }
            valid_file = Path(temp_dir) / "valid.yaml"
            with open(valid_file, "w") as f:
                yaml.dump(valid_config, f)

            loader = EnsembleLoader()
            result = loader.list_ensembles(temp_dir)

            # Should only return valid ensemble, invalid ones are skipped
            assert len(result) == 1
            assert result[0].name == "valid_ensemble"

    def test_list_ensembles_empty_directory(self) -> None:
        """Test listing ensembles from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = EnsembleLoader()
            result = loader.list_ensembles(temp_dir)

            assert result == []

    def test_list_ensembles_no_yaml_files(self) -> None:
        """Test listing ensembles from directory with no YAML files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create non-YAML files
            (Path(temp_dir) / "readme.txt").write_text("Not a YAML file")
            (Path(temp_dir) / "config.json").write_text('{"not": "yaml"}')

            loader = EnsembleLoader()
            result = loader.list_ensembles(temp_dir)

            assert result == []


class TestValidateDependenciesHelperMethods:
    """Test helper methods extracted from _validate_dependencies for complexity."""

    def test_check_missing_dependencies_no_errors(self) -> None:
        """Test missing dependency check with valid dependencies."""
        from llm_orc.core.config.ensemble_config import _check_missing_dependencies

        agents: list[AgentConfig] = [
            LlmAgentConfig(name="agent1", model_profile="test"),
            LlmAgentConfig(name="agent2", model_profile="test", depends_on=["agent1"]),
            LlmAgentConfig(
                name="agent3",
                model_profile="test",
                depends_on=["agent1", "agent2"],
            ),
        ]

        # Should not raise any exception
        _check_missing_dependencies(agents)

    def test_check_missing_dependencies_single_missing(self) -> None:
        """Test missing dependency check with single missing dependency."""
        from llm_orc.core.config.ensemble_config import _check_missing_dependencies

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="agent1", model_profile="test", depends_on=["missing_agent"]
            ),
        ]

        with pytest.raises(ValueError, match="missing dependency: 'missing_agent'"):
            _check_missing_dependencies(agents)

    def test_check_missing_dependencies_multiple_missing(self) -> None:
        """Test missing dependency check with multiple missing dependencies."""
        from llm_orc.core.config.ensemble_config import _check_missing_dependencies

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="agent1",
                model_profile="test",
                depends_on=["missing1", "missing2"],
            ),
            LlmAgentConfig(
                name="agent2", model_profile="test", depends_on=["missing3"]
            ),
        ]

        with pytest.raises(ValueError, match="missing dependency"):
            _check_missing_dependencies(agents)

    def test_check_missing_dependencies_no_depends_on_field(self) -> None:
        """Test missing dependency check with agents that have no depends_on field."""
        from llm_orc.core.config.ensemble_config import _check_missing_dependencies

        agents: list[AgentConfig] = [
            LlmAgentConfig(name="agent1", model_profile="test"),
            LlmAgentConfig(name="agent2", model_profile="test", depends_on=["agent1"]),
        ]

        # Should not raise any exception
        _check_missing_dependencies(agents)


class TestCycleDetection:
    """Test the consolidated cycle detection (detect_cycle + assert_no_cycles)."""

    def test_detect_cycle_returns_none_when_acyclic(self) -> None:
        """No cycle in a simple chain."""
        agents: list[AgentConfig] = [
            LlmAgentConfig(name="a", model_profile="test"),
            LlmAgentConfig(name="b", model_profile="test", depends_on=["a"]),
            LlmAgentConfig(name="c", model_profile="test", depends_on=["a", "b"]),
        ]
        assert detect_cycle(agents) is None

    def test_detect_cycle_returns_path_for_simple_cycle(self) -> None:
        """A -> B -> A should return the cycle path."""
        agents: list[AgentConfig] = [
            LlmAgentConfig(name="a", model_profile="test", depends_on=["b"]),
            LlmAgentConfig(name="b", model_profile="test", depends_on=["a"]),
        ]
        cycle = detect_cycle(agents)
        assert cycle is not None
        assert "a" in cycle
        assert "b" in cycle

    def test_detect_cycle_returns_path_for_complex_cycle(self) -> None:
        """A -> B -> C -> A should return the 3-node cycle."""
        agents: list[AgentConfig] = [
            LlmAgentConfig(name="a", model_profile="test", depends_on=["b"]),
            LlmAgentConfig(name="b", model_profile="test", depends_on=["c"]),
            LlmAgentConfig(name="c", model_profile="test", depends_on=["a"]),
        ]
        cycle = detect_cycle(agents)
        assert cycle is not None
        assert len(cycle) == 3
        assert set(cycle) == {"a", "b", "c"}

    def test_detect_cycle_self_dependency(self) -> None:
        """Self-dependency is a cycle of length 1."""
        agents: list[AgentConfig] = [
            LlmAgentConfig(name="a", model_profile="test", depends_on=["a"]),
        ]
        cycle = detect_cycle(agents)
        assert cycle == ["a"]

    def test_detect_cycle_mixed_valid_and_cyclic(self) -> None:
        """Only the cyclic subset should appear in the result."""
        agents: list[AgentConfig] = [
            LlmAgentConfig(name="a", model_profile="test"),
            LlmAgentConfig(name="b", model_profile="test", depends_on=["a"]),
            LlmAgentConfig(name="c", model_profile="test", depends_on=["d"]),
            LlmAgentConfig(name="d", model_profile="test", depends_on=["c"]),
        ]
        cycle = detect_cycle(agents)
        assert cycle is not None
        assert set(cycle) == {"c", "d"}

    def test_detect_cycle_handles_dict_form_dependencies(self) -> None:
        """Dict-form deps (conditional) should be traversed."""
        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="a",
                model_profile="test",
                depends_on=[{"agent_name": "b"}],
            ),
            LlmAgentConfig(
                name="b",
                model_profile="test",
                depends_on=[{"agent_name": "a"}],
            ),
        ]
        cycle = detect_cycle(agents)
        assert cycle is not None
        assert set(cycle) == {"a", "b"}

    def test_detect_cycle_empty_agents(self) -> None:
        """Empty agent list has no cycles."""
        assert detect_cycle([]) is None

    def test_assert_no_cycles_passes_for_acyclic(self) -> None:
        """assert_no_cycles should not raise for valid graphs."""
        agents: list[AgentConfig] = [
            LlmAgentConfig(name="a", model_profile="test"),
            LlmAgentConfig(name="b", model_profile="test", depends_on=["a"]),
        ]
        assert_no_cycles(agents)  # Should not raise

    def test_assert_no_cycles_raises_with_path(self) -> None:
        """assert_no_cycles should raise ValueError with cycle path."""
        agents: list[AgentConfig] = [
            LlmAgentConfig(name="a", model_profile="test", depends_on=["b"]),
            LlmAgentConfig(name="b", model_profile="test", depends_on=["a"]),
        ]
        with pytest.raises(ValueError, match="Circular dependency detected:"):
            assert_no_cycles(agents)

    def test_assert_no_cycles_error_message_contains_arrow_notation(self) -> None:
        """Error message should use 'x -> y -> x' arrow notation."""
        agents: list[AgentConfig] = [
            LlmAgentConfig(name="a", model_profile="test", depends_on=["b"]),
            LlmAgentConfig(name="b", model_profile="test", depends_on=["a"]),
        ]
        with pytest.raises(ValueError, match="->"):
            assert_no_cycles(agents)


class TestFindAgentByName:
    """Test _find_agent_by_name helper."""

    def test_find_agent_by_name_existing(self) -> None:
        """Test finding an existing agent by name."""
        from llm_orc.core.config.ensemble_config import _find_agent_by_name

        agents: list[AgentConfig] = [
            LlmAgentConfig(name="agent1", model_profile="model1"),
            LlmAgentConfig(name="agent2", model_profile="model2"),
            LlmAgentConfig(name="agent3", model_profile="model3"),
        ]

        result = _find_agent_by_name(agents, "agent2")

        assert result is not None
        assert result.name == "agent2"
        assert isinstance(result, LlmAgentConfig)
        assert result.model_profile == "model2"

    def test_find_agent_by_name_non_existing(self) -> None:
        """Test finding a non-existing agent by name."""
        from llm_orc.core.config.ensemble_config import _find_agent_by_name

        agents: list[AgentConfig] = [
            LlmAgentConfig(name="agent1", model_profile="model1"),
            LlmAgentConfig(name="agent2", model_profile="model2"),
        ]

        result = _find_agent_by_name(agents, "non_existing")

        assert result is None

    def test_find_agent_by_name_empty_list(self) -> None:
        """Test finding agent in empty list."""
        from llm_orc.core.config.ensemble_config import _find_agent_by_name

        agents: list[AgentConfig] = []

        result = _find_agent_by_name(agents, "any_agent")

        assert result is None


class TestFanOutValidation:
    """Test fan_out field validation for issue #73."""

    def test_validate_fan_out_requires_depends_on(self) -> None:
        """fan_out: true without depends_on should raise ValueError."""
        from llm_orc.core.config.ensemble_config import _validate_fan_out_dependencies

        agents: list[AgentConfig] = [
            ScriptAgentConfig(name="chunker", script="split.py"),
            LlmAgentConfig(
                name="extractor",
                model_profile="ollama-llama3",
                fan_out=True,
                # Missing depends_on - should fail
            ),
        ]

        with pytest.raises(ValueError, match="fan_out.*requires.*depends_on"):
            _validate_fan_out_dependencies(agents)

    def test_validate_fan_out_with_depends_on_valid(self) -> None:
        """fan_out: true with depends_on should pass validation."""
        from llm_orc.core.config.ensemble_config import _validate_fan_out_dependencies

        agents: list[AgentConfig] = [
            ScriptAgentConfig(name="chunker", script="split.py"),
            LlmAgentConfig(
                name="extractor",
                model_profile="ollama-llama3",
                fan_out=True,
                depends_on=["chunker"],
            ),
            LlmAgentConfig(
                name="synthesizer",
                model_profile="ollama-llama3",
                depends_on=["extractor"],
            ),
        ]

        # Should not raise any exception
        _validate_fan_out_dependencies(agents)

    def test_validate_fan_out_false_no_depends_on_valid(self) -> None:
        """fan_out: false or absent without depends_on should be valid."""
        from llm_orc.core.config.ensemble_config import _validate_fan_out_dependencies

        agents: list[AgentConfig] = [
            LlmAgentConfig(name="agent1", model_profile="test"),
            LlmAgentConfig(name="agent2", model_profile="test", fan_out=False),
        ]

        # Should not raise any exception
        _validate_fan_out_dependencies(agents)

    def test_validate_fan_out_empty_depends_on_invalid(self) -> None:
        """fan_out: true with empty depends_on should raise ValueError."""
        from llm_orc.core.config.ensemble_config import _validate_fan_out_dependencies

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="extractor",
                model_profile="ollama-llama3",
                fan_out=True,
                depends_on=[],  # Empty - should fail
            ),
        ]

        with pytest.raises(ValueError, match="fan_out.*requires.*depends_on"):
            _validate_fan_out_dependencies(agents)

    def test_loader_validates_fan_out_on_load(self) -> None:
        """EnsembleLoader should validate fan_out dependencies on load."""
        ensemble_yaml = {
            "name": "invalid_fan_out_ensemble",
            "description": "Ensemble with invalid fan_out config",
            "agents": [
                {"name": "chunker", "script": "split.py"},
                {
                    "name": "extractor",
                    "model_profile": "test-model",
                    "fan_out": True,
                    # Missing depends_on
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            loader = EnsembleLoader()
            with pytest.raises(ValueError, match="fan_out.*requires.*depends_on"):
                loader.load_from_file(yaml_path)
        finally:
            Path(yaml_path).unlink()


class TestCrossEnsembleCycleDetection:
    """Scenarios 8-9: Cross-ensemble cycle detection at load time."""

    def test_direct_cycle_detected(self, tmp_path: Path) -> None:
        """Ensemble A -> B -> A raises cycle error."""

        ensemble_a = {
            "name": "ensemble-a",
            "description": "Ensemble A",
            "agents": [
                {"name": "worker", "ensemble": "ensemble-b"},
            ],
        }
        ensemble_b = {
            "name": "ensemble-b",
            "description": "Ensemble B",
            "agents": [
                {"name": "worker", "ensemble": "ensemble-a"},
            ],
        }

        (tmp_path / "ensemble-a.yaml").write_text(yaml.dump(ensemble_a))
        (tmp_path / "ensemble-b.yaml").write_text(yaml.dump(ensemble_b))

        loader = EnsembleLoader()

        with pytest.raises(ValueError, match="cross-ensemble cycle"):
            loader.load_from_file(
                str(tmp_path / "ensemble-a.yaml"),
                search_dirs=[str(tmp_path)],
            )

    def test_transitive_cycle_detected(self, tmp_path: Path) -> None:
        """Ensemble A -> B -> C -> A raises cycle error."""

        ensemble_a = {
            "name": "ens-a",
            "description": "A",
            "agents": [
                {"name": "step", "ensemble": "ens-b"},
            ],
        }
        ensemble_b = {
            "name": "ens-b",
            "description": "B",
            "agents": [
                {"name": "step", "ensemble": "ens-c"},
            ],
        }
        ensemble_c = {
            "name": "ens-c",
            "description": "C",
            "agents": [
                {"name": "step", "ensemble": "ens-a"},
            ],
        }

        (tmp_path / "ens-a.yaml").write_text(yaml.dump(ensemble_a))
        (tmp_path / "ens-b.yaml").write_text(yaml.dump(ensemble_b))
        (tmp_path / "ens-c.yaml").write_text(yaml.dump(ensemble_c))

        loader = EnsembleLoader()

        with pytest.raises(ValueError, match="cross-ensemble cycle"):
            loader.load_from_file(
                str(tmp_path / "ens-a.yaml"),
                search_dirs=[str(tmp_path)],
            )

    def test_no_cycle_with_linear_references(self, tmp_path: Path) -> None:
        """A -> B (no cycle) loads without error."""
        ensemble_a = {
            "name": "ens-a",
            "description": "A",
            "agents": [
                {"name": "step", "ensemble": "ens-b"},
            ],
        }
        ensemble_b = {
            "name": "ens-b",
            "description": "B",
            "agents": [
                {
                    "name": "worker",
                    "script": "echo ok",
                },
            ],
        }

        (tmp_path / "ens-a.yaml").write_text(yaml.dump(ensemble_a))
        (tmp_path / "ens-b.yaml").write_text(yaml.dump(ensemble_b))

        loader = EnsembleLoader()
        config = loader.load_from_file(
            str(tmp_path / "ens-a.yaml"),
            search_dirs=[str(tmp_path)],
        )
        assert config.name == "ens-a"
