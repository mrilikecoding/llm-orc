"""Tests for ensemble configuration loading."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.core.config.ensemble_config import (
    EnsembleConfig,
    EnsembleLoader,
    assert_no_cycles,
    detect_cycle,
    validate_ensemble_reference_graph,
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

    def test_raw_output_defaults_to_false(self) -> None:
        """ADR-004 contract: summarization is the default; raw_output is opt-in."""
        config = EnsembleConfig(name="any", description="any")

        assert config.raw_output is False

    def test_raw_output_is_explicitly_settable(self) -> None:
        """Escape hatch is honored when the author flags it on construction."""
        config = EnsembleConfig(name="classifier", description="any", raw_output=True)

        assert config.raw_output is True


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

    def test_load_ensemble_reads_raw_output_flag_when_present(self) -> None:
        """ADR-004: a classifier ensemble that declares raw_output: true is loaded
        with the flag set, so Tool Dispatch can skip the Result Summarizer."""
        ensemble_yaml = {
            "name": "intent_classifier",
            "description": "Returns a structured intent label",
            "raw_output": True,
            "agents": [
                {"name": "classifier", "model": "haiku", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.raw_output is True
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_defaults_raw_output_to_false_when_absent(self) -> None:
        """Backward compat: existing ensembles without raw_output keep summarization."""
        ensemble_yaml = {
            "name": "existing_ensemble",
            "description": "Pre-WP-D ensemble without the raw_output key",
            "agents": [
                {"name": "agent", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.raw_output is False
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_reads_output_schema_when_present(self) -> None:
        """ADR-024 (Cycle 6 WP-D): output_schema declared in YAML becomes
        the loaded config's output_schema dict. The schema is opaque to
        the loader — it's a JSON-Schema-shaped dict the dispatch site
        consults to advisorily populate envelope.structured."""
        schema = {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["text", "label"],
                        "properties": {
                            "text": {"type": "string"},
                            "label": {
                                "type": "string",
                                "enum": ["established", "contested"],
                            },
                        },
                    },
                }
            },
        }
        ensemble_yaml = {
            "name": "claim_extractor",
            "description": "Extracts factual claims",
            "output_schema": schema,
            "agents": [
                {"name": "extractor", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.output_schema == schema
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_defaults_output_schema_to_none_when_absent(self) -> None:
        """Backward compat: pre-WP-D ensembles without output_schema load
        with the field set to None; downstream code reads it as 'no
        schema declared' and leaves envelope.structured unpopulated."""
        ensemble_yaml = {
            "name": "no_schema_ensemble",
            "description": "Pre-WP-D ensemble",
            "agents": [
                {"name": "agent", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.output_schema is None
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_ignores_non_dict_output_schema_values(self) -> None:
        """A malformed output_schema entry (string, list, scalar) loads
        as None rather than raising. The loader is tolerant; schema
        validity is the dispatch site's advisory concern, not the
        loader's structural enforcement."""
        ensemble_yaml = {
            "name": "malformed_schema_ensemble",
            "description": "Test malformed output_schema",
            "output_schema": "this is not a dict",
            "agents": [
                {"name": "agent", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.output_schema is None
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_reads_output_substrate_when_present(self) -> None:
        """ADR-025 (Cycle 6 WP-E): output_substrate declared in YAML
        becomes the loaded config's output_substrate string. The loader
        accepts either ``artifact`` or ``inline`` verbatim; downstream
        dispatch reads this to route the deliverable through
        SessionArtifactStore vs. the inline-response path."""
        ensemble_yaml = {
            "name": "code_gen",
            "description": "Capability ensemble — substrate-routed",
            "output_substrate": "artifact",
            "agents": [
                {"name": "coder", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.output_substrate == "artifact"
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_defaults_output_substrate_to_none_when_absent(self) -> None:
        """Backward compat: pre-WP-E ensembles without output_substrate
        load with the field set to None. ADR-025's "default per ensemble
        category" defaulting (capability ensembles → artifact; system
        ensembles → inline) is the dispatch-site's concern when the YAML
        declaration is absent — the loader stays a faithful YAML→dataclass
        translator without baking in category judgments."""
        ensemble_yaml = {
            "name": "no_substrate_ensemble",
            "description": "Pre-WP-E ensemble",
            "agents": [
                {"name": "agent", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.output_substrate is None
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_ignores_invalid_output_substrate_values(self) -> None:
        """An output_substrate value outside the closed set
        (``artifact``, ``inline``) loads as None rather than raising.
        Operators using a typo or future-extension keyword get the
        documented default behavior; dispatch logs an advisory when the
        absence shows up at a substrate-routing decision point."""
        ensemble_yaml = {
            "name": "typo_substrate_ensemble",
            "description": "Test typo output_substrate",
            "output_substrate": "artefact",  # British spelling — typo
            "agents": [
                {"name": "agent", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.output_substrate is None
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_reads_output_retention_when_present(self) -> None:
        """ADR-025 (Cycle 6 WP-E): output_retention declared in YAML
        becomes the loaded config's output_retention string. Accepted
        values are ``session`` / ``durable`` / ``ephemeral`` per ADR-025
        §"Retention semantics"; the loader translates verbatim."""
        ensemble_yaml = {
            "name": "durable_ensemble",
            "description": "Capability ensemble whose deliverables persist",
            "output_substrate": "artifact",
            "output_retention": "durable",
            "agents": [
                {"name": "agent", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.output_retention == "durable"
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_defaults_output_retention_to_none_when_absent(self) -> None:
        """Backward compat + default-at-dispatch posture: pre-WP-E
        ensembles load with output_retention = None; the dispatch path
        applies the documented default (``session`` for substrate-routed
        ensembles per ADR-025)."""
        ensemble_yaml = {
            "name": "no_retention_ensemble",
            "description": "Pre-WP-E ensemble",
            "agents": [
                {"name": "agent", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.output_retention is None
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_ignores_invalid_output_retention_values(self) -> None:
        """An output_retention value outside ``session`` / ``durable`` /
        ``ephemeral`` loads as None rather than raising. Same tolerant
        posture as output_substrate — typos and future keywords surface
        as ``defaults applied`` at dispatch time, not loader crashes."""
        ensemble_yaml = {
            "name": "typo_retention_ensemble",
            "description": "Test typo output_retention",
            "output_substrate": "artifact",
            "output_retention": "forever",  # not a valid retention literal
            "agents": [
                {"name": "agent", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.output_retention is None
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_reads_calibration_substrate_access_when_present(
        self,
    ) -> None:
        """ADR-025 §"Calibration-gate evaluation surface" (Cycle 6 WP-E):
        calibration_substrate_access declared in YAML becomes the loaded
        config's field. ``artifact`` opts the ensemble into critic-agent
        file-read of the deliverable; the default (absent or ``summary``)
        keeps critics evaluating envelope.primary + artifacts[0].summary
        only."""
        ensemble_yaml = {
            "name": "code_generator",
            "description": "Code generation — calibration reads artifact content",
            "output_substrate": "artifact",
            "calibration_substrate_access": "artifact",
            "agents": [
                {"name": "coder", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.calibration_substrate_access == "artifact"
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_defaults_calibration_substrate_access_to_none(
        self,
    ) -> None:
        """Backward compat: pre-WP-E ensembles without
        calibration_substrate_access load with the field set to None; the
        Calibration Gate applies summary-only evaluation as documented."""
        ensemble_yaml = {
            "name": "no_calibration_access_ensemble",
            "description": "Pre-WP-E ensemble",
            "agents": [
                {"name": "agent", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.calibration_substrate_access is None
        finally:
            Path(yaml_path).unlink()

    def test_load_ensemble_ignores_invalid_calibration_substrate_access(
        self,
    ) -> None:
        """A calibration_substrate_access value outside ``summary`` /
        ``artifact`` loads as None rather than raising; the gate applies
        summary-only evaluation per the documented default. Mirrors the
        tolerant load posture used for output_substrate / output_retention.
        """
        ensemble_yaml = {
            "name": "typo_calibration_access_ensemble",
            "description": "Test typo calibration_substrate_access",
            "output_substrate": "artifact",
            "calibration_substrate_access": "summery",  # typo
            "agents": [
                {"name": "agent", "model": "sonnet", "provider": "anthropic"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            config = EnsembleLoader().load_from_file(yaml_path)
            assert config.calibration_substrate_access is None
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

    def test_list_ensembles_logs_warning_for_invalid_files(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Invalid ensemble files should log a warning, not fail silently."""
        import logging

        with tempfile.TemporaryDirectory() as temp_dir:
            # Ensemble with an extra field that extra="forbid" rejects
            bad_config = {
                "name": "bad_ensemble",
                "description": "Has invalid agent field",
                "agents": [
                    {
                        "name": "agent1",
                        "model_profile": "test",
                        "synthesis_timeout_seconds": 90,
                    }
                ],
            }
            bad_file = Path(temp_dir) / "bad.yaml"
            with open(bad_file, "w") as f:
                yaml.dump(bad_config, f)

            loader = EnsembleLoader()
            with caplog.at_level(logging.WARNING):
                result = loader.list_ensembles(temp_dir)

            assert result == []
            assert any("bad.yaml" in rec.message for rec in caplog.records)

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


class TestValidateEnsembleReferenceGraph:
    """Public validator exposed for composition-time and load-time callers.

    WP-A / scenarios.md §Structural Debt Remediation: the cross-ensemble
    cycle validator must be a public function callable without loading, so
    the MCP/web validate path and the future compose_ensemble path share a
    single routine with the loader.
    """

    def _write(self, tmp_path: Path, specs: dict[str, dict[str, Any]]) -> None:
        for spec_name, spec in specs.items():
            (tmp_path / f"{spec_name}.yaml").write_text(yaml.dump(spec))

    def test_raises_on_direct_cycle(self, tmp_path: Path) -> None:
        """A -> B -> A is reported as a cross-ensemble cycle."""
        self._write(
            tmp_path,
            {
                "ens-a": {
                    "name": "ens-a",
                    "description": "A",
                    "agents": [{"name": "step", "ensemble": "ens-b"}],
                },
                "ens-b": {
                    "name": "ens-b",
                    "description": "B",
                    "agents": [{"name": "step", "ensemble": "ens-a"}],
                },
            },
        )
        agents = EnsembleLoader().load_from_file(str(tmp_path / "ens-a.yaml")).agents

        with pytest.raises(ValueError, match="cross-ensemble cycle"):
            validate_ensemble_reference_graph("ens-a", agents, [str(tmp_path)])

    def test_raises_on_transitive_cycle(self, tmp_path: Path) -> None:
        """A -> B -> C -> A is reported as a cross-ensemble cycle."""
        self._write(
            tmp_path,
            {
                "ens-a": {
                    "name": "ens-a",
                    "description": "A",
                    "agents": [{"name": "step", "ensemble": "ens-b"}],
                },
                "ens-b": {
                    "name": "ens-b",
                    "description": "B",
                    "agents": [{"name": "step", "ensemble": "ens-c"}],
                },
                "ens-c": {
                    "name": "ens-c",
                    "description": "C",
                    "agents": [{"name": "step", "ensemble": "ens-a"}],
                },
            },
        )
        agents = EnsembleLoader().load_from_file(str(tmp_path / "ens-a.yaml")).agents

        with pytest.raises(ValueError, match="cross-ensemble cycle"):
            validate_ensemble_reference_graph("ens-a", agents, [str(tmp_path)])

    def test_returns_none_on_acyclic_graph(self, tmp_path: Path) -> None:
        """A -> B with B a leaf is acyclic — function returns None."""
        self._write(
            tmp_path,
            {
                "ens-a": {
                    "name": "ens-a",
                    "description": "A",
                    "agents": [{"name": "step", "ensemble": "ens-b"}],
                },
                "ens-b": {
                    "name": "ens-b",
                    "description": "B",
                    "agents": [{"name": "worker", "script": "echo ok"}],
                },
            },
        )
        agents = EnsembleLoader().load_from_file(str(tmp_path / "ens-a.yaml")).agents

        # Does not raise — acyclic graph
        validate_ensemble_reference_graph("ens-a", agents, [str(tmp_path)])

    def test_returns_none_when_no_ensemble_references(self, tmp_path: Path) -> None:
        """An ensemble with only script/llm agents has no references to check."""
        self._write(
            tmp_path,
            {
                "solo": {
                    "name": "solo",
                    "description": "standalone",
                    "agents": [{"name": "worker", "script": "echo ok"}],
                },
            },
        )
        agents = EnsembleLoader().load_from_file(str(tmp_path / "solo.yaml")).agents

        # Does not raise — no references means nothing to check
        validate_ensemble_reference_graph("solo", agents, [str(tmp_path)])


class TestListEnsemblesFiresCycleCheck:
    """WP-A scenario refactor 2: list_ensembles passes search_dirs so the
    cross-ensemble cycle check actually fires during directory listing.
    """

    def test_cyclic_pair_is_skipped_from_listing(self, tmp_path: Path) -> None:
        """Two cyclic ensembles in a directory are excluded from list_ensembles.

        Previously list_ensembles called load_from_file with no search_dirs,
        silently skipping cross-ensemble cycle detection. After the fix,
        the cycle is detected per-file and the invalid ensembles are
        logged-and-skipped inside list_ensembles' existing try/except.
        """
        (tmp_path / "cyc-a.yaml").write_text(
            yaml.dump(
                {
                    "name": "cyc-a",
                    "description": "A",
                    "agents": [{"name": "step", "ensemble": "cyc-b"}],
                }
            )
        )
        (tmp_path / "cyc-b.yaml").write_text(
            yaml.dump(
                {
                    "name": "cyc-b",
                    "description": "B",
                    "agents": [{"name": "step", "ensemble": "cyc-a"}],
                }
            )
        )

        loader = EnsembleLoader()
        result = loader.list_ensembles(str(tmp_path))

        assert result == []

    def test_acyclic_ensembles_still_listed(self, tmp_path: Path) -> None:
        """Passing search_dirs to list_ensembles does not break acyclic cases."""
        (tmp_path / "ok-a.yaml").write_text(
            yaml.dump(
                {
                    "name": "ok-a",
                    "description": "A",
                    "agents": [{"name": "step", "ensemble": "ok-b"}],
                }
            )
        )
        (tmp_path / "ok-b.yaml").write_text(
            yaml.dump(
                {
                    "name": "ok-b",
                    "description": "B",
                    "agents": [{"name": "worker", "script": "echo ok"}],
                }
            )
        )

        loader = EnsembleLoader()
        names = sorted(cfg.name for cfg in loader.list_ensembles(str(tmp_path)))

        assert names == ["ok-a", "ok-b"]


class TestSharedRoutineRegression:
    """scenarios.md regression: composition-time and load-time validator
    share a single routine. This test mechanically forbids split
    implementations by exercising both paths on the same input and
    asserting identical outcomes.
    """

    def test_load_path_and_public_function_agree_on_cycle(self, tmp_path: Path) -> None:
        """Same cyclic input → same ValueError via either entry point."""
        (tmp_path / "r-a.yaml").write_text(
            yaml.dump(
                {
                    "name": "r-a",
                    "description": "A",
                    "agents": [{"name": "step", "ensemble": "r-b"}],
                }
            )
        )
        (tmp_path / "r-b.yaml").write_text(
            yaml.dump(
                {
                    "name": "r-b",
                    "description": "B",
                    "agents": [{"name": "step", "ensemble": "r-a"}],
                }
            )
        )

        loader = EnsembleLoader()
        agents = loader.load_from_file(str(tmp_path / "r-a.yaml")).agents

        with pytest.raises(ValueError, match="cross-ensemble cycle") as load_exc:
            loader.load_from_file(
                str(tmp_path / "r-a.yaml"),
                search_dirs=[str(tmp_path)],
            )

        with pytest.raises(ValueError, match="cross-ensemble cycle") as pub_exc:
            validate_ensemble_reference_graph("r-a", agents, [str(tmp_path)])

        assert str(load_exc.value) == str(pub_exc.value)

    def test_load_path_and_public_function_agree_on_acyclic(
        self, tmp_path: Path
    ) -> None:
        """Same acyclic input → both paths succeed (no divergence)."""
        (tmp_path / "ok.yaml").write_text(
            yaml.dump(
                {
                    "name": "ok",
                    "description": "ok",
                    "agents": [{"name": "worker", "script": "echo ok"}],
                }
            )
        )

        loader = EnsembleLoader()
        agents = loader.load_from_file(str(tmp_path / "ok.yaml")).agents

        loader.load_from_file(
            str(tmp_path / "ok.yaml"),
            search_dirs=[str(tmp_path)],
        )
        validate_ensemble_reference_graph("ok", agents, [str(tmp_path)])


class TestEnsembleLoaderValidateOnceAtLoad:
    """Cycle 6 WP-B piece 3 — validate-once-at-load library cache.

    Per ``docs/agentic-serving/scenarios.md`` §Observability Event Routing
    scenario "Validate-once-at-load eliminates per-enumeration noise" and
    `system-design.agents.md` §Ensemble Engine §Cycle 6 extensions.
    Backward-compat fitness: un-primed callers (CLI, MCP) keep the
    existing per-call on-demand validation path with its current
    `logger.warning` emission.
    """

    @staticmethod
    def _write_valid_ensemble(directory: Path, name: str) -> Path:
        path = directory / f"{name}.yaml"
        path.write_text(
            yaml.dump(
                {
                    "name": name,
                    "description": f"{name} description",
                    "agents": [{"name": "a1", "script": "echo hi"}],
                }
            )
        )
        return path

    @staticmethod
    def _write_invalid_ensemble(directory: Path, name: str) -> Path:
        """Write a YAML the schema rejects (`extra='forbid'` on agents)."""
        path = directory / f"{name}.yaml"
        path.write_text(
            yaml.dump(
                {
                    "name": name,
                    "description": f"{name} description",
                    "agents": [
                        {
                            "name": "a1",
                            "model_profile": "test",
                            "synthesis_timeout_seconds": 90,
                        }
                    ],
                }
            )
        )
        return path

    def test_validation_results_empty_before_prime(self, tmp_path: Path) -> None:
        """A fresh loader has no validation results until prime runs."""
        loader = EnsembleLoader()

        assert loader.validation_results() == ()

    def test_prime_caches_validated_ensembles_for_list_lookups(
        self, tmp_path: Path
    ) -> None:
        """Prime walks the directory once; list_ensembles returns the cache."""
        self._write_valid_ensemble(tmp_path, "alpha")
        self._write_valid_ensemble(tmp_path, "beta")

        loader = EnsembleLoader()
        loader.prime(str(tmp_path))

        cached = loader.list_ensembles(str(tmp_path))
        names = sorted(e.name for e in cached)
        assert names == ["alpha", "beta"]

    def test_prime_records_invalid_yamls_in_validation_results(
        self, tmp_path: Path
    ) -> None:
        """Invalid YAMLs flow to ``validation_results()``; the cache only
        carries the valid subset."""
        self._write_valid_ensemble(tmp_path, "good")
        bad_path = self._write_invalid_ensemble(tmp_path, "bad")

        loader = EnsembleLoader()
        loader.prime(str(tmp_path))

        cached = loader.list_ensembles(str(tmp_path))
        assert [e.name for e in cached] == ["good"]

        results = loader.validation_results()
        assert len(results) == 1
        (result,) = results
        assert result.yaml_path == str(bad_path)
        assert result.error  # non-empty rationale

    def test_prime_path_is_silent_at_the_loader_logger(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Priming records failures in validation_results; the loader does
        not emit its own ``logger.warning`` lines on the prime path —
        emission is delegated to the operator-terminal sink."""
        import logging as logging_module

        self._write_invalid_ensemble(tmp_path, "bad")
        loader = EnsembleLoader()

        with caplog.at_level(
            logging_module.WARNING, logger="llm_orc.core.config.ensemble_config"
        ):
            loader.prime(str(tmp_path))

        loader_warnings = [
            rec
            for rec in caplog.records
            if rec.name == "llm_orc.core.config.ensemble_config"
            and rec.levelno >= logging_module.WARNING
        ]
        assert loader_warnings == []

    def test_list_ensembles_after_prime_does_not_re_emit_warnings(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Eight enumerations after prime produce zero additional WARNs."""
        import logging as logging_module

        self._write_valid_ensemble(tmp_path, "good")
        self._write_invalid_ensemble(tmp_path, "bad")

        loader = EnsembleLoader()
        loader.prime(str(tmp_path))

        with caplog.at_level(
            logging_module.WARNING, logger="llm_orc.core.config.ensemble_config"
        ):
            for _ in range(8):
                loader.list_ensembles(str(tmp_path))

        loader_warnings = [
            rec
            for rec in caplog.records
            if rec.name == "llm_orc.core.config.ensemble_config"
            and rec.levelno >= logging_module.WARNING
        ]
        assert loader_warnings == []

    def test_reload_replaces_prior_cached_state_for_directory(
        self, tmp_path: Path
    ) -> None:
        """Reload re-walks the directory; a newly-added ensemble surfaces."""
        self._write_valid_ensemble(tmp_path, "alpha")

        loader = EnsembleLoader()
        loader.prime(str(tmp_path))
        assert {e.name for e in loader.list_ensembles(str(tmp_path))} == {"alpha"}

        self._write_valid_ensemble(tmp_path, "beta")
        # Without reload, the cache still reflects only alpha.
        assert {e.name for e in loader.list_ensembles(str(tmp_path))} == {"alpha"}

        loader.reload(str(tmp_path))
        assert {e.name for e in loader.list_ensembles(str(tmp_path))} == {
            "alpha",
            "beta",
        }

    def test_reload_clears_stale_validation_results_for_directory(
        self, tmp_path: Path
    ) -> None:
        """When a previously invalid YAML is fixed, reload removes its entry."""
        bad_path = self._write_invalid_ensemble(tmp_path, "drift")

        loader = EnsembleLoader()
        loader.prime(str(tmp_path))
        assert len(loader.validation_results()) == 1

        bad_path.unlink()
        self._write_valid_ensemble(tmp_path, "drift")
        loader.reload(str(tmp_path))

        assert loader.validation_results() == ()
        assert {e.name for e in loader.list_ensembles(str(tmp_path))} == {"drift"}

    def test_unprimed_list_ensembles_keeps_on_demand_logging(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Backward-compat: a loader that was never primed continues to walk
        the directory on demand and emits ``logger.warning`` per failure —
        CLI and MCP callers depend on this surface."""
        import logging as logging_module

        self._write_valid_ensemble(tmp_path, "good")
        self._write_invalid_ensemble(tmp_path, "bad")

        loader = EnsembleLoader()

        with caplog.at_level(
            logging_module.WARNING, logger="llm_orc.core.config.ensemble_config"
        ):
            result = loader.list_ensembles(str(tmp_path))

        assert [e.name for e in result] == ["good"]
        loader_warnings = [
            rec
            for rec in caplog.records
            if rec.name == "llm_orc.core.config.ensemble_config"
            and rec.levelno >= logging_module.WARNING
            and "bad.yaml" in rec.message
        ]
        assert len(loader_warnings) == 1

    def test_prime_accepts_nonexistent_directory_without_raising(
        self, tmp_path: Path
    ) -> None:
        """Priming a missing directory is a no-op — the orchestrator's
        startup-prime path should not crash on operators with a partial
        deployment shape."""
        loader = EnsembleLoader()
        missing = tmp_path / "does-not-exist"

        loader.prime(str(missing))

        assert loader.list_ensembles(str(missing)) == []
        assert loader.validation_results() == ()

    def test_prime_two_directories_keeps_results_per_directory(
        self, tmp_path: Path
    ) -> None:
        """``OrchestraService`` walks local / global / library dirs; the
        cache and validation_results must compose across primed dirs."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        self._write_valid_ensemble(dir_a, "alpha")
        self._write_invalid_ensemble(dir_a, "bad_a")
        self._write_valid_ensemble(dir_b, "beta")

        loader = EnsembleLoader()
        loader.prime(str(dir_a))
        loader.prime(str(dir_b))

        assert {e.name for e in loader.list_ensembles(str(dir_a))} == {"alpha"}
        assert {e.name for e in loader.list_ensembles(str(dir_b))} == {"beta"}

        # One failure spans both primed directories — the b directory has none.
        results = loader.validation_results()
        assert len(results) == 1
        assert "bad_a.yaml" in results[0].yaml_path


class TestLoopAndDispatchReferenceCoverage:
    """Issue #94: AS-2 must cover loop bodies and static dispatch targets."""

    def _write(self, tmp_path: Path, specs: dict[str, dict[str, Any]]) -> None:
        for spec_name, spec in specs.items():
            (tmp_path / f"{spec_name}.yaml").write_text(yaml.dump(spec))

    def test_raises_on_loop_body_cycle(self, tmp_path: Path) -> None:
        """A loop whose body re-enters its parent is rejected at load, not
        left to burn nested executions until the runtime depth limit."""
        self._write(
            tmp_path,
            {
                "ens-a": {
                    "name": "ens-a",
                    "description": "A",
                    "agents": [
                        {
                            "name": "round",
                            "loop": {
                                "body": "ens-a",
                                "until": "${done}",
                                "max_iterations": 2,
                            },
                        }
                    ],
                },
            },
        )
        agents = EnsembleLoader().load_from_file(str(tmp_path / "ens-a.yaml")).agents

        with pytest.raises(ValueError, match="cross-ensemble cycle"):
            validate_ensemble_reference_graph("ens-a", agents, [str(tmp_path)])

    def test_raises_on_static_dispatch_cycle(self, tmp_path: Path) -> None:
        """A literal (non-templated) dispatch target participates in the
        reference graph; runtime-resolved ${...} targets cannot."""
        self._write(
            tmp_path,
            {
                "ens-a": {
                    "name": "ens-a",
                    "description": "A",
                    "agents": [{"name": "seat", "dispatch": "ens-b"}],
                },
                "ens-b": {
                    "name": "ens-b",
                    "description": "B",
                    "agents": [{"name": "step", "ensemble": "ens-a"}],
                },
            },
        )
        agents = EnsembleLoader().load_from_file(str(tmp_path / "ens-a.yaml")).agents

        with pytest.raises(ValueError, match="cross-ensemble cycle"):
            validate_ensemble_reference_graph("ens-a", agents, [str(tmp_path)])

    def test_templated_dispatch_is_not_statically_followed(
        self, tmp_path: Path
    ) -> None:
        self._write(
            tmp_path,
            {
                "ens-a": {
                    "name": "ens-a",
                    "description": "A",
                    "agents": [
                        {"name": "pick", "script": "echo x"},
                        {
                            "name": "seat",
                            "dispatch": "${pick.target}",
                            "depends_on": ["pick"],
                        },
                    ],
                },
            },
        )
        agents = EnsembleLoader().load_from_file(str(tmp_path / "ens-a.yaml")).agents

        validate_ensemble_reference_graph("ens-a", agents, [str(tmp_path)])
