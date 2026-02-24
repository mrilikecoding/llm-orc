"""Tests for input_key selective upstream consumption (ADR-014)."""

import json
from typing import Any

from llm_orc.schemas.agent_config import (
    AgentConfig,
    EnsembleAgentConfig,
    LlmAgentConfig,
    ScriptAgentConfig,
)


class TestInputKeySelectsFromUpstream:
    """Scenario: Input key selects a specific key from upstream output."""

    def test_input_key_selects_pdfs(self) -> None:
        """Downstream agent receives only the selected key's value."""
        from llm_orc.core.execution.phases.dependency_resolver import (
            DependencyResolver,
        )

        resolver = DependencyResolver(lambda _: "Test Role")

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="pdf-processor",
                model_profile="gpt4",
                depends_on=["classifier"],
                input_key="pdfs",
            ),
        ]

        results_dict: dict[str, Any] = {
            "classifier": {
                "status": "success",
                "response": json.dumps(
                    {"pdfs": ["a.pdf", "b.pdf"], "audio": ["c.mp3"]}
                ),
            },
        }

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        # pdf-processor should receive only the pdfs array
        input_str = enhanced["pdf-processor"]
        assert "a.pdf" in input_str
        assert "b.pdf" in input_str
        # Should NOT contain audio data
        assert "c.mp3" not in input_str


class TestInputKeyWithFanOut:
    """Scenario: Input key with fan-out."""

    def test_input_key_selects_then_fan_out_expands(self) -> None:
        """input_key selects array, fan_out expands to N instances."""
        from llm_orc.core.execution.fan_out.coordinator import (
            FanOutCoordinator,
        )
        from llm_orc.core.execution.fan_out.expander import (
            FanOutExpander,
        )
        from llm_orc.core.execution.fan_out.gatherer import (
            FanOutGatherer,
        )

        expander = FanOutExpander()
        gatherer = FanOutGatherer(expander)
        coordinator = FanOutCoordinator(expander, gatherer)

        agent = LlmAgentConfig(
            name="pdf-processor",
            model_profile="gpt4",
            depends_on=["classifier"],
            fan_out=True,
            input_key="pdfs",
        )

        results_dict: dict[str, Any] = {
            "classifier": {
                "status": "success",
                "response": json.dumps(
                    {"pdfs": ["a.pdf", "b.pdf"], "audio": ["c.mp3"]}
                ),
            },
        }

        detected = coordinator.detect_in_phase([agent], results_dict)

        assert len(detected) == 1
        config, array = detected[0]
        assert array == ["a.pdf", "b.pdf"]


class TestMissingInputKeyIsRuntimeError:
    """Scenario: Missing input key is a runtime error."""

    def test_missing_key_produces_error(self) -> None:
        """Non-existent key in upstream output is a runtime error."""
        from llm_orc.core.execution.phases.dependency_resolver import (
            DependencyResolver,
        )

        resolver = DependencyResolver(lambda _: "Test Role")

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="video-processor",
                model_profile="gpt4",
                depends_on=["classifier"],
                input_key="videos",
            ),
        ]

        results_dict: dict[str, Any] = {
            "classifier": {
                "status": "success",
                "response": json.dumps({"pdfs": ["a.pdf"]}),
            },
        }

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        # Should contain error information about missing key
        input_str = enhanced["video-processor"]
        assert "videos" in input_str
        assert "not found" in input_str.lower() or "error" in input_str.lower()


class TestNonDictUpstreamWithInputKey:
    """Scenario: Non-dict upstream output with input key is error."""

    def test_plain_string_upstream_produces_error(self) -> None:
        """input_key on non-dict upstream is a runtime error."""
        from llm_orc.core.execution.phases.dependency_resolver import (
            DependencyResolver,
        )

        resolver = DependencyResolver(lambda _: "Test Role")

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="processor",
                model_profile="gpt4",
                depends_on=["greeter"],
                input_key="message",
            ),
        ]

        results_dict: dict[str, Any] = {
            "greeter": {
                "status": "success",
                "response": "hello world",
            },
        }

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        input_str = enhanced["processor"]
        assert "not dict" in input_str.lower() or "error" in input_str.lower()


class TestNoInputKeyBackwardCompatible:
    """Scenario: No input key passes full output."""

    def test_no_input_key_passes_full_output(self) -> None:
        """Without input_key, agent receives full upstream output."""
        from llm_orc.core.execution.phases.dependency_resolver import (
            DependencyResolver,
        )

        resolver = DependencyResolver(lambda _: "Test Role")

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="summarizer",
                model_profile="gpt4",
                depends_on=["classifier"],
            ),
        ]

        results_dict: dict[str, Any] = {
            "classifier": {
                "status": "success",
                "response": json.dumps({"pdfs": ["a.pdf"], "audio": ["c.mp3"]}),
            },
        }

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        input_str = enhanced["summarizer"]
        # Should contain both keys
        assert "pdfs" in input_str
        assert "audio" in input_str


class TestInputKeyWorksWithAllAgentTypes:
    """Scenario: Input key works with all agent types."""

    def test_input_key_on_all_types(self) -> None:
        """input_key is available on LLM, Script, and Ensemble configs."""
        llm = LlmAgentConfig(name="llm", model_profile="gpt4", input_key="text")
        script = ScriptAgentConfig(name="script", script="run.py", input_key="data")
        ensemble = EnsembleAgentConfig(name="ens", ensemble="child", input_key="items")

        assert llm.input_key == "text"
        assert script.input_key == "data"
        assert ensemble.input_key == "items"


class TestIntegrationInputKeyRoutingPattern:
    """Scenario: Integration — input key with ensemble agent routing."""

    def test_routing_pattern_composes_correctly(self) -> None:
        """classifier → pdf-extractor(input_key=pdfs, fan_out)
        + audio-extractor(input_key=audio, fan_out) → synthesizer.

        Verifies DependencyAnalyzer, FanOutCoordinator, and
        DependencyResolver compose with input_key using real components.
        """
        from llm_orc.core.execution.fan_out.coordinator import (
            FanOutCoordinator,
        )
        from llm_orc.core.execution.fan_out.expander import (
            FanOutExpander,
        )
        from llm_orc.core.execution.fan_out.gatherer import (
            FanOutGatherer,
        )
        from llm_orc.core.execution.phases.dependency_analyzer import (
            DependencyAnalyzer,
        )
        from llm_orc.core.execution.phases.dependency_resolver import (
            DependencyResolver,
        )

        # Build the routing ensemble
        agents: list[AgentConfig] = [
            ScriptAgentConfig(
                name="classifier",
                script="classify.py",
            ),
            EnsembleAgentConfig(
                name="pdf-extractor",
                ensemble="pdf-pipeline",
                depends_on=["classifier"],
                input_key="pdfs",
                fan_out=True,
            ),
            EnsembleAgentConfig(
                name="audio-extractor",
                ensemble="audio-pipeline",
                depends_on=["classifier"],
                input_key="audio",
                fan_out=True,
            ),
            LlmAgentConfig(
                name="synthesizer",
                model_profile="gpt4",
                depends_on=["pdf-extractor", "audio-extractor"],
            ),
        ]

        # Phase ordering via real DependencyAnalyzer
        analyzer = DependencyAnalyzer()
        result = analyzer.analyze_enhanced_dependency_graph(agents)
        phase_names = [[a.name for a in phase] for phase in result["phases"]]

        assert phase_names[0] == ["classifier"]
        assert set(phase_names[1]) == {
            "pdf-extractor",
            "audio-extractor",
        }
        assert phase_names[2] == ["synthesizer"]

        # Simulate classifier output
        classifier_output: dict[str, Any] = {
            "classifier": {
                "status": "success",
                "response": json.dumps({"pdfs": ["a.pdf"], "audio": ["c.mp3"]}),
            },
        }

        # Fan-out detection via real FanOutCoordinator
        expander = FanOutExpander()
        gatherer = FanOutGatherer(expander)
        coordinator = FanOutCoordinator(expander, gatherer)

        phase_2_agents = result["phases"][1]
        detected = coordinator.detect_in_phase(phase_2_agents, classifier_output)

        # Both extractors should detect their respective arrays
        detected_map = {cfg.name: arr for cfg, arr in detected}
        assert detected_map["pdf-extractor"] == ["a.pdf"]
        assert detected_map["audio-extractor"] == ["c.mp3"]

        # Simulate extractor results for synthesizer input
        extractor_results: dict[str, Any] = {
            **classifier_output,
            "pdf-extractor": {
                "status": "success",
                "response": json.dumps({"text": "PDF content"}),
            },
            "audio-extractor": {
                "status": "success",
                "response": json.dumps({"transcript": "Audio transcript"}),
            },
        }

        # DependencyResolver enhances synthesizer input
        resolver = DependencyResolver(lambda _: "Test Role")
        enhanced = resolver.enhance_input_with_dependencies(
            "Synthesize all results",
            [agents[3]],  # synthesizer
            extractor_results,
        )

        synth_input = enhanced["synthesizer"]
        assert "PDF content" in synth_input
        assert "Audio transcript" in synth_input
