"""Unit tests for progress controller interface and implementations."""

from unittest.mock import Mock

import pytest

from llm_orc.core.execution.progress_controller import (
    NoOpProgressController,
    ProgressController,
)


class TestProgressControllerInterface:
    """Test the abstract ProgressController interface."""

    def test_progress_controller_is_abstract(self) -> None:
        """Test that ProgressController cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ProgressController()  # type: ignore

    def test_progress_controller_requires_pause_implementation(self) -> None:
        """Test that subclasses must implement pause_for_user_input."""

        class IncompleteController(ProgressController):
            def resume_from_user_input(self, agent_name: str) -> None:
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteController()  # type: ignore

    def test_progress_controller_requires_resume_implementation(self) -> None:
        """Test that subclasses must implement resume_from_user_input."""

        class IncompleteController(ProgressController):
            def pause_for_user_input(self, agent_name: str, prompt: str = "") -> None:
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteController()  # type: ignore

    async def test_progress_controller_complete_implementation(self) -> None:
        """Test that a complete implementation can be instantiated."""

        class CompleteController(ProgressController):
            def pause_for_user_input(self, agent_name: str, prompt: str = "") -> None:
                pass

            def resume_from_user_input(self, agent_name: str) -> None:
                pass

            async def start_ensemble(self, ensemble_name: str) -> None:
                pass

            async def update_agent_progress(self, agent_name: str, status: str) -> None:
                pass

            async def complete_ensemble(self) -> None:
                pass

        # Should not raise
        controller = CompleteController()
        assert isinstance(controller, ProgressController)

        # Exercise all methods to ensure they work
        controller.pause_for_user_input("test", "prompt")
        controller.resume_from_user_input("test")
        await controller.start_ensemble("ensemble")
        await controller.update_agent_progress("agent", "status")
        await controller.complete_ensemble()


class TestNoOpProgressController:
    """Test the NoOpProgressController implementation."""

    def test_noop_controller_instantiation(self) -> None:
        """Test that NoOpProgressController can be instantiated."""
        controller = NoOpProgressController()
        assert isinstance(controller, ProgressController)
        assert isinstance(controller, NoOpProgressController)

    def test_noop_pause_for_user_input(self) -> None:
        """Test that pause_for_user_input does nothing."""
        controller = NoOpProgressController()

        # Should not raise any exceptions
        controller.pause_for_user_input("test_agent")
        controller.pause_for_user_input("test_agent", "Enter value: ")
        controller.pause_for_user_input("", "")

    def test_noop_resume_from_user_input(self) -> None:
        """Test that resume_from_user_input does nothing."""
        controller = NoOpProgressController()

        # Should not raise any exceptions
        controller.resume_from_user_input("test_agent")
        controller.resume_from_user_input("")

    def test_noop_controller_workflow(self) -> None:
        """Test a complete pause/resume workflow with NoOp controller."""
        controller = NoOpProgressController()

        # Simulate a workflow
        agents = ["agent1", "agent2", "agent3"]
        prompts = ["Enter name: ", "Enter age: ", "Enter location: "]

        for agent, prompt in zip(agents, prompts, strict=False):
            controller.pause_for_user_input(agent, prompt)
            # In real usage, user input would happen here
            controller.resume_from_user_input(agent)

        # Should complete without any side effects

    def test_noop_controller_with_mock_tracking(self) -> None:
        """Test NoOp controller behavior with method call tracking."""
        controller = NoOpProgressController()

        # Mock the methods to track calls
        controller.pause_for_user_input = Mock(  # type: ignore
            side_effect=controller.pause_for_user_input
        )
        controller.resume_from_user_input = Mock(  # type: ignore
            side_effect=controller.resume_from_user_input
        )

        # Execute operations
        controller.pause_for_user_input("agent1", "Prompt 1")
        controller.resume_from_user_input("agent1")
        controller.pause_for_user_input("agent2", "Prompt 2")
        controller.resume_from_user_input("agent2")

        # Verify calls were made
        assert controller.pause_for_user_input.call_count == 2
        assert controller.resume_from_user_input.call_count == 2

    async def test_noop_start_ensemble(self) -> None:
        """Test that start_ensemble does nothing."""
        controller = NoOpProgressController()

        # Should not raise any exceptions
        await controller.start_ensemble("test_ensemble")
        await controller.start_ensemble("")

    async def test_noop_update_agent_progress(self) -> None:
        """Test that update_agent_progress does nothing."""
        controller = NoOpProgressController()

        # Should not raise any exceptions
        await controller.update_agent_progress("agent1", "running")
        await controller.update_agent_progress("agent2", "completed")
        await controller.update_agent_progress("", "")

    async def test_noop_complete_ensemble(self) -> None:
        """Test that complete_ensemble does nothing."""
        controller = NoOpProgressController()

        # Should not raise any exceptions
        await controller.complete_ensemble()
        await controller.complete_ensemble()

    async def test_noop_async_workflow(self) -> None:
        """Test a complete async workflow with NoOp controller."""
        controller = NoOpProgressController()

        # Simulate full ensemble workflow
        await controller.start_ensemble("test_ensemble")
        await controller.update_agent_progress("agent1", "starting")
        await controller.update_agent_progress("agent1", "running")
        await controller.update_agent_progress("agent1", "completed")
        await controller.update_agent_progress("agent2", "starting")
        await controller.update_agent_progress("agent2", "completed")
        await controller.complete_ensemble()


class TestCustomProgressController:
    """Test custom implementations of ProgressController."""

    def test_custom_controller_with_state_tracking(self) -> None:
        """Test a custom controller that tracks pause/resume state."""

        class StatefulController(ProgressController):
            def __init__(self) -> None:
                self.is_paused = False
                self.current_agent: str | None = None
                self.pause_count = 0
                self.resume_count = 0

            def pause_for_user_input(self, agent_name: str, prompt: str = "") -> None:
                self.is_paused = True
                self.current_agent = agent_name
                self.pause_count += 1

            def resume_from_user_input(self, agent_name: str) -> None:
                self.is_paused = False
                self.current_agent = None
                self.resume_count += 1

            async def start_ensemble(self, ensemble_name: str) -> None:
                pass

            async def update_agent_progress(self, agent_name: str, status: str) -> None:
                pass

            async def complete_ensemble(self) -> None:
                pass

        controller = StatefulController()

        # Initial state
        assert controller.is_paused is False
        assert controller.current_agent is None
        assert controller.pause_count == 0
        assert controller.resume_count == 0

        # Pause
        controller.pause_for_user_input("test_agent", "Enter input: ")
        assert controller.is_paused is True
        assert controller.current_agent == "test_agent"
        assert controller.pause_count == 1
        assert controller.resume_count == 0

        # Resume
        controller.resume_from_user_input("test_agent")
        assert controller.is_paused is False
        assert controller.current_agent is None
        assert controller.pause_count == 1
        assert controller.resume_count == 1

    def test_custom_controller_with_validation(self) -> None:
        """Test a custom controller that validates agent names."""

        class ValidatingController(ProgressController):
            def __init__(self, allowed_agents: list[str]) -> None:
                self.allowed_agents = allowed_agents
                self.paused_agent: str | None = None

            def pause_for_user_input(self, agent_name: str, prompt: str = "") -> None:
                if agent_name not in self.allowed_agents:
                    raise ValueError(f"Unknown agent: {agent_name}")
                self.paused_agent = agent_name

            def resume_from_user_input(self, agent_name: str) -> None:
                if agent_name != self.paused_agent:
                    raise ValueError(
                        f"Cannot resume {agent_name}, {self.paused_agent} is paused"
                    )
                self.paused_agent = None

            async def start_ensemble(self, ensemble_name: str) -> None:
                pass

            async def update_agent_progress(self, agent_name: str, status: str) -> None:
                pass

            async def complete_ensemble(self) -> None:
                pass

        controller = ValidatingController(["agent1", "agent2"])

        # Valid pause/resume
        controller.pause_for_user_input("agent1", "Input: ")
        controller.resume_from_user_input("agent1")

        # Invalid agent name
        with pytest.raises(ValueError, match="Unknown agent: agent3"):
            controller.pause_for_user_input("agent3", "Input: ")

        # Resume wrong agent
        controller.pause_for_user_input("agent2", "Input: ")
        with pytest.raises(ValueError, match="Cannot resume agent1, agent2 is paused"):
            controller.resume_from_user_input("agent1")

    def test_custom_controller_with_logging(self) -> None:
        """Test a custom controller that logs operations."""

        class LoggingController(ProgressController):
            def __init__(self) -> None:
                self.log: list[str] = []

            def pause_for_user_input(self, agent_name: str, prompt: str = "") -> None:
                self.log.append(f"PAUSE: {agent_name} - {prompt}")

            def resume_from_user_input(self, agent_name: str) -> None:
                self.log.append(f"RESUME: {agent_name}")

            async def start_ensemble(self, ensemble_name: str) -> None:
                pass

            async def update_agent_progress(self, agent_name: str, status: str) -> None:
                pass

            async def complete_ensemble(self) -> None:
                pass

        controller = LoggingController()

        # Execute workflow
        controller.pause_for_user_input("agent1", "Enter name: ")
        controller.resume_from_user_input("agent1")
        controller.pause_for_user_input("agent2", "Enter age: ")
        controller.resume_from_user_input("agent2")

        # Check log
        assert controller.log == [
            "PAUSE: agent1 - Enter name: ",
            "RESUME: agent1",
            "PAUSE: agent2 - Enter age: ",
            "RESUME: agent2",
        ]


class TestProgressControllerIntegration:
    """Test integration scenarios with progress controllers."""

    def test_controller_polymorphism(self) -> None:
        """Test that different controllers can be used polymorphically."""

        def execute_with_controller(controller: ProgressController) -> None:
            """Execute operations with any ProgressController."""
            controller.pause_for_user_input("worker", "Input needed: ")
            controller.resume_from_user_input("worker")

        # Should work with NoOp controller
        noop = NoOpProgressController()
        execute_with_controller(noop)

        # Should work with custom controller
        class CustomController(ProgressController):
            def pause_for_user_input(self, agent_name: str, prompt: str = "") -> None:
                pass

            def resume_from_user_input(self, agent_name: str) -> None:
                pass

            async def start_ensemble(self, ensemble_name: str) -> None:
                pass

            async def update_agent_progress(self, agent_name: str, status: str) -> None:
                pass

            async def complete_ensemble(self) -> None:
                pass

        custom = CustomController()
        execute_with_controller(custom)

    def test_controller_as_dependency(self) -> None:
        """Test progress controller as a dependency injection."""

        class WorkflowExecutor:
            def __init__(self, controller: ProgressController) -> None:
                self.controller = controller

            def execute_with_input(self, agent: str, prompt: str) -> str:
                self.controller.pause_for_user_input(agent, prompt)
                # Simulate getting input
                result = "user_input"
                self.controller.resume_from_user_input(agent)
                return result

        # Test with NoOp controller
        executor = WorkflowExecutor(NoOpProgressController())
        result = executor.execute_with_input("test_agent", "Enter value: ")
        assert result == "user_input"

    def test_controller_factory_pattern(self) -> None:
        """Test factory pattern for creating appropriate controllers."""

        def create_controller(enable_progress: bool) -> ProgressController:
            """Factory function to create appropriate controller."""
            if enable_progress:
                # In real code, would return actual progress controller
                return NoOpProgressController()  # Using NoOp as placeholder
            else:
                return NoOpProgressController()

        # Test factory
        controller_with_progress = create_controller(True)
        assert isinstance(controller_with_progress, ProgressController)

        controller_without_progress = create_controller(False)
        assert isinstance(controller_without_progress, NoOpProgressController)
