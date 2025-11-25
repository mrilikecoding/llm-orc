"""Tests for conversational dependency resolver with condition support."""

from unittest.mock import Mock

from llm_orc.core.execution.conversational_dependency_resolver import (
    ConversationalDependencyResolver,
)
from llm_orc.schemas.conversational_agent import (
    ConversationalAgent,
    ConversationalDependency,
    ConversationState,
)


class TestConversationalDependencyResolver:
    """Test conversational dependency resolution with conditions."""

    def setup_resolver(self) -> tuple[ConversationalDependencyResolver, Mock]:
        """Set up resolver with mocked role description function."""
        mock_role_resolver = Mock()
        mock_role_resolver.return_value = "Test Role"

        resolver = ConversationalDependencyResolver(role_resolver=mock_role_resolver)

        return resolver, mock_role_resolver

    def test_get_ready_agents_for_conversation_no_dependencies(self) -> None:
        """Test getting ready agents when no dependencies exist."""
        resolver, _ = self.setup_resolver()

        agents = [
            ConversationalAgent(name="agent1", script="script1.py"),
            ConversationalAgent(name="agent2", model_profile="gpt-4"),
        ]
        conversation_state = ConversationState()

        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )

        assert len(ready_agents) == 2
        assert ready_agents[0].name == "agent1"
        assert ready_agents[1].name == "agent2"

    def test_get_ready_agents_for_conversation_unconditional_dependencies(
        self,
    ) -> None:
        """Test getting ready agents with unconditional dependencies."""
        resolver, _ = self.setup_resolver()

        agents = [
            ConversationalAgent(name="agent1", script="script1.py"),
            ConversationalAgent(
                name="agent2",
                model_profile="gpt-4",
                dependencies=[ConversationalDependency(agent_name="agent1")],
            ),
        ]
        conversation_state = ConversationState()

        # Initially, only agent1 should be ready
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 1
        assert ready_agents[0].name == "agent1"

        # After agent1 executes, agent2 should be ready
        conversation_state.agent_execution_count["agent1"] = 1
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 1
        assert ready_agents[0].name == "agent2"

    def test_get_ready_agents_for_conversation_conditional_dependencies(self) -> None:
        """Test getting ready agents with conditional dependencies."""
        resolver, _ = self.setup_resolver()

        agents = [
            ConversationalAgent(name="agent1", script="script1.py"),
            ConversationalAgent(
                name="agent2",
                model_profile="gpt-4",
                dependencies=[
                    ConversationalDependency(
                        agent_name="agent1",
                        condition="context.get('needs_analysis', False)",
                    )
                ],
            ),
        ]
        conversation_state = ConversationState()
        conversation_state.agent_execution_count["agent1"] = 1

        # Initially, condition is False, so agent2 should not be ready
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 0

        # After setting condition to True, agent2 should be ready
        conversation_state.accumulated_context["needs_analysis"] = True
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 1
        assert ready_agents[0].name == "agent2"

    def test_get_ready_agents_for_conversation_max_executions_limit(self) -> None:
        """Test that agents respect max_executions limit."""
        resolver, _ = self.setup_resolver()

        agents = [
            ConversationalAgent(name="agent1", script="script1.py"),
            ConversationalAgent(
                name="agent2",
                model_profile="gpt-4",
                dependencies=[
                    ConversationalDependency(
                        agent_name="agent1",
                        condition="turn_count > 0",
                        max_executions=2,
                    )
                ],
            ),
        ]
        conversation_state = ConversationState()
        conversation_state.turn_count = 1
        conversation_state.agent_execution_count["agent1"] = 1

        # Agent2 should be ready initially (0 executions < 2 max)
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 1
        assert ready_agents[0].name == "agent2"

        # After one execution, still under limit
        conversation_state.agent_execution_count["agent2"] = 1
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 1
        assert ready_agents[0].name == "agent2"

        # After reaching max_executions, should not be ready
        conversation_state.agent_execution_count["agent2"] = 2
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 0

    def test_get_ready_agents_for_conversation_complex_conditions(self) -> None:
        """Test agents with complex condition expressions."""
        resolver, _ = self.setup_resolver()

        agents = [
            ConversationalAgent(name="agent1", script="script1.py"),
            ConversationalAgent(
                name="agent2",
                model_profile="gpt-4",
                dependencies=[
                    ConversationalDependency(
                        agent_name="agent1",
                        condition="turn_count >= 2 and len(context) > 0",
                    )
                ],
            ),
        ]
        conversation_state = ConversationState()
        conversation_state.agent_execution_count["agent1"] = 1

        # Condition not met (turn_count < 2)
        conversation_state.turn_count = 1
        conversation_state.accumulated_context = {"key": "value"}
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 0

        # Condition not met (context empty)
        conversation_state.turn_count = 2
        conversation_state.accumulated_context = {}
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 0

        # Condition met
        conversation_state.turn_count = 2
        conversation_state.accumulated_context = {"key": "value"}
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 1
        assert ready_agents[0].name == "agent2"

    def test_get_ready_agents_for_conversation_invalid_condition(self) -> None:
        """Test agents with invalid conditions default to False."""
        resolver, _ = self.setup_resolver()

        agents = [
            ConversationalAgent(name="agent1", script="script1.py"),
            ConversationalAgent(
                name="agent2",
                model_profile="gpt-4",
                dependencies=[
                    ConversationalDependency(
                        agent_name="agent1",
                        condition="invalid_syntax ??",  # Invalid Python
                    )
                ],
            ),
        ]
        conversation_state = ConversationState()
        conversation_state.agent_execution_count["agent1"] = 1

        # Invalid condition should default to False
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 0

    def test_get_ready_agents_for_conversation_multiple_dependencies(self) -> None:
        """Test agents with multiple dependencies including conditions."""
        resolver, _ = self.setup_resolver()

        agents = [
            ConversationalAgent(name="agent1", script="script1.py"),
            ConversationalAgent(name="agent2", script="script2.py"),
            ConversationalAgent(
                name="agent3",
                model_profile="gpt-4",
                dependencies=[
                    ConversationalDependency(agent_name="agent1"),  # Unconditional
                    ConversationalDependency(
                        agent_name="agent2",
                        condition="context.get('ready', False)",
                    ),
                ],
            ),
        ]
        conversation_state = ConversationState()

        # Both agent1 and agent2 executed, but condition not met
        conversation_state.agent_execution_count["agent1"] = 1
        conversation_state.agent_execution_count["agent2"] = 1
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 0

        # Both executed and condition met
        conversation_state.accumulated_context["ready"] = True
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 1
        assert ready_agents[0].name == "agent3"

    def test_get_ready_agents_for_conversation_requires_all_false(self) -> None:
        """Test agents with requires_all=False (OR logic)."""
        resolver, _ = self.setup_resolver()

        agents = [
            ConversationalAgent(name="agent1", script="script1.py"),
            ConversationalAgent(name="agent2", script="script2.py"),
            ConversationalAgent(
                name="agent3",
                model_profile="gpt-4",
                dependencies=[
                    ConversationalDependency(
                        agent_name="agent1",
                        requires_all=False,
                    ),
                    ConversationalDependency(
                        agent_name="agent2",
                        condition="context.get('ready', False)",
                        requires_all=False,
                    ),
                ],
            ),
        ]
        conversation_state = ConversationState()

        # Only agent1 executed (should be sufficient with OR logic)
        # Agent2 should also be ready since it has no dependencies
        conversation_state.agent_execution_count["agent1"] = 1
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 2
        ready_names = {agent.name for agent in ready_agents}
        assert ready_names == {"agent2", "agent3"}

        # Reset and test with only conditional dependency met
        # Agent1 and Agent2 should be at max executions, only agent3 ready
        conversation_state.agent_execution_count["agent1"] = 1  # at max executions
        conversation_state.agent_execution_count["agent2"] = 1  # at max executions
        conversation_state.accumulated_context["ready"] = True
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 1
        assert ready_agents[0].name == "agent3"

    def test_get_ready_agents_for_conversation_backward_compatibility(self) -> None:
        """Test backward compatibility with non-conversational agents."""
        resolver, _ = self.setup_resolver()

        # Mix of conversational and traditional dependency patterns
        agents = [
            ConversationalAgent(name="agent1", script="script1.py"),
            ConversationalAgent(
                name="agent2",
                model_profile="gpt-4",
                dependencies=[ConversationalDependency(agent_name="agent1")],
            ),
        ]
        conversation_state = ConversationState()

        # Should work the same as traditional dependencies
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 1
        assert ready_agents[0].name == "agent1"

        conversation_state.agent_execution_count["agent1"] = 1
        ready_agents = resolver.get_ready_agents_for_conversation(
            agents, conversation_state
        )
        assert len(ready_agents) == 1
        assert ready_agents[0].name == "agent2"

    def test_evaluate_dependency_condition_safe_evaluation(self) -> None:
        """Test that condition evaluation is safe and restricted."""
        resolver, _ = self.setup_resolver()

        conversation_state = ConversationState()
        conversation_state.turn_count = 5
        conversation_state.accumulated_context = {"key": "value"}

        # Test safe expressions work
        assert resolver._evaluate_dependency_condition(
            "turn_count > 3", conversation_state
        )
        assert resolver._evaluate_dependency_condition(
            "context.get('key') == 'value'", conversation_state
        )
        assert resolver._evaluate_dependency_condition(
            "len(context) == 1", conversation_state
        )

        # Test dangerous expressions are blocked or fail safely
        assert not resolver._evaluate_dependency_condition(
            "__import__('os').system('rm -rf /')", conversation_state
        )
        assert not resolver._evaluate_dependency_condition(
            "open('/etc/passwd')", conversation_state
        )

    def test_has_dependencies_satisfied_for_agent(self) -> None:
        """Test checking if all dependencies are satisfied for an agent."""
        resolver, _ = self.setup_resolver()

        agent = ConversationalAgent(
            name="test_agent",
            model_profile="gpt-4",
            dependencies=[
                ConversationalDependency(agent_name="dep1"),
                ConversationalDependency(
                    agent_name="dep2",
                    condition="context.get('ready', False)",
                ),
            ],
        )
        conversation_state = ConversationState()

        # No dependencies executed
        assert not resolver._has_dependencies_satisfied_for_agent(
            agent, conversation_state
        )

        # One dependency executed but condition not met
        conversation_state.agent_execution_count["dep1"] = 1
        assert not resolver._has_dependencies_satisfied_for_agent(
            agent, conversation_state
        )

        # Both dependencies executed and condition met
        conversation_state.agent_execution_count["dep2"] = 1
        conversation_state.accumulated_context["ready"] = True
        assert resolver._has_dependencies_satisfied_for_agent(agent, conversation_state)

    def test_can_agent_execute_more_times(self) -> None:
        """Test checking if agent can execute more times."""
        resolver, _ = self.setup_resolver()

        agent = ConversationalAgent(
            name="test_agent",
            model_profile="gpt-4",
            dependencies=[
                ConversationalDependency(
                    agent_name="dep1",
                    max_executions=3,
                )
            ],
        )
        conversation_state = ConversationState()

        # Agent hasn't executed yet
        assert resolver._can_agent_execute_more_times(agent, conversation_state)

        # Agent executed once (under limit)
        conversation_state.agent_execution_count["test_agent"] = 1
        assert resolver._can_agent_execute_more_times(agent, conversation_state)

        # Agent executed 3 times (at limit with max dependency executions)
        conversation_state.agent_execution_count["test_agent"] = 3
        assert not resolver._can_agent_execute_more_times(agent, conversation_state)
