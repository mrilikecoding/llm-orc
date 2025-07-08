"""Test suite for agent orchestration."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.models import ModelInterface
from llm_orc.roles import RoleDefinition


class TestAgent:
    """Test the Agent class."""

    def test_agent_creation(self) -> None:
        """Should create an agent with name, role, and model."""
        # Arrange
        role = RoleDefinition(
            name="shakespeare",
            prompt="You are William Shakespeare, the renowned playwright.",
        )
        model = Mock(spec=ModelInterface)
        model.name = "test-model"

        # Act
        from llm_orc.orchestration import Agent

        agent = Agent(name="shakespeare", role=role, model=model)

        # Assert
        assert agent.name == "shakespeare"
        assert agent.role == role
        assert agent.model == model
        assert len(agent.conversation_history) == 0

    @pytest.mark.asyncio
    async def test_agent_respond_to_message(self) -> None:
        """Should generate response using role and model."""
        # Arrange
        role = RoleDefinition(
            name="shakespeare",
            prompt="You are William Shakespeare, the renowned playwright.",
        )
        model = AsyncMock(spec=ModelInterface)
        model.generate_response.return_value = (
            "Hark! What light through yonder window breaks?"
        )

        from llm_orc.orchestration import Agent

        agent = Agent(name="shakespeare", role=role, model=model)

        # Act - This will fail because respond_to_message doesn't exist yet
        response = await agent.respond_to_message("Tell me about beauty.")

        # Assert
        assert response == "Hark! What light through yonder window breaks?"
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0]["message"] == "Tell me about beauty."
        assert agent.conversation_history[0]["response"] == response

        # Verify model was called with correct parameters
        model.generate_response.assert_called_once_with(
            message="Tell me about beauty.", role_prompt=role.prompt
        )


class TestPracticalExamples:
    """Test practical multi-agent conversation examples."""

    @pytest.mark.asyncio
    async def test_shakespeare_einstein_conversation(self) -> None:
        """Should orchestrate a conversation between Shakespeare and Einstein."""
        # Arrange - This will fail because ConversationOrchestrator doesn't exist yet
        from llm_orc.orchestration import Agent, ConversationOrchestrator

        # Create Shakespeare agent
        shakespeare_role = RoleDefinition(
            name="shakespeare",
            prompt="You are William Shakespeare, the renowned playwright and poet. "
            "Speak in eloquent Elizabethan English with poetic flair. "
            "You are curious about science and the natural world.",
            context={
                "era": "Elizabethan",
                "specialties": ["poetry", "drama", "language"],
            },
        )
        shakespeare_model = AsyncMock(spec=ModelInterface)
        shakespeare_model.generate_response.return_value = (
            "Hark! Good sir, what mysteries doth the cosmos hold? "
            "Methinks the stars themselves dance to hidden laws most wondrous."
        )
        shakespeare_agent = Agent("shakespeare", shakespeare_role, shakespeare_model)

        # Create Einstein agent
        einstein_role = RoleDefinition(
            name="einstein",
            prompt="You are Albert Einstein, the brilliant theoretical physicist. "
            "Speak thoughtfully about science, imagination, and the mysteries of the universe. "  # noqa: E501
            "You appreciate the beauty in both science and art.",
            context={
                "era": "20th century",
                "specialties": ["physics", "relativity", "philosophy"],
            },
        )
        einstein_model = AsyncMock(spec=ModelInterface)
        einstein_model.generate_response.return_value = (
            "Ah, my poetic friend! The universe is indeed a symphony of mathematical harmony. "  # noqa: E501
            "What you call the dance of stars, I see as the elegant curvature of spacetime itself."  # noqa: E501
        )
        einstein_agent = Agent("einstein", einstein_role, einstein_model)

        # Create orchestrator
        orchestrator = ConversationOrchestrator()
        # Mock the message delivery to avoid async timeout issues in tests
        with patch.object(
            orchestrator.message_protocol, "deliver_message", new_callable=AsyncMock
        ):
            orchestrator.register_agent(shakespeare_agent)
            orchestrator.register_agent(einstein_agent)

            # Act - Start conversation
            conversation_id = await orchestrator.start_conversation(
                participants=["shakespeare", "einstein"],
                topic="The Nature of Beauty in Art and Science",
                initial_message="What think you of the relationship between beauty and truth?",  # noqa: E501
            )

            # Einstein responds
            einstein_response = await orchestrator.send_agent_message(
                sender="einstein",
                recipient="shakespeare",
                content="Beauty in science comes from elegant equations that reveal deep truths.",  # noqa: E501
                conversation_id=conversation_id,
            )

            # Assert
            assert conversation_id is not None
            assert einstein_response is not None
            # The response should be from Shakespeare (recipient) when Einstein sends a message to Shakespeare  # noqa: E501
            assert (
                "hark" in einstein_response.lower()
                or "cosmos" in einstein_response.lower()
            )

            # Verify both agents were called
            shakespeare_model.generate_response.assert_called_once()
            einstein_model.generate_response.assert_called_once()

            # Verify conversation history
            assert len(shakespeare_agent.conversation_history) >= 1
            assert len(einstein_agent.conversation_history) >= 1

    @pytest.mark.asyncio
    async def test_pr_review_panel(self) -> None:
        """Should orchestrate a PR review with multiple specialist agents."""
        # Arrange - This will fail because PRReviewOrchestrator doesn't exist yet
        from llm_orc.orchestration import Agent, PRReviewOrchestrator

        # Mock PR data
        pr_data = {
            "title": "Add user authentication system",
            "description": "Implements JWT-based authentication with password hashing",
            "diff": """
+import bcrypt
+import jwt
+from datetime import datetime, timedelta
+
+def hash_password(password: str) -> str:
+    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
+
+def verify_password(password: str, hashed: str) -> bool:
+    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
+
+def create_jwt_token(user_id: int) -> str:
+    payload = {
+        'user_id': user_id,
+        'exp': datetime.utcnow() + timedelta(hours=24)
+    }
+    return jwt.encode(payload, 'secret_key', algorithm='HS256')
""",
            "files_changed": ["auth.py", "requirements.txt"],
            "additions": 15,
            "deletions": 2,
        }

        # Create specialist agents
        senior_dev_role = RoleDefinition(
            name="senior_developer",
            prompt="You are a senior software developer focused on code quality, best practices, and maintainability. "  # noqa: E501
            "Review code for: design patterns, naming conventions, error handling, testing needs, and technical debt. "  # noqa: E501
            "Provide constructive feedback in 2-3 sentences.",
            context={"specialties": ["code_quality", "best_practices", "architecture"]},
        )
        senior_dev_model = AsyncMock(spec=ModelInterface)
        senior_dev_model.generate_response.return_value = (
            "Code structure looks clean with proper separation of concerns. "
            "Consider adding error handling for JWT encoding/decoding and using environment variables for the secret key. "  # noqa: E501
            "Unit tests for password hashing and token validation would strengthen this implementation."  # noqa: E501
        )
        senior_dev = Agent("senior_dev", senior_dev_role, senior_dev_model)

        security_expert_role = RoleDefinition(
            name="security_expert",
            prompt="You are a cybersecurity expert focused on identifying security vulnerabilities and best practices. "  # noqa: E501
            "Review code for: authentication flaws, data validation, secret management, encryption standards. "  # noqa: E501
            "Provide security-focused feedback in 2-3 sentences.",
            context={"specialties": ["security", "encryption", "authentication"]},
        )
        security_expert_model = AsyncMock(spec=ModelInterface)
        security_expert_model.generate_response.return_value = (
            "CRITICAL: Hardcoded secret key 'secret_key' is a major security vulnerability - use environment variables. "  # noqa: E501
            "Password hashing with bcrypt is good practice, but consider adding rate limiting for authentication attempts. "  # noqa: E501
            "JWT expiration time of 24 hours might be too long for sensitive applications."  # noqa: E501
        )
        security_expert = Agent(
            "security_expert", security_expert_role, security_expert_model
        )

        ux_reviewer_role = RoleDefinition(
            name="ux_reviewer",
            prompt="You are a UX specialist focused on user experience and accessibility. "  # noqa: E501
            "Review code changes for: user impact, error messages, accessibility, usability implications. "  # noqa: E501
            "Provide UX-focused feedback in 2-3 sentences.",
            context={"specialties": ["user_experience", "accessibility", "usability"]},
        )
        ux_reviewer_model = AsyncMock(spec=ModelInterface)
        ux_reviewer_model.generate_response.return_value = (
            "From a UX perspective, this backend authentication system needs clear error messaging for users. "  # noqa: E501
            "Consider implementing proper password strength requirements and user-friendly feedback for login failures. "  # noqa: E501
            "The 24-hour token expiration should align with user expectations and include session extension options."  # noqa: E501
        )
        ux_reviewer = Agent("ux_reviewer", ux_reviewer_role, ux_reviewer_model)

        # Create PR review orchestrator
        pr_orchestrator = PRReviewOrchestrator()
        pr_orchestrator.register_reviewer(senior_dev)
        pr_orchestrator.register_reviewer(security_expert)
        pr_orchestrator.register_reviewer(ux_reviewer)

        # Act - Conduct PR review
        review_results = await pr_orchestrator.review_pr(pr_data)

        # Assert
        assert review_results is not None
        assert "reviews" in review_results
        assert len(review_results["reviews"]) == 3

        # Check that each specialist provided feedback
        reviews = review_results["reviews"]
        senior_dev_review = next(r for r in reviews if r["reviewer"] == "senior_dev")
        security_review = next(r for r in reviews if r["reviewer"] == "security_expert")
        ux_review = next(r for r in reviews if r["reviewer"] == "ux_reviewer")

        assert (
            "code structure" in senior_dev_review["feedback"].lower()
            or "error handling" in senior_dev_review["feedback"].lower()
        )
        assert (
            "security" in security_review["feedback"].lower()
            or "vulnerability" in security_review["feedback"].lower()
        )
        assert (
            "ux" in ux_review["feedback"].lower()
            or "user" in ux_review["feedback"].lower()
        )

        # Verify all models were called
        senior_dev_model.generate_response.assert_called_once()
        security_expert_model.generate_response.assert_called_once()
        ux_reviewer_model.generate_response.assert_called_once()

        # Check consolidated summary exists
        assert "summary" in review_results
        assert review_results["summary"] is not None
