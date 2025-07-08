"""Agent orchestration for multi-agent conversations."""

from datetime import datetime
from typing import Any

from llm_orc.communication import ConversationManager, MessageProtocol
from llm_orc.models import ModelInterface
from llm_orc.roles import RoleDefinition


class Agent:
    """Agent that combines role, model, and conversation capabilities."""

    def __init__(self, name: str, role: RoleDefinition, model: ModelInterface):
        self.name = name
        self.role = role
        self.model = model
        self.conversation_history: list[dict[str, Any]] = []

    async def respond_to_message(self, message: str) -> str:
        """Generate response to a message using the agent's role and model."""
        # Generate response using model
        response = await self.model.generate_response(
            message=message, role_prompt=self.role.prompt
        )

        # Store in conversation history
        self.conversation_history.append(
            {"message": message, "response": response, "timestamp": datetime.now()}
        )

        return response


class ConversationOrchestrator:
    """Orchestrates multi-agent conversations."""

    def __init__(self) -> None:
        self.conversation_manager = ConversationManager()
        self.message_protocol = MessageProtocol(self.conversation_manager)
        self.agents: dict[str, Agent] = {}

    def register_agent(self, agent: Agent) -> None:
        """Register an agent for orchestration."""
        self.agents[agent.name] = agent

    async def start_conversation(
        self, participants: list[str], topic: str, initial_message: str | None = None
    ) -> str:
        """Start a conversation between registered agents."""
        # Validate participants
        for participant in participants:
            if participant not in self.agents:
                raise ValueError(f"Agent '{participant}' not registered")

        # Create conversation
        conversation_id = self.conversation_manager.start_conversation(
            participants=participants, topic=topic
        )

        # Send initial message if provided
        if initial_message and len(participants) >= 2:
            await self.send_agent_message(
                sender=participants[0],
                recipient=participants[1],
                content=initial_message,
                conversation_id=conversation_id,
            )

        return conversation_id

    async def send_agent_message(
        self, sender: str, recipient: str, content: str, conversation_id: str
    ) -> str:
        """Send message from one agent to another and get response."""
        if sender not in self.agents or recipient not in self.agents:
            raise ValueError("Both sender and recipient must be registered agents")

        # Send message through protocol
        await self.message_protocol.send_message(
            sender=sender,
            recipient=recipient,
            content=content,
            conversation_id=conversation_id,
        )

        # Get recipient agent to respond
        recipient_agent = self.agents[recipient]
        response = await recipient_agent.respond_to_message(content)

        return str(response)


class PRReviewOrchestrator:
    """Orchestrates multi-agent PR reviews with specialist feedback."""

    def __init__(self) -> None:
        self.reviewers: dict[str, Agent] = {}

    def register_reviewer(self, agent: Agent) -> None:
        """Register a specialist reviewer agent."""
        self.reviewers[agent.name] = agent

    async def review_pr(self, pr_data: dict[str, Any]) -> dict[str, Any]:
        """Conduct multi-agent PR review and return consolidated results."""
        # Format PR data for review
        pr_summary = self._format_pr_for_review(pr_data)

        # Collect reviews from all specialist agents
        reviews: list[dict[str, Any]] = []
        for reviewer_name, reviewer_agent in self.reviewers.items():
            feedback = await reviewer_agent.respond_to_message(pr_summary)
            reviews.append(
                {
                    "reviewer": reviewer_name,
                    "feedback": feedback,
                    "specialization": (reviewer_agent.role.context or {}).get(
                        "specialties", []
                    ),
                }
            )

        # Generate consolidated summary
        summary = await self._generate_summary(reviews, pr_data)

        return {
            "pr_title": pr_data["title"],
            "reviews": reviews,
            "summary": summary,
            "total_reviewers": len(reviews),
        }

    def _format_pr_for_review(self, pr_data: dict[str, Any]) -> str:
        """Format PR data into a review-friendly string."""
        return f"""
PR Title: {pr_data["title"]}
Description: {pr_data["description"]}

Files Changed: {", ".join(pr_data.get("files_changed", []))}
Additions: {pr_data.get("additions", 0)} lines
Deletions: {pr_data.get("deletions", 0)} lines

Code Changes:
{pr_data.get("diff", "No diff available")}

Please provide your specialist review focusing on your area of expertise.
""".strip()

    async def _generate_summary(
        self, reviews: list[dict[str, Any]], pr_data: dict[str, Any]
    ) -> str:
        """Generate a consolidated summary of all reviews."""
        # For now, create a simple summary - could be enhanced with LLM later
        security_issues = []
        code_quality_issues = []
        ux_issues = []

        for review in reviews:
            feedback = review["feedback"].lower()
            if (
                "security" in feedback
                or "vulnerability" in feedback
                or "critical" in feedback
            ):
                security_issues.append(review["reviewer"])
            if (
                "code quality" in feedback
                or "best practices" in feedback
                or "testing" in feedback
            ):
                code_quality_issues.append(review["reviewer"])
            if "ux" in feedback or "user" in feedback or "accessibility" in feedback:
                ux_issues.append(review["reviewer"])

        summary_parts = []
        if security_issues:
            summary_parts.append(
                f"🔒 Security concerns raised by: {', '.join(security_issues)}"
            )
        if code_quality_issues:
            summary_parts.append(
                f"📝 Code quality feedback from: {', '.join(code_quality_issues)}"
            )
        if ux_issues:
            summary_parts.append(
                f"👤 UX considerations noted by: {', '.join(ux_issues)}"
            )

        if not summary_parts:
            summary_parts.append(
                "✅ No major issues identified across all review areas"
            )

        return " | ".join(summary_parts)
