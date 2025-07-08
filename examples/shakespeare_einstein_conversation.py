#!/usr/bin/env python3
"""
Shakespeare ↔ Einstein Conversation Example

Demonstrates practical multi-agent conversation using Ollama.

Usage:
    python examples/shakespeare_einstein_conversation.py

Requirements:
    - Ollama running locally
    - llama3 model available: ollama pull llama3
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_orc.models import OllamaModel
from llm_orc.orchestration import Agent, ConversationOrchestrator
from llm_orc.roles import RoleDefinition


async def main():
    """Run Shakespeare ↔ Einstein conversation using Ollama."""
    print("🎭 Starting Shakespeare ↔ Einstein Conversation")
    print("=" * 50)

    try:
        # Create Ollama models (using llama3 - adjust if you have different models)
        shakespeare_model = OllamaModel(
            model_name="llama3", host="http://localhost:11434"
        )
        einstein_model = OllamaModel(model_name="llama3", host="http://localhost:11434")

        # Create role definitions
        shakespeare_role = RoleDefinition(
            name="shakespeare",
            prompt=(
                "You are William Shakespeare, the renowned playwright and poet from "
                "the Elizabethan era. Speak in eloquent, poetic English with 'thee', "
                "'thou', 'doth', 'hath', etc. You are deeply curious about the natural "
                "world and the cosmos. Keep responses to 2-3 sentences and stay in "
                "character. You find great wonder in both art and the mysteries of "
                "existence."
            ),
            context={
                "era": "Elizabethan",
                "specialties": ["poetry", "drama", "language"],
            },
        )

        einstein_role = RoleDefinition(
            name="einstein",
            prompt=(
                "You are Albert Einstein, the brilliant theoretical physicist. "
                "Speak thoughtfully and with wonder about science, imagination, and "
                "the universe. You appreciate the beauty in both science and art. "
                "Keep responses to 2-3 sentences and stay in character. You see deep "
                "connections between mathematical elegance and artistic beauty."
            ),
            context={
                "era": "20th century",
                "specialties": ["physics", "relativity", "philosophy"],
            },
        )

        # Create agents
        shakespeare = Agent("shakespeare", shakespeare_role, shakespeare_model)
        einstein = Agent("einstein", einstein_role, einstein_model)

        # Create orchestrator
        orchestrator = ConversationOrchestrator()
        orchestrator.register_agent(shakespeare)
        orchestrator.register_agent(einstein)

        print("🤖 Agents created and registered")
        print("📡 Starting conversation...\n")

        # Start conversation
        conversation_id = await orchestrator.start_conversation(
            participants=["shakespeare", "einstein"],
            topic="The Nature of Beauty in Art and Science",
        )

        print("💬 CONVERSATION: The Nature of Beauty in Art and Science")
        print("-" * 50)

        # Shakespeare opens the conversation
        print("\n🎭 SHAKESPEARE:")
        shakespeare_opening = await shakespeare.respond_to_message(
            "Good sir Einstein, I have heard tell of thy wondrous discoveries "
            "about the cosmos. Pray tell, what connection dost thou perceive "
            "between the beauty of mathematical truth and the beauty found in verse?"
        )
        print(f"   {shakespeare_opening}")

        # Einstein responds
        print("\n🧠 EINSTEIN:")
        einstein_response = await orchestrator.send_agent_message(
            sender="shakespeare",
            recipient="einstein",
            content=shakespeare_opening,
            conversation_id=conversation_id,
        )
        print(f"   {einstein_response}")

        # Shakespeare responds back
        print("\n🎭 SHAKESPEARE:")
        shakespeare_response = await orchestrator.send_agent_message(
            sender="einstein",
            recipient="shakespeare",
            content=einstein_response,
            conversation_id=conversation_id,
        )
        print(f"   {shakespeare_response}")

        # One more round - Einstein's final thought
        print("\n🧠 EINSTEIN:")
        einstein_final = await orchestrator.send_agent_message(
            sender="shakespeare",
            recipient="einstein",
            content=shakespeare_response,
            conversation_id=conversation_id,
        )
        print(f"   {einstein_final}")

        print("\n" + "=" * 50)
        print("✨ Conversation complete!")
        print(f"📊 Shakespeare spoke {len(shakespeare.conversation_history)} times")
        print(f"📊 Einstein spoke {len(einstein.conversation_history)} times")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure Ollama is running and llama3 model is available:")
        print("   ollama serve")
        print("   ollama pull llama3")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
