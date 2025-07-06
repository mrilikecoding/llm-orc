# Agent Orchestration Documentation

## Overview

LLM Orchestra provides a powerful agent orchestration system that enables multi-agent conversations and specialized workflows. This document covers the core concepts, usage patterns, and practical examples.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Quick Start](#quick-start)
3. [Agent Creation](#agent-creation)
4. [Conversation Orchestration](#conversation-orchestration)
5. [Specialized Orchestrators](#specialized-orchestrators)
6. [Performance Characteristics](#performance-characteristics)
7. [Examples](#examples)
8. [Best Practices](#best-practices)

## Core Concepts

### Agent

An **Agent** combines three key components:
- **Role**: Defines the agent's personality, expertise, and behavior
- **Model**: The underlying LLM (Claude, Gemini, Ollama, etc.)
- **Conversation History**: Maintains context across interactions

```python
from llm_orc.orchestration import Agent
from llm_orc.models import OllamaModel
from llm_orc.roles import RoleDefinition

# Create a role
role = RoleDefinition(
    name="shakespeare",
    prompt="You are William Shakespeare, the renowned playwright.",
    context={"era": "Elizabethan", "specialties": ["poetry", "drama"]}
)

# Create a model
model = OllamaModel(model_name="llama3")

# Create an agent
agent = Agent(name="shakespeare", role=role, model=model)
```

### ConversationOrchestrator

The **ConversationOrchestrator** manages multi-agent conversations by:
- Registering agents
- Managing conversation state
- Routing messages between agents
- Handling turn-taking and conversation flow

### Specialized Orchestrators

For specific use cases, specialized orchestrators provide optimized workflows:
- **PRReviewOrchestrator**: Multi-agent code review with specialist feedback

## Quick Start

### 1. Basic Agent Response

```python
import asyncio
from llm_orc.orchestration import Agent
from llm_orc.models import OllamaModel
from llm_orc.roles import RoleDefinition

async def simple_agent_example():
    # Create agent
    role = RoleDefinition(
        name="helpful_assistant",
        prompt="You are a helpful assistant that provides clear, concise answers."
    )
    model = OllamaModel(model_name="llama3")
    agent = Agent("assistant", role, model)
    
    # Get response
    response = await agent.respond_to_message("What is machine learning?")
    print(f"Agent: {response}")

# Run example
asyncio.run(simple_agent_example())
```

### 2. Two-Agent Conversation

```python
async def two_agent_conversation():
    # Create agents
    teacher_role = RoleDefinition(
        name="teacher",
        prompt="You are a patient teacher who explains concepts clearly."
    )
    student_role = RoleDefinition(
        name="student", 
        prompt="You are a curious student who asks thoughtful questions."
    )
    
    teacher = Agent("teacher", teacher_role, OllamaModel("llama3"))
    student = Agent("student", student_role, OllamaModel("llama3"))
    
    # Create orchestrator
    orchestrator = ConversationOrchestrator()
    orchestrator.register_agent(teacher)
    orchestrator.register_agent(student)
    
    # Start conversation
    conversation_id = await orchestrator.start_conversation(
        participants=["teacher", "student"],
        topic="Learning about AI"
    )
    
    # Exchange messages
    teacher_response = await orchestrator.send_agent_message(
        sender="student",
        recipient="teacher",
        content="Can you explain what neural networks are?",
        conversation_id=conversation_id
    )
    
    print(f"Teacher: {teacher_response}")

asyncio.run(two_agent_conversation())
```

## Agent Creation

### Role Definition

Roles define an agent's personality and expertise:

```python
# Basic role
basic_role = RoleDefinition(
    name="analyst",
    prompt="You are a data analyst focused on deriving insights from data."
)

# Advanced role with context
advanced_role = RoleDefinition(
    name="security_expert",
    prompt="You are a cybersecurity expert specializing in threat analysis.",
    context={
        "specialties": ["penetration_testing", "vulnerability_assessment"],
        "experience_level": "senior",
        "certifications": ["CISSP", "CEH"]
    }
)
```

### Model Selection

LLM Orchestra supports multiple model providers:

```python
# Ollama (local)
ollama_model = OllamaModel(model_name="llama3", host="http://localhost:11434")

# Claude (API)
claude_model = ClaudeModel(api_key="your-api-key", model="claude-3-5-sonnet-20241022")

# Gemini (API)
gemini_model = GeminiModel(api_key="your-api-key", model="gemini-pro")
```

### Agent Factory Pattern

For creating multiple similar agents:

```python
def create_review_agent(specialty: str, model_name: str = "llama3") -> Agent:
    """Factory function for creating review agents."""
    role = RoleDefinition(
        name=f"{specialty}_reviewer",
        prompt=f"You are a {specialty} specialist providing expert review feedback.",
        context={"specialty": specialty}
    )
    
    model = OllamaModel(model_name=model_name)
    return Agent(f"{specialty}_reviewer", role, model)

# Create multiple reviewers
code_reviewer = create_review_agent("code_quality")
security_reviewer = create_review_agent("security")
performance_reviewer = create_review_agent("performance")
```

## Conversation Orchestration

### Basic Orchestration

```python
async def orchestrate_conversation():
    orchestrator = ConversationOrchestrator()
    
    # Register agents
    orchestrator.register_agent(agent1)
    orchestrator.register_agent(agent2)
    
    # Start conversation
    conversation_id = await orchestrator.start_conversation(
        participants=["agent1", "agent2"],
        topic="Discussion Topic",
        initial_message="Hello! Let's discuss..."
    )
    
    # Send messages
    response = await orchestrator.send_agent_message(
        sender="agent1",
        recipient="agent2",
        content="Your message here",
        conversation_id=conversation_id
    )
    
    return response
```

### Multi-Agent Conversations

```python
async def multi_agent_discussion():
    """Example of 3+ agent conversation."""
    orchestrator = ConversationOrchestrator()
    
    # Register multiple agents
    agents = ["moderator", "expert1", "expert2", "expert3"]
    for agent_name in agents:
        # Create and register each agent
        agent = create_agent(agent_name)
        orchestrator.register_agent(agent)
    
    # Start group conversation
    conversation_id = await orchestrator.start_conversation(
        participants=agents,
        topic="Technical Discussion"
    )
    
    # Moderate discussion
    for round_num in range(3):
        for i, agent in enumerate(agents[1:]):  # Skip moderator
            response = await orchestrator.send_agent_message(
                sender="moderator",
                recipient=agent,
                content=f"Round {round_num + 1}: Share your perspective",
                conversation_id=conversation_id
            )
            print(f"{agent}: {response}")
```

## Specialized Orchestrators

### PR Review Orchestrator

The `PRReviewOrchestrator` enables multi-agent code review:

```python
from llm_orc.orchestration import PRReviewOrchestrator

async def pr_review_example():
    # Create specialist agents
    senior_dev = create_review_agent("senior_developer")
    security_expert = create_review_agent("security_expert")
    ux_reviewer = create_review_agent("ux_reviewer")
    
    # Create orchestrator
    pr_orchestrator = PRReviewOrchestrator()
    pr_orchestrator.register_reviewer(senior_dev)
    pr_orchestrator.register_reviewer(security_expert)
    pr_orchestrator.register_reviewer(ux_reviewer)
    
    # PR data (from GitHub CLI or manual)
    pr_data = {
        "title": "Add user authentication system",
        "description": "Implements JWT-based authentication",
        "diff": "...",  # Code changes
        "files_changed": ["auth.py", "models.py"],
        "additions": 45,
        "deletions": 12
    }
    
    # Conduct review
    results = await pr_orchestrator.review_pr(pr_data)
    
    # Process results
    print(f"PR: {results['pr_title']}")
    print(f"Reviewers: {results['total_reviewers']}")
    print(f"Summary: {results['summary']}")
    
    for review in results['reviews']:
        print(f"{review['reviewer']}: {review['feedback']}")
```

### Custom Orchestrators

Create your own specialized orchestrators:

```python
class BrainstormingOrchestrator:
    """Custom orchestrator for brainstorming sessions."""
    
    def __init__(self):
        self.participants = {}
        self.ideas = []
    
    def register_participant(self, agent: Agent):
        """Register a brainstorming participant."""
        self.participants[agent.name] = agent
    
    async def run_brainstorming_session(self, topic: str, rounds: int = 3):
        """Run structured brainstorming session."""
        session_results = {"topic": topic, "ideas": [], "refinements": []}
        
        # Idea generation phase
        for round_num in range(rounds):
            for name, agent in self.participants.items():
                idea = await agent.respond_to_message(
                    f"Brainstorm creative ideas for: {topic}. Round {round_num + 1}."
                )
                session_results["ideas"].append({"participant": name, "idea": idea})
        
        # Refinement phase
        for idea_entry in session_results["ideas"]:
            for name, agent in self.participants.items():
                if name != idea_entry["participant"]:  # Don't refine your own ideas
                    refinement = await agent.respond_to_message(
                        f"How could this idea be improved: {idea_entry['idea']}"
                    )
                    session_results["refinements"].append({
                        "original_idea": idea_entry["idea"],
                        "refined_by": name,
                        "refinement": refinement
                    })
        
        return session_results
```

## Performance Characteristics

### Message Routing Performance

LLM Orchestra is designed for efficient message routing:

- **Message routing latency**: < 50ms between agents
- **Multi-agent conversations**: < 150ms for 3-message exchanges
- **Agent registration**: < 10ms for 10 agents
- **PR review orchestration**: < 100ms (with mock models)

### Performance Testing

```python
import time
import pytest

@pytest.mark.asyncio
async def test_message_routing_performance():
    """Performance test for message routing."""
    # Setup
    orchestrator = ConversationOrchestrator()
    agent1 = create_fast_agent("agent1")
    agent2 = create_fast_agent("agent2")
    
    orchestrator.register_agent(agent1)
    orchestrator.register_agent(agent2)
    
    conversation_id = await orchestrator.start_conversation(
        participants=["agent1", "agent2"],
        topic="Performance Test"
    )
    
    # Measure
    start_time = time.perf_counter()
    
    await orchestrator.send_agent_message(
        sender="agent1",
        recipient="agent2",
        content="Hello",
        conversation_id=conversation_id
    )
    
    end_time = time.perf_counter()
    
    # Assert
    routing_time_ms = (end_time - start_time) * 1000
    assert routing_time_ms < 50.0
```

### Performance Optimization Tips

1. **Use local models** (Ollama) for faster response times
2. **Batch agent registration** when creating multiple agents
3. **Limit conversation history** for long-running conversations
4. **Use async/await patterns** for concurrent operations
5. **Monitor model response times** and switch models if needed

## Examples

### 1. Shakespeare ↔ Einstein Conversation

```python
# See: examples/shakespeare_einstein_conversation.py
async def historical_dialogue():
    """Conversation between historical figures."""
    # Create Shakespeare agent
    shakespeare = Agent(
        name="shakespeare",
        role=RoleDefinition(
            name="shakespeare",
            prompt="You are William Shakespeare. Speak in eloquent Elizabethan English."
        ),
        model=OllamaModel("llama3")
    )
    
    # Create Einstein agent  
    einstein = Agent(
        name="einstein",
        role=RoleDefinition(
            name="einstein",
            prompt="You are Albert Einstein. Discuss science with wonder and curiosity."
        ),
        model=OllamaModel("llama3")
    )
    
    # Orchestrate conversation
    orchestrator = ConversationOrchestrator()
    orchestrator.register_agent(shakespeare)
    orchestrator.register_agent(einstein)
    
    conversation_id = await orchestrator.start_conversation(
        participants=["shakespeare", "einstein"],
        topic="The Nature of Beauty in Art and Science"
    )
    
    # Shakespeare asks Einstein about beauty
    response = await orchestrator.send_agent_message(
        sender="shakespeare",
        recipient="einstein",
        content="What connection dost thou perceive between mathematical beauty and poetic beauty?",
        conversation_id=conversation_id
    )
    
    return response
```

### 2. GitHub PR Review with CLI Integration

```python
# See: examples/pr_review_with_gh_cli.py
async def review_github_pr(pr_url: str):
    """Review a GitHub PR using specialist agents."""
    # Extract PR data using GitHub CLI
    pr_data = fetch_pr_data_with_gh_cli(pr_url)
    
    # Create specialist reviewers
    senior_dev = create_senior_developer_agent()
    security_expert = create_security_expert_agent()
    ux_reviewer = create_ux_reviewer_agent()
    
    # Setup orchestrator
    pr_orchestrator = PRReviewOrchestrator()
    pr_orchestrator.register_reviewer(senior_dev)
    pr_orchestrator.register_reviewer(security_expert)
    pr_orchestrator.register_reviewer(ux_reviewer)
    
    # Conduct review
    results = await pr_orchestrator.review_pr(pr_data)
    
    # Display results
    print(f"PR: {results['pr_title']}")
    for review in results['reviews']:
        print(f"{review['reviewer']}: {review['feedback']}")
    
    return results
```

### 3. Technical Discussion Panel

```python
async def technical_discussion():
    """Multi-expert technical discussion."""
    # Create expert agents
    experts = {
        "architect": create_expert_agent("software_architect"),
        "security": create_expert_agent("security_specialist"),
        "performance": create_expert_agent("performance_engineer"),
        "ux": create_expert_agent("ux_designer")
    }
    
    orchestrator = ConversationOrchestrator()
    for expert in experts.values():
        orchestrator.register_agent(expert)
    
    # Start discussion
    conversation_id = await orchestrator.start_conversation(
        participants=list(experts.keys()),
        topic="Microservices Architecture Design"
    )
    
    # Facilitate discussion
    topic = "Should we use event-driven architecture for our user service?"
    
    responses = {}
    for expert_name in experts.keys():
        response = await orchestrator.send_agent_message(
            sender="architect",  # Moderator
            recipient=expert_name,
            content=f"From your expertise perspective, {topic}",
            conversation_id=conversation_id
        )
        responses[expert_name] = response
    
    return responses
```

## Best Practices

### 1. Agent Design

- **Clear role definition**: Write specific, focused prompts
- **Appropriate model selection**: Match model capabilities to task complexity
- **Context management**: Use role context for specialized knowledge
- **Conversation history**: Monitor and manage history length

### 2. Orchestration Patterns

- **Single responsibility**: Each orchestrator handles one type of workflow
- **Error handling**: Implement graceful failure and recovery
- **Performance monitoring**: Track message routing and response times
- **Scalability**: Design for multiple concurrent conversations

### 3. Testing Strategies

- **Unit tests**: Test individual agents and orchestrators
- **Integration tests**: Test end-to-end conversations
- **Performance tests**: Validate latency requirements
- **Mock models**: Use fast mocks for testing logic

### 4. Production Considerations

- **Model management**: Handle API rate limits and quotas
- **Conversation state**: Persist important conversation data
- **Monitoring**: Track agent performance and conversation quality
- **Security**: Validate inputs and sanitize outputs

### 5. Debugging Tips

- **Conversation logging**: Log all messages for debugging
- **Agent state inspection**: Monitor conversation history
- **Performance profiling**: Identify bottlenecks in message routing
- **Error tracking**: Capture and analyze orchestration failures

## Integration with eddi-lab

LLM Orchestra is designed to integrate with the eddi-lab ecosystem:

### WebSocket Integration (Issue #5)

```python
# Future: WebSocket integration for real-time communication
class WebSocketOrchestrator(ConversationOrchestrator):
    """Orchestrator with WebSocket support for real-time communication."""
    
    async def broadcast_to_agents(self, message: str, conversation_id: str):
        """Broadcast message to all agents in conversation."""
        # Implementation will connect to eddi-lab WebSocket infrastructure
        pass
```

### MCP Integration (Issues #3, #4)

```python
# Future: MCP (Model Context Protocol) integration
class MCPOrchestrator(ConversationOrchestrator):
    """Orchestrator with MCP support for external resource access."""
    
    async def query_external_resource(self, resource_type: str, query: str):
        """Query external resources via MCP."""
        # Implementation will use MCP client for external data access
        pass
```

## Conclusion

LLM Orchestra's agent orchestration system provides a robust foundation for multi-agent conversations and specialized workflows. The system is designed for:

- **Flexibility**: Support for multiple models and conversation patterns
- **Performance**: Sub-50ms message routing for responsive interactions
- **Extensibility**: Easy creation of custom orchestrators and agents
- **Integration**: Designed to work with eddi-lab ecosystem components

For more examples and advanced usage patterns, see the `examples/` directory and test files.