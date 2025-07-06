# LLM Orchestra

Multi-agent LLM communication system with flexible role definitions and MCP integration.

## Installation

### For End Users
```bash
pip install llm-orc
```

### For Development
```bash
# Clone the repository
git clone https://github.com/mrilikecoding/llm-orc.git
cd llm-orc

# Install uv (if not already installed)
pip install uv

# Set up development environment
make setup
# or manually: uv sync
```

## Development Workflow

### Running Tests
```bash
make test          # Run all tests with coverage
make red           # Run tests with verbose output (TDD Red phase)
make green         # Run tests with short output (TDD Green phase)  
make refactor      # Run tests + linting (TDD Refactor phase)
```

### Code Quality
```bash
make lint          # Run linting checks
make format        # Format code with black and ruff
```

### Other Commands
```bash
make clean         # Clean build artifacts
```

## Architecture

- **Flexible Role System**: Define custom agent personas (Shakespeare, Einstein, engineer, dancer, etc.)
- **Multi-Model Support**: Claude, Gemini, local models via Ollama
- **MCP Integration**: Client and server capabilities for external resources
- **Real-time Communication**: WebSocket-based agent interaction
- **Extensible Design**: Plugin architecture for custom roles and models

## Usage

### Quick Start: Single Agent

```python
from llm_orc.orchestration import Agent
from llm_orc.models import OllamaModel
from llm_orc.roles import RoleDefinition
import asyncio

async def main():
    # Create agent
    role = RoleDefinition(
        name="assistant",
        prompt="You are a helpful assistant."
    )
    model = OllamaModel(model_name="llama3")
    agent = Agent("assistant", role, model)
    
    # Get response
    response = await agent.respond_to_message("What is machine learning?")
    print(f"Agent: {response}")

asyncio.run(main())
```

### Multi-Agent Conversation

```python
from llm_orc.orchestration import Agent, ConversationOrchestrator

async def conversation_example():
    # Create agents
    shakespeare = Agent("shakespeare", shakespeare_role, ollama_model)
    einstein = Agent("einstein", einstein_role, ollama_model)
    
    # Orchestrate conversation
    orchestrator = ConversationOrchestrator()
    orchestrator.register_agent(shakespeare)
    orchestrator.register_agent(einstein)
    
    conversation_id = await orchestrator.start_conversation(
        participants=["shakespeare", "einstein"],
        topic="Art and Science"
    )
    
    response = await orchestrator.send_agent_message(
        sender="shakespeare",
        recipient="einstein",
        content="What is the nature of beauty in mathematics?",
        conversation_id=conversation_id
    )
    
    print(f"Einstein: {response}")

asyncio.run(conversation_example())
```

### PR Review Example

```python
# Review a GitHub PR with specialist agents
python examples/pr_review_with_gh_cli.py https://github.com/owner/repo/pull/123
```

## Documentation

- **[Agent Orchestration Guide](docs/agent_orchestration.md)** - Comprehensive guide to multi-agent conversations
- **[Examples Directory](examples/)** - Practical examples and use cases
- **[API Reference](src/llm_orc/)** - Core module documentation

## Examples

- `examples/shakespeare_einstein_conversation.py` - Historical figure dialogue
- `examples/pr_review_with_gh_cli.py` - GitHub PR review with specialist agents
- `tests/test_agent_orchestration.py` - Testing patterns and usage examples

## Development Status

Following TDD principles and eddi-lab workflow standards. See [Issues](https://github.com/mrilikecoding/llm-orc/issues) for current development priorities.