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

```python
from llm_orc.roles import RoleDefinition, RoleManager
from llm_orc.models import ClaudeModel, ModelManager

# Define agent roles
shakespeare = RoleDefinition(
    name="shakespeare",
    prompt="You are William Shakespeare, the renowned playwright.",
    context={"era": "Elizabethan", "specialties": ["poetry", "drama"]}
)

engineer = RoleDefinition(
    name="engineer", 
    prompt="You are a senior software engineer focused on clean code and TDD."
)

# Set up models
claude = ClaudeModel(api_key="your-api-key")
models = ModelManager()
models.register_model("claude", claude)

# Create role manager
roles = RoleManager()
roles.register_role(shakespeare)
roles.register_role(engineer)
```

## Development Status

Following TDD principles and eddi-lab workflow standards. See [Issues](https://github.com/mrilikecoding/llm-orc/issues) for current development priorities.