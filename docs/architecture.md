# LLM Orchestra Architecture

## Overview

LLM Orchestra is a multi-agent LLM communication system designed for ensemble orchestration and intelligent analysis. The architecture follows a conductor-ensemble pattern where agents can work independently or in complex dependency chains.

## Core Architecture Principles

### Multi-Agent Orchestration
- **Agent Specialization**: Each agent has a specific role and expertise area
- **Dependency-Based Execution**: Agents can depend on other agents using `depends_on` relationships
- **Parallel Execution**: Independent agents run concurrently for optimal performance
- **Automatic Validation**: Circular dependencies and missing dependencies detected at load time

### Performance-First Design
- **Async Parallel Execution**: Uses `asyncio.gather()` for concurrent LLM API calls
- **Resource Optimization**: Mix expensive cloud models with free local models
- **Connection Pooling**: Efficient HTTP client management for API calls
- **Timeout Management**: Per-agent timeout configuration with performance tuning

### Model Abstraction
- **Provider Agnostic**: Support for Anthropic, Google, Ollama, and extensible interfaces
- **Model Profiles**: Named shortcuts combining model + provider + configuration
- **Cost Optimization**: Intelligent routing based on task complexity and model capabilities
- **Authentication Management**: Secure credential storage with OAuth support

## System Components

### Core Execution Engine

#### EnsembleExecutor (`llm_orc/core/execution/ensemble_execution.py`)
- **Phase-based execution**: Resolves dependencies and executes agents in phases
- **Parallel agent coordination**: Multiple agents per phase execute concurrently
- **Result synthesis**: Combines agent outputs according to ensemble configuration
- **Error handling**: Graceful degradation with per-agent error isolation

#### AgentExecutionCoordinator (`llm_orc/core/execution/agent_execution_coordinator.py`)
- **Agent lifecycle management**: Spawns, monitors, and coordinates individual agents
- **Timeout enforcement**: Per-agent timeout with graceful termination
- **Model instance management**: Handles provider-specific model instantiation
- **Performance monitoring**: Usage tracking and timing metrics

#### DependencyResolver (`llm_orc/core/execution/dependency_resolver.py`)
- **Dependency graph analysis**: Topological sorting of agent dependencies
- **Circular dependency detection**: Prevents invalid ensemble configurations
- **Phase optimization**: Groups independent agents for parallel execution
- **Validation**: Ensures all dependencies are satisfied

### Script Agent Infrastructure

#### EnhancedScriptAgent (`llm_orc/agents/enhanced_script_agent.py`)
- **JSON I/O Support**: Structured input/output communication with scripts
- **Script Resolution**: Automatic discovery and path resolution for project scripts
- **Parameter Injection**: Configuration parameters passed as JSON to scripts
- **Context Sharing**: Results from script agents become context for LLM agents

#### ScriptResolver (`llm_orc/core/execution/script_resolver.py`)
- **Script Discovery**: Finds scripts in project `.llm-orc/scripts/` directories
- **Path Resolution**: Resolves relative script paths within project structure
- **Metadata Extraction**: Extracts documentation and parameters from script headers
- **Validation**: Ensures scripts are executable and have proper permissions

#### ArtifactManager (`llm_orc/core/execution/artifact_manager.py`)
- **Execution Persistence**: Saves ensemble results to structured artifact directories
- **Timestamped Storage**: Creates versioned execution records with timestamps
- **Markdown Generation**: Converts execution results to readable markdown reports
- **Latest Symlinks**: Maintains easy access to most recent execution results

### Library Integration

#### LLM Orchestra Library (`llm-orchestra-library/` submodule)
- **Primitive Scripts**: Reusable building blocks for common operations
  - File operations: Read/write files with JSON I/O
  - User interaction: Human-in-the-loop workflows  
  - Data transformation: JSON manipulation and formatting
  - Control flow: Loops, replication, and conditional execution
- **Specialized Scripts**: Domain-specific tools
  - Network science: Topology generation and analysis
  - Research: Statistical tests and experimental design
- **Library Management**: Dynamic source configuration
  - Local submodule support for development
  - Remote GitHub fallback for production
  - Automatic script copying during initialization

### Configuration System

#### EnsembleConfig (`llm_orc/core/config/ensemble_config.py`)
- **YAML-based configuration**: Human-readable ensemble definitions
- **Model profile resolution**: Expands named profiles to full configurations
- **Configuration hierarchy**: Local project configs override global configs
- **Validation**: Schema validation and dependency checking

#### ModelProfiles (`llm_orc/core/config/config_manager.py`)
- **Named configurations**: Simplified model + provider + settings combinations
- **Cost tracking**: Built-in cost information for budget management
- **Override support**: Allow agent-specific overrides of profile defaults
- **Provider availability**: Automatic detection of available providers

### Provider Integration

#### Model Abstractions (`llm_orc/models/`)
- **Base interface**: Common API across all model providers
- **Provider-specific implementations**: Anthropic, Google, Ollama support
- **Authentication handling**: OAuth flows and API key management
- **Response streaming**: Real-time progress updates during execution

#### ModelFactory (`llm_orc/core/models/model_factory.py`)
- **Dynamic instantiation**: Creates model instances based on configurations
- **Provider routing**: Determines appropriate provider for model requests
- **Connection management**: Handles HTTP clients and connection pooling
- **Error recovery**: Fallback strategies for provider failures

## Data Flow Architecture

### Request Processing Pipeline

```
User Input → Ensemble Config → Dependency Resolution → Phase Execution → Result Synthesis
     ↓              ↓                    ↓                   ↓               ↓
   CLI/API    YAML Parser      Topological Sort      Async Parallel     JSON/Text
```

### Agent Execution Flow

#### LLM Agent Flow
```
Agent Config → Model Profile → Provider Instance → LLM API Call → Response Processing
      ↓              ↓              ↓                ↓               ↓
  Validation    Model + Auth    HTTP Client      Stream Handler    Usage Tracking
```

#### Script Agent Flow
```
Agent Config → Script Resolution → Process Execution → JSON I/O → Result Processing
      ↓              ↓                  ↓              ↓             ↓
  Validation    Script Discovery   Subprocess Call   Data Exchange  Usage Tracking
```

#### Agent Type Detection
- **Implicit Detection**: Determined by configuration fields present
  - `script` field → Script agent execution
  - `model_profile` field → LLM agent execution
- **Explicit Override**: `type: script` or `type: llm` field for backward compatibility
- **Hybrid Workflows**: Script agents can provide context data for LLM agents

### Dependency Resolution

```
Ensemble → Agent Dependencies → Dependency Graph → Execution Phases → Parallel Groups
    ↓            ↓                    ↓                 ↓               ↓
   YAML      depends_on fields   Topological Sort   Phase Groups    Async Tasks
```

## Performance Characteristics

### Execution Performance
- **Async Parallel**: 3-15x faster than sequential execution
- **I/O Optimization**: Efficient handling of LLM API latency (1-2 seconds)
- **Resource Efficiency**: Minimal memory overhead (<0.2MB per agent)
- **Scalability**: Performance scales linearly with agent parallelism

### Cost Optimization
- **Model Mixing**: Use free local models for systematic analysis
- **Strategic Routing**: Reserve expensive models for synthesis and strategic insights
- **Usage Tracking**: Real-time token counting and cost estimation
- **Intelligent Defaults**: Cost-effective model profiles for common use cases

## Integration Patterns

### CLI Integration
- **Pipe Support**: `cat code.py | llm-orc invoke code-review`
- **Multiple Output Formats**: Rich, JSON, and text output modes
- **Real-time Streaming**: Progress updates during ensemble execution
- **Configuration Management**: Global and local config hierarchies
- **Script Management**: `scripts list`, `scripts show`, `scripts test` commands for development workflow
- **Artifact Management**: `artifacts list`, `artifacts show` commands for execution result inspection

### API Integration
- **RESTful Interface**: HTTP API for programmatic access
- **WebSocket Support**: Real-time updates for long-running ensembles
- **Batch Processing**: Multiple inputs with shared ensemble configuration
- **Authentication**: API key and OAuth-based access control

### Library Integration
- **Python Package**: Direct import and programmatic usage
- **Configuration Objects**: Programmatic ensemble construction
- **Event Hooks**: Custom handlers for execution lifecycle events
- **Extension Points**: Plugin architecture for custom providers

## Security Architecture

### Credential Management
- **Encrypted Storage**: API keys stored with AES encryption
- **Secure Defaults**: No credentials in configuration files
- **OAuth Integration**: Modern authentication flows for cloud providers
- **Credential Isolation**: Per-provider credential management

### API Security
- **TLS Required**: All external API calls use HTTPS
- **Timeout Protection**: Prevents hanging connections
- **Input Validation**: Sanitization of user inputs and configurations
- **Error Handling**: No credential leakage in error messages

## Extensibility

### Provider Plugins
- **Base Interface**: `llm_orc.models.base.BaseModel` for new providers
- **Registration System**: Dynamic provider discovery and registration
- **Configuration Schema**: Standardized provider configuration format
- **Testing Framework**: Common test patterns for provider validation

### Output Formatters
- **Pluggable Outputs**: Custom result formatters beyond text/JSON
- **Template System**: Configurable output templates
- **Integration Hooks**: Custom processing for specific output needs
- **Streaming Support**: Real-time formatting during execution

### Analysis Extensions
- **Custom Agents**: Domain-specific agent implementations
- **Script Agent Integration**: Execute custom scripts as agents with JSON I/O
- **Analysis Pipelines**: Multi-stage processing workflows combining scripts and LLMs
- **Result Processors**: Custom synthesis and aggregation logic
- **Hybrid Workflows**: Script agents providing context and data for LLM processing
- **Metrics Collection**: Performance and quality measurement hooks

## Testing Architecture

### Unit Testing
- **Pytest Framework**: Comprehensive test coverage across all components
- **Mock Providers**: Test doubles for LLM API calls
- **Configuration Testing**: Validation of YAML configurations and model profiles
- **Error Scenario Testing**: Exception handling and graceful degradation

### Integration Testing
- **Real API Testing**: Validation against actual LLM providers
- **End-to-End Workflows**: Complete ensemble execution testing
- **Performance Testing**: Benchmark parallel execution improvements
- **Configuration Integration**: Test global/local config hierarchies

### Continuous Integration
- **Automated Testing**: Full test suite on every commit
- **Coverage Reporting**: Maintain high test coverage standards
- **Performance Regression**: Detect execution time regressions
- **Security Scanning**: Credential handling and dependency vulnerabilities