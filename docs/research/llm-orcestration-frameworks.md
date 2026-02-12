# LLM orchestration frameworks for local-first, workflow-centric agent coordination

The search for tools architecturally similar to LLM-Orc reveals a fragmented landscape where **no single framework fully combines DAG-based orchestration, declarative configuration, local-first deployment, and hybrid compute**. However, several tools excel in individual dimensions, with **LangGraph**, **Haystack**, **Burr**, and **DSPy** emerging as the closest architectural matches for different use cases. This report maps the ecosystem against LLM-Orc's five core requirements.

---

## Workflow-first orchestration: LangGraph leads, but alternatives emerge

True DAG-based orchestration with explicit dependency graphs remains surprisingly rare in the LLM agent space. Most frameworks default to linear chains, hierarchical delegation, or purely reactive patterns.

**LangGraph** stands out as the most architecturally aligned framework for workflow-first orchestration. It implements a genuine **graph-based state machine** where nodes represent agents or functions, edges define explicit data flow and execution order, and conditional branching enables dynamic routing. The StateGraph abstraction maintains centralized context across workflow steps. Unlike fixed topologies, LangGraph supports cycles (enabling iterative refinement), parallel scatter-gather patterns, and critical path scheduling.

```python
# LangGraph example: explicit DAG with conditional routing
graph = StateGraph(State)
graph.add_node("researcher", research_agent)
graph.add_node("writer", write_agent)
graph.add_conditional_edges("researcher", should_continue, {"continue": "researcher", "finish": "writer"})
```

**Burr** offers the cleanest explicit state machine implementation, using a framework-agnostic approach that works equally well for LLM and non-LLM workflows. Its telemetry UI provides real-time monitoring of state transitions—valuable for debugging complex DAGs. Burr's Hamilton DAG integration enables data pipeline composition alongside agent orchestration.

**Haystack** implements pipeline-based component orchestration where directed graphs connect reusable components (retrievers, generators, embedders). While less flexible than LangGraph for arbitrary workflows, its pipeline abstraction is more intuitive for RAG-centric use cases and **serializes directly to YAML** for portability.

| Framework | Orchestration Model | Explicit DAG Support | Cycles Supported |
|-----------|---------------------|---------------------|------------------|
| **LangGraph** | Graph state machine | ✅ Native | ✅ Yes |
| **Burr** | Explicit state machine | ✅ Native | ✅ Yes |
| **Haystack** | Pipeline graph | ✅ Native | ⚠️ Limited |
| **DSPy** | Module composition | ⚠️ Implicit | Via Python |
| **CrewAI** | Role-based + Flows | ⚠️ Via Flows | ✅ Yes |
| **Temporal** | Durable workflows | ✅ Native | ✅ Yes |

**Temporal** deserves special mention for mission-critical workflows. Its durable execution model survives crashes and restarts, replaying from exact state using recorded event history. OpenAI uses Temporal for Codex, and Replit for their coding agent—evidence of its production readiness for complex agent orchestration.

---

## Local-first deployment: Haystack and LocalAI excel

Local-first deployment requires native integration with Ollama, llama.cpp, MLX, or similar runtimes without mandatory cloud dependencies.

**Haystack** offers the strongest local model integration. Its **default model is Qwen3-0.6B**—evidence that small model support is a first-class concern. The ollama-haystack package provides native Ollama integration, while HuggingFaceLocalChatGenerator enables local inference without Ollama. Models as small as **0.6B parameters** work out of the box.

```python
# Haystack with Ollama
from haystack_integrations.components.generators.ollama import OllamaGenerator
generator = OllamaGenerator(model="qwen3:0.6b", url="http://localhost:11434")
```

**LangGraph** supports Ollama through the langchain-ollama package with models tested down to **1.5B parameters** (DeepSeek R1 1.5B). The integration is mature, though JSON mode may have issues with very small models requiring fallback mechanisms.

**LocalAI** (38.5k GitHub stars) provides a drop-in OpenAI-compatible API that supports llama.cpp, vLLM, MLX, and transformers backends. Its **P2P distributed inference** mode connects multiple LocalAI instances for federated deployment—unique for edge scenarios requiring coordination across devices.

| Tool | Ollama Native | llama.cpp | MLX | Smallest Tested Model |
|------|---------------|-----------|-----|----------------------|
| **Haystack** | ✅ Native pkg | Via Ollama | Via HF | **0.6B** (default) |
| **LangGraph** | ✅ Native pkg | Via Ollama | Via adapter | 1.5B |
| **DSPy** | ✅ LiteLLM | ✅ OpenAI-compat | Via adapter | 770M (paper) |
| **LocalAI** | N/A (is inference) | ✅ Native | ✅ Native | Any |
| **CrewAI** | ✅ LiteLLM | Via LiteLLM | Via LiteLLM | ~7B (struggles below) |

**MLX-specific tools** for Apple Silicon include vllm-mlx (OpenAI-compatible server with continuous batching, **400+ tok/s**) and llm-mlx (CLI tool with 1000+ mlx-community models). Research shows MLX achieves **21-87% higher throughput** than llama.cpp on Apple Silicon.

**Edge deployment research** is advancing rapidly. Academic work on "Sustainable LLM Inference for Edge AI" tested 28 quantized models on Raspberry Pi 4, while LSGLLM-E partitions tasks across multiple lightweight LLMs on edge devices.

---

## Configuration approaches: the code-vs-declarative divide

The ecosystem splits sharply between code-first frameworks (LangGraph, DSPy, CrewAI) and declarative approaches (Kestra, Haystack YAML export, Fractalic).

**Kestra** offers the most fully declarative approach with **YAML-defined workflows** and an AI copilot that generates workflow YAML from natural language. Its Agent plugin supports autonomous LLM-powered processes defined entirely in configuration:

```yaml
id: multi_agent_workflow
namespace: company.ai
tasks:
  - id: research_agent
    type: io.kestra.plugin.ai.agent.Agent
    systemMessage: "You are a research specialist..."
    tools: [web_search, document_reader]
  - id: writer_agent
    type: io.kestra.plugin.ai.agent.Agent
    dependsOn: [research_agent]
```

**CrewAI** supports a hybrid approach: agents and tasks defined in **YAML files**, with crew orchestration logic in Python. This separation allows non-developers to modify agent behaviors while developers manage workflow structure:

```yaml
# agents.yaml
researcher:
  role: "{topic} Senior Researcher"
  goal: "Uncover cutting-edge developments"
  backstory: "You're a seasoned researcher..."
```

**Haystack** pipelines serialize to YAML via `pipeline.to_yaml()` and can export as Python code via `pipeline.to_code()`. The deepset Studio enterprise product provides visual pipeline editing.

**DSPy's declarative signatures** offer a middle ground—natural language module declarations that abstract away prompt engineering while remaining embedded in Python:

```python
class ResearchAgent(dspy.Signature):
    """Analyze documents and extract key findings."""
    documents: list[str] = dspy.InputField()
    findings: ResearchReport = dspy.OutputField()
```

**Fractalic** represents an emerging declarative approach: YAML/Markdown-based AI workflow definition with Git-native version tracking—worth monitoring for local-first orchestration.

---

## Multi-agent coordination patterns

Multi-agent coordination architectures vary significantly in how specialist agents communicate and delegate.

**CrewAI** implements **role-based multi-agent collaboration** where agents have explicit roles, goals, and backstories. Its hierarchical process mode enables a manager agent to coordinate worker delegation. The framework supports **43,200+ GitHub stars** and has trained 100,000+ certified developers, making it the most popular multi-agent framework.

**LangGraph** supports multi-agent patterns through explicit graph composition—a supervisor node routes to specialist worker nodes based on state. This is more flexible than CrewAI's role abstraction but requires more implementation effort.

**Langroid** (used in production by Nullify for software security) offers task-based hierarchical orchestration with native Ollama support. Its Agent→Task wrapping enables clean delegation patterns.

**LlamaIndex AgentWorkflow** provides event-driven, asyncio-based multi-agent orchestration with a distributed microservice architecture: message queue + control plane + orchestrator. Human-in-the-loop support is built in.

**Research patterns** worth noting:
- **Blackboard Architecture**: Agents communicate via shared state (gpt-multi-atomic-agents implements this)
- **Swarm Intelligence**: Multi-LLM systems with emergent collective behavior (MetaGPT, AgentVerse)
- **Cascaded Inference**: Route easy queries to small models, escalate hard cases to larger models

The **Strategic Coordination Framework for Small LLMs** (2024-2025 research) demonstrates that SLM coordination can match large LLM performance in data synthesis tasks—validating the micro-LLM ensemble approach.

---

## Hybrid compute integration

Mixing LLM agents with scripts, tools, and deterministic compute is essential for production workflows.

**LangGraph** handles this cleanly—graph nodes can be pure Python functions (non-LLM) alongside LLM-powered agents. The ReAct agent pattern interleaves reasoning and tool execution natively.

**Haystack** excels here through **ComponentTool**—any Haystack component (retriever, web search, calculator) can be wrapped as a tool for agents. Pipelines can mix embedders, retrievers, LLM generators, and custom Python components:

```python
# Wrap any component as a tool
web_tool = ComponentTool(
    component=SerperDevWebSearch(top_k=3),
    name="web_search"
)
agent = Agent(chat_generator=generator, tools=[web_tool, calculator])
```

**Temporal** treats all non-deterministic operations (LLM calls, API requests) as Activities while keeping workflow orchestration deterministic. This separation enables automatic retries, rate limiting, and crash recovery for hybrid compute pipelines.

**DSPy** integrates tools via the `@agent.tool` decorator pattern with Pydantic validation. The `dspy.PythonInterpreter` enables code execution within workflows.

**MCP (Model Context Protocol)** is becoming the standard for tool integration across platforms. LocalAI, Open WebUI, Jan.ai, and Temporal all support MCP, enabling interoperable tool calling.

---

## Development activity and production readiness

| Framework | GitHub Stars | Enterprise Backing | License | Production Evidence |
|-----------|-------------|-------------------|---------|---------------------|
| **LangGraph** | 21K | LangChain Inc | MIT | Extensive tutorials, LangGraph Studio |
| **CrewAI** | 43K | CrewAI Inc | MIT | 100K certified developers |
| **Haystack** | 23K | deepset GmbH | Apache 2.0 | Airbus, Netflix, Intel, Apple |
| **Temporal** | High | Temporal Inc | MIT | **OpenAI Codex, Replit Agent** |
| **DSPy** | Growing | Stanford NLP | MIT | ICLR 2024 paper, DeepLearning.AI course |
| **LocalAI** | 38.5K | Community | MIT | 159 contributors |
| **Burr** | Growing | Apache Foundation | Apache 2.0 | Framework-agnostic design |

**Temporal's enterprise adoption** for AI workloads (OpenAI, Replit, Retool) signals that durable execution patterns are becoming critical for production agent systems.

---

## Architectural patterns for LLM-Orc development

Based on this analysis, several patterns emerge as valuable for local-first, data-sovereign micro-LLM orchestration:

**From LangGraph**: The StateGraph abstraction with typed state dictionaries and conditional edge routing provides the cleanest DAG implementation. Consider adopting its pattern of separating state schema from execution logic.

**From Haystack**: Native small model defaults (0.6B) and YAML pipeline serialization offer a model for local-first design. The ComponentTool pattern elegantly bridges structured computation and agent capabilities.

**From DSPy**: Declarative signatures that separate interface from implementation enable automatic optimization for small models. DSPy's optimizers (MIPROv2, BootstrapFinetune) **specifically improve small model performance**—showing 8% to 40%+ recall improvements.

**From Burr**: Explicit state machines with framework-agnostic design and built-in telemetry provide debuggability that complex DAGs require. Its approach of treating LLM and non-LLM workflows identically enables true hybrid compute.

**From LocalAI**: P2P distributed inference for federated edge deployment, where multiple lightweight instances coordinate across devices.

**From academic research**: The **Division-Fusion paradigm** (LLM as planner, SLMs as specialized executors) and **cascaded inference** (complexity-based routing) directly inform micro-LLM ensemble design.

---

## Tools excluded or briefly covered

Per the request, the following were excluded or minimally covered:
- **optillm**: Inference optimization layer, not orchestration-focused
- **Together AI MoA**: Cloud-focused mixture of agents, not local-first
- **RouteLLM**: Single-query routing, not workflow orchestration
- **LLM-Blender**: Ensemble output fusion, not multi-agent coordination

---

## Conclusion: gaps and opportunities

The local-first, DAG-based orchestration space remains **underserved by existing tools**. LangGraph offers the best DAG primitives but is code-first; Kestra offers the best declarative configuration but lacks deep local model optimization; Haystack excels at local deployment but focuses on RAG pipelines rather than general agent workflows.

**Key gaps** that LLM-Orc could address:
1. **Unified declarative DAG + local-first design**: No tool combines YAML workflow definition with native Ollama/llama.cpp optimization
2. **Micro-LLM routing**: Most frameworks assume capable models; few optimize for small model coordination
3. **Privacy-preserving multi-agent**: Limited research on encrypted or air-gapped multi-agent coordination
4. **Visual DAG editing for local tools**: Code-first frameworks lack visual builders; visual tools lack local optimization

The architectural direction is clear: **explicit state machines** (Burr/LangGraph) + **declarative configuration** (Kestra/Haystack YAML) + **small model optimization** (DSPy) + **P2P edge coordination** (LocalAI) would represent a novel synthesis currently unavailable in any single framework.