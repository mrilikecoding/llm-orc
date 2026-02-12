# Performant Algorithms for LLM Orchestration: A Technical Reference

Five canonical algorithms enable small language models to achieve performance approaching or exceeding larger models: **self-consistency sampling** boosts reasoning accuracy by 6-18% through majority voting across diverse reasoning paths; **confidence-based routing** reduces costs by 35-85% while maintaining quality; **capability-aware task routing** matches task complexity to appropriate model tiers; **OpenTelemetry-based DAG telemetry** provides production-grade observability; and **DSPy prompt optimization** improves small model performance by 8-40% through automated instruction and few-shot tuning. These techniques are particularly valuable for local-first deployments where maximizing performance per parameter is essential.

---

## Priority 1: Self-Consistency Sampling

**+17.9% on GSM8K through diversity**

### Background

The self-consistency algorithm, introduced by Wang et al. (2022/2023) in "Self-Consistency Improves Chain of Thought Reasoning in Language Models," exploits a fundamental insight: complex reasoning problems typically admit multiple reasoning paths that converge on the same correct answer. Rather than using greedy decoding, the algorithm samples **N diverse reasoning paths** at temperature T > 0, extracts the final answer from each path, and selects the most frequent answer via majority voting.

### Performance Benchmarks

| Benchmark | Standard CoT | Self-Consistency | Improvement |
|-----------|--------------|------------------|-------------|
| GSM8K | 56.5% | 74.4% | **+17.9%** |
| SVAMP | - | - | **+11.0%** |
| AQuA | - | - | **+12.2%** |
| StrategyQA | - | - | **+6.4%** |

### Optimal Parameters

| Parameter | Recommendation | Notes |
|-----------|----------------|-------|
| **N (samples)** | 20-40 | Most gains by N=20; diminishing returns after N=40 |
| **Temperature** | 0.7 | Most robust; use 0.5 with fewer samples |
| **Simple tasks** | N=5-10 | Classification, extraction |
| **Math reasoning** | N=20-40 | GSM8K, arithmetic |
| **Hard problems** | N=50-256 | MATH competition |

### Early Stopping Strategy

Early-Stopping Self-Consistency (ESC) reduces sampling requirements by **33.8% to 84.2%** while maintaining comparable accuracy:

- 80.1% fewer samples on GSM8K
- 76.8% fewer samples on StrategyQA

### Algorithm Pseudocode

```python
def self_consistency_with_early_stopping(
    question: str,
    model: LLM,
    max_samples: int = 40,
    window_size: int = 5,
    temperature: float = 0.7
) -> str:
    """
    Self-consistency sampling with early stopping.
    
    Args:
        question: The input question/prompt
        model: Language model to sample from
        max_samples: Maximum number of reasoning paths to generate
        window_size: Number of consecutive matching answers to trigger early stop
        temperature: Sampling temperature (higher = more diverse)
    
    Returns:
        Most consistent answer across samples
    """
    answers = []
    reasoning_paths = []
    
    for i in range(max_samples):
        # Generate diverse reasoning path
        response = model.generate(
            prompt=format_cot_prompt(question),
            temperature=temperature,
            max_tokens=512
        )
        
        # Extract final answer using regex
        # Pattern matches "The answer is X" or "#### X" formats
        answer = extract_answer(response)
        
        answers.append(answer)
        reasoning_paths.append(response)
        
        # Early stopping: check if last W answers are unanimous
        if len(answers) >= window_size:
            window = answers[-window_size:]
            if len(set(window)) == 1:
                return window[0]
    
    # Return majority vote
    return majority_vote(answers)


def extract_answer(response: str) -> str:
    """Extract final answer from CoT response."""
    import re
    
    # Try common answer patterns
    patterns = [
        r"[Tt]he answer is[:\s]*([^\n\.]+)",
        r"####\s*([^\n]+)",
        r"[Aa]nswer[:\s]*([^\n\.]+)",
        r"=\s*([0-9\.\-]+)\s*$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
    
    # Fallback: return last line
    return response.strip().split('\n')[-1]


def majority_vote(answers: list[str]) -> str:
    """Return most frequent answer."""
    from collections import Counter
    counts = Counter(answers)
    return counts.most_common(1)[0][0]


def format_cot_prompt(question: str) -> str:
    """Format question with chain-of-thought instruction."""
    return f"""Solve this problem step by step, then provide the final answer.

Question: {question}

Let's think through this carefully:"""
```

### Small Model Adaptations

For models under 7B parameters:

- Use higher sample counts (N=64+)
- Lower temperature (T=0.5) due to higher variance
- Combine with Tree-PLV verification for additional gains
- Mistral 7B: 67.55% → **82.79%** with self-consistency + Tree-PLV

### Reference Implementation

The [optillm](https://github.com/codelion/optillm) framework implements advanced self-consistency with clustering, using k=6 candidate solutions by default.

---

## Priority 2: Confidence-Based Output Selection

**35-85% cost reduction via calibrated logprob scoring**

### Background

Confidence-based routing encompasses two complementary approaches:

1. **SLM-MUX**: Selecting among multiple small models based on confidence
2. **RouteLLM/FrugalGPT**: Routing between cheap/expensive model tiers

### Performance Benchmarks

**SLM-MUX (October 2025):**
- Up to **13.4% improvement on MATH**
- **8.8% improvement on GPQA**
- Two SLMs together outperform Qwen 2.5 72B on certain benchmarks

**RouteLLM (LMSYS, ICLR 2025):**

| Benchmark | Cost Reduction | Quality Retained |
|-----------|----------------|------------------|
| MT-Bench | **85%** | 95% of GPT-4 |
| MMLU | **45%** | 95% of GPT-4 |
| GSM8K | **35%** | 95% of GPT-4 |

**FrugalGPT:**
- Up to **98% cost reduction** while matching GPT-4 on certain tasks

### Confidence Scoring Methods

#### Logprob-Based Scoring

```python
def compute_confidence_logprob(
    logprobs: list[float],
    method: str = "mean"
) -> float:
    """
    Compute confidence score from token log probabilities.
    
    Args:
        logprobs: List of log probabilities for each token
        method: Aggregation method - "mean", "min", or "sequence"
    
    Returns:
        Confidence score in [0, 1]
    """
    import math
    
    if not logprobs:
        return 0.0
    
    probs = [math.exp(lp) for lp in logprobs]
    
    if method == "mean":
        # Average token probability - most stable
        return sum(probs) / len(probs)
    
    elif method == "min":
        # Minimum probability - captures weakest link
        return min(probs)
    
    elif method == "sequence":
        # Product of probabilities - penalizes length
        result = 1.0
        for p in probs:
            result *= p
        return result
    
    else:
        raise ValueError(f"Unknown method: {method}")
```

#### Self-Consistency Fallback (When Logprobs Unavailable)

```python
def compute_confidence_sampling(
    model: LLM,
    prompt: str,
    n_samples: int = 3,
    temperature: float = 0.3
) -> tuple[str, float]:
    """
    Compute confidence via sampling consistency.
    Use when logprobs are unavailable (e.g., some Ollama models).
    
    Returns:
        (most_common_answer, confidence_score)
    """
    from collections import Counter
    
    samples = []
    for _ in range(n_samples):
        response = model.generate(prompt, temperature=temperature)
        answer = extract_answer(response)
        samples.append(answer)
    
    counts = Counter(samples)
    most_common, frequency = counts.most_common(1)[0]
    confidence = frequency / n_samples
    
    return most_common, confidence
```

### Confidence Thresholds

| Confidence | Action |
|------------|--------|
| > 90% | Accept response directly |
| 50-90% | Trigger verification or self-consistency |
| < 50% | Escalate to stronger model or human review |

### Temperature Scaling for Calibration

Small models exhibit significant overconfidence. Apply temperature scaling:

```python
def calibrate_confidence(
    logits: list[float],
    calibration_temperature: float = 2.0
) -> list[float]:
    """
    Apply temperature scaling to improve calibration.
    
    Typical calibration temperatures: 1.5-3.0
    Target ECE (Expected Calibration Error): < 0.1
    """
    import math
    
    # Apply temperature
    scaled = [l / calibration_temperature for l in logits]
    
    # Softmax
    max_val = max(scaled)
    exp_vals = [math.exp(s - max_val) for s in scaled]
    total = sum(exp_vals)
    
    return [e / total for e in exp_vals]
```

### Cascaded Routing Algorithm

```python
class CascadedRouter:
    """
    Route queries through models from cheapest to most expensive,
    stopping when confidence threshold is met.
    """
    
    def __init__(
        self,
        models: list[tuple[str, LLM, float]],  # (name, model, cost_per_token)
        confidence_threshold: float = 0.85
    ):
        # Sort by cost, cheapest first
        self.models = sorted(models, key=lambda x: x[2])
        self.threshold = confidence_threshold
    
    def route(self, query: str) -> dict:
        """
        Route query through cascade.
        
        Returns:
            {
                "response": str,
                "model_used": str,
                "confidence": float,
                "total_cost": float,
                "models_tried": list[str]
            }
        """
        models_tried = []
        total_cost = 0.0
        
        for name, model, cost in self.models:
            models_tried.append(name)
            
            # Generate response
            response, tokens = model.generate_with_usage(query)
            total_cost += tokens * cost
            
            # Compute confidence
            if hasattr(model, 'get_logprobs'):
                confidence = compute_confidence_logprob(
                    model.get_logprobs(),
                    method="mean"
                )
            else:
                _, confidence = compute_confidence_sampling(
                    model, query, n_samples=3
                )
            
            # Check threshold
            if confidence >= self.threshold:
                return {
                    "response": response,
                    "model_used": name,
                    "confidence": confidence,
                    "total_cost": total_cost,
                    "models_tried": models_tried
                }
        
        # Return last (most expensive) model's response
        return {
            "response": response,
            "model_used": name,
            "confidence": confidence,
            "total_cost": total_cost,
            "models_tried": models_tried
        }
```

### Ollama Logprobs Support

As of Ollama v0.12.11, logprobs are supported:

```python
import ollama

response = ollama.generate(
    model="llama3.2",
    prompt="What is 2+2?",
    options={"logprobs": True}
)

# Access logprobs
logprobs = response.get("logprobs", [])
```

### Reference Implementations

- [RouteLLM](https://github.com/lm-sys/RouteLLM) - LMSYS routing framework
- [SLM-MUX](https://arxiv.org/html/2510.05077v1) - Small model orchestration

---

## Priority 3: Model Capability Assessment and Task Routing

**Understanding the 3B reasoning threshold**

### The 3B Parameter Threshold

Research consistently identifies **3B parameters as a critical minimum** for basic reasoning:

| Model | Parameters | Achievement |
|-------|------------|-------------|
| TinyZero | 3B | Emergent self-verification via RL |
| SmolLM3 | 3B | Competitive with 4B on commonsense |
| BTLM-3B-8K | 3B | 7B-level performance |
| Stable Code 3B | 3B | Matches CodeLLaMA 7B (60% fewer params) |

### Task Routing Taxonomy

| Model Size | Suitable Tasks | Example Use Cases |
|------------|----------------|-------------------|
| **1B-3B** | Simple classification, extraction, moderation | Sentiment, keywords, content filtering |
| **3B-7B** | Basic Q&A, well-defined instructions | FAQ, structured data extraction |
| **7B-13B** | Code generation, moderate reasoning | Code completion, summarization |
| **13B-30B** | Complex analysis, multi-step reasoning | Research synthesis, code review |
| **70B+** | Expert-level sophisticated reasoning | Complex math, nuanced analysis |

**Scaling insight:** Moving from 7B to 70B typically improves reasoning by **15-30%** but increases costs **8-10x**.

### Task Complexity Classification

```python
from enum import Enum
from dataclasses import dataclass

class CognitiveLevel(Enum):
    """Bloom's Taxonomy levels - LLMs perform best at lower levels."""
    REMEMBER = 1      # Recall facts
    UNDERSTAND = 2    # Explain concepts
    APPLY = 3         # Use in new situations
    ANALYZE = 4       # Break down, compare
    EVALUATE = 5      # Judge, critique
    CREATE = 6        # Generate novel content


class TaskType(Enum):
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    QA_SIMPLE = "qa_simple"
    QA_COMPLEX = "qa_complex"
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    CREATIVE = "creative"
    ANALYSIS = "analysis"


@dataclass
class TaskComplexity:
    """Assessment of task complexity for routing."""
    task_type: TaskType
    cognitive_level: CognitiveLevel
    estimated_steps: int          # Reasoning steps required
    requires_world_knowledge: bool
    requires_code_execution: bool
    min_recommended_params: float  # In billions
    
    @property
    def complexity_score(self) -> float:
        """Compute overall complexity score 0-1."""
        score = 0.0
        score += self.cognitive_level.value / 6 * 0.3
        score += min(self.estimated_steps / 10, 1.0) * 0.3
        score += 0.2 if self.requires_world_knowledge else 0
        score += 0.2 if self.requires_code_execution else 0
        return score


def classify_task(query: str) -> TaskComplexity:
    """
    Classify task complexity for routing decisions.
    
    This is a heuristic classifier - production systems should
    use a trained classifier or LLM-based assessment.
    """
    query_lower = query.lower()
    
    # Simple keyword heuristics
    if any(w in query_lower for w in ["classify", "is this", "yes or no"]):
        return TaskComplexity(
            task_type=TaskType.CLASSIFICATION,
            cognitive_level=CognitiveLevel.UNDERSTAND,
            estimated_steps=1,
            requires_world_knowledge=False,
            requires_code_execution=False,
            min_recommended_params=1.0
        )
    
    if any(w in query_lower for w in ["extract", "find all", "list the"]):
        return TaskComplexity(
            task_type=TaskType.EXTRACTION,
            cognitive_level=CognitiveLevel.UNDERSTAND,
            estimated_steps=2,
            requires_world_knowledge=False,
            requires_code_execution=False,
            min_recommended_params=3.0
        )
    
    if any(w in query_lower for w in ["write code", "implement", "function"]):
        return TaskComplexity(
            task_type=TaskType.CODE_GENERATION,
            cognitive_level=CognitiveLevel.CREATE,
            estimated_steps=5,
            requires_world_knowledge=False,
            requires_code_execution=True,
            min_recommended_params=7.0
        )
    
    if any(w in query_lower for w in ["analyze", "compare", "evaluate", "why"]):
        return TaskComplexity(
            task_type=TaskType.ANALYSIS,
            cognitive_level=CognitiveLevel.ANALYZE,
            estimated_steps=4,
            requires_world_knowledge=True,
            requires_code_execution=False,
            min_recommended_params=13.0
        )
    
    if any(w in query_lower for w in ["solve", "calculate", "prove"]):
        return TaskComplexity(
            task_type=TaskType.REASONING,
            cognitive_level=CognitiveLevel.APPLY,
            estimated_steps=6,
            requires_world_knowledge=False,
            requires_code_execution=False,
            min_recommended_params=7.0
        )
    
    # Default: moderate complexity
    return TaskComplexity(
        task_type=TaskType.QA_COMPLEX,
        cognitive_level=CognitiveLevel.UNDERSTAND,
        estimated_steps=3,
        requires_world_knowledge=True,
        requires_code_execution=False,
        min_recommended_params=7.0
    )
```

### Task Decomposition Strategy

```python
def should_decompose(task: TaskComplexity) -> bool:
    """
    Determine if task should be decomposed into subtasks.
    
    Decomposition helps when:
    - Task complexity is high
    - Subtasks have known solutions
    - Parallelization is possible
    
    Decomposition hurts when:
    - Coordination overhead exceeds benefits
    - Task requires holistic understanding
    """
    return (
        task.complexity_score > 0.6 and
        task.estimated_steps > 3 and
        task.cognitive_level.value >= CognitiveLevel.ANALYZE.value
    )


def decompose_task(query: str, model: LLM) -> list[str]:
    """
    Use LLM to decompose complex task into simpler subtasks.
    """
    decomposition_prompt = f"""Break down this task into 2-4 simpler subtasks that can be solved independently.

Task: {query}

Return ONLY a numbered list of subtasks, nothing else:"""
    
    response = model.generate(decomposition_prompt, temperature=0.3)
    
    # Parse numbered list
    subtasks = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove number prefix
            subtask = line.lstrip('0123456789.)-: ')
            if subtask:
                subtasks.append(subtask)
    
    return subtasks
```

### Capability Guardrails

```python
@dataclass
class ModelCapability:
    """Model capability metadata for routing."""
    name: str
    parameters_b: float  # Billions
    context_length: int
    strengths: list[TaskType]
    weaknesses: list[TaskType]


def check_capability_match(
    task: TaskComplexity,
    model: ModelCapability
) -> tuple[bool, str]:
    """
    Check if model is appropriate for task.
    
    Returns:
        (is_capable, warning_message)
    """
    warnings = []
    
    # Check parameter threshold
    if model.parameters_b < task.min_recommended_params:
        warnings.append(
            f"Model {model.name} ({model.parameters_b}B) may struggle with "
            f"this task (recommended: {task.min_recommended_params}B+)"
        )
    
    # Check known weaknesses
    if task.task_type in model.weaknesses:
        warnings.append(
            f"Task type '{task.task_type.value}' is a known weakness "
            f"of {model.name}"
        )
    
    # Check cognitive level vs size
    if (task.cognitive_level.value >= CognitiveLevel.ANALYZE.value and 
        model.parameters_b < 7):
        warnings.append(
            f"Complex reasoning (level {task.cognitive_level.name}) "
            f"typically requires 7B+ parameters"
        )
    
    is_capable = len(warnings) == 0
    warning_msg = "; ".join(warnings) if warnings else ""
    
    return is_capable, warning_msg
```

### Reference: Emergent Capabilities

Wei et al. (2022) documented **137 emergent abilities**—capabilities that appear at larger scales:

- Chain-of-thought prompting emerges at ~10²² FLOPs
- Multi-step arithmetic emerges at ~10²³ FLOPs
- Word unscrambling shows sharp emergence

Note: Schaeffer et al. (2023) argue apparent emergence may be a measurement artifact. Practical implication: proper prompting can unlock latent abilities in smaller models.

---

## Priority 4: DAG Execution Telemetry and Visualization

**OpenTelemetry GenAI conventions and Sugiyama layouts**

### OpenTelemetry Semantic Conventions (v1.37+)

```python
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Standard GenAI span attributes
GENAI_ATTRIBUTES = {
    "gen_ai.system": "openai",              # Provider name
    "gen_ai.request.model": "gpt-4o",       # Requested model
    "gen_ai.response.model": "gpt-4o-2024", # Actual model used
    "gen_ai.request.max_tokens": 1024,
    "gen_ai.request.temperature": 0.7,
    "gen_ai.usage.input_tokens": 150,
    "gen_ai.usage.output_tokens": 200,
    "gen_ai.usage.total_tokens": 350,
}

# Span naming convention: "{operation} {model}"
# Examples: "chat gpt-4o", "completion llama3.2"


class LLMTelemetry:
    """OpenTelemetry instrumentation for LLM calls."""
    
    def __init__(self, service_name: str = "llm-orc"):
        provider = TracerProvider()
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(service_name)
    
    def trace_llm_call(
        self,
        model: str,
        operation: str = "chat"
    ):
        """Context manager for tracing LLM calls."""
        return self.tracer.start_as_current_span(
            f"{operation} {model}",
            kind=SpanKind.CLIENT,
            attributes={
                "gen_ai.system": self._infer_system(model),
                "gen_ai.request.model": model,
                "gen_ai.operation.name": operation,
            }
        )
    
    def record_usage(
        self,
        span,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = None
    ):
        """Record token usage on span."""
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("gen_ai.usage.total_tokens", input_tokens + output_tokens)
        if cost_usd is not None:
            span.set_attribute("gen_ai.cost.total_usd", cost_usd)
    
    def _infer_system(self, model: str) -> str:
        if "gpt" in model.lower():
            return "openai"
        if "claude" in model.lower():
            return "anthropic"
        if "llama" in model.lower() or "mistral" in model.lower():
            return "ollama"
        return "unknown"
```

### Cost Tracking Implementation

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json

# Pricing per 1M tokens (as of early 2025)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    
    # Anthropic
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    
    # Local (Ollama) - effectively free
    "llama3.2": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "qwen2.5": {"input": 0.0, "output": 0.0},
}


@dataclass
class UsageRecord:
    """Record of a single LLM call."""
    timestamp: datetime
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    agent_name: Optional[str] = None
    ensemble_name: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "agent_name": self.agent_name,
            "ensemble_name": self.ensemble_name,
        }


class CostTracker:
    """Track LLM usage and costs across ensemble execution."""
    
    def __init__(self):
        self.records: list[UsageRecord] = []
    
    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        agent_name: str = None,
        ensemble_name: str = None
    ) -> UsageRecord:
        """Record a single LLM call."""
        pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
        cost = (
            input_tokens * pricing["input"] / 1_000_000 +
            output_tokens * pricing["output"] / 1_000_000
        )
        
        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            provider=self._infer_provider(model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            agent_name=agent_name,
            ensemble_name=ensemble_name,
        )
        
        self.records.append(record)
        return record
    
    def summary(self) -> dict:
        """Get usage summary."""
        if not self.records:
            return {"total_cost": 0, "total_tokens": 0, "calls": 0}
        
        return {
            "total_cost_usd": sum(r.cost_usd for r in self.records),
            "total_input_tokens": sum(r.input_tokens for r in self.records),
            "total_output_tokens": sum(r.output_tokens for r in self.records),
            "total_calls": len(self.records),
            "by_model": self._group_by("model"),
            "by_agent": self._group_by("agent_name"),
        }
    
    def _group_by(self, field: str) -> dict:
        groups = {}
        for r in self.records:
            key = getattr(r, field) or "unknown"
            if key not in groups:
                groups[key] = {"cost": 0, "tokens": 0, "calls": 0}
            groups[key]["cost"] += r.cost_usd
            groups[key]["tokens"] += r.input_tokens + r.output_tokens
            groups[key]["calls"] += 1
        return groups
    
    def _infer_provider(self, model: str) -> str:
        if "gpt" in model.lower():
            return "openai"
        if "claude" in model.lower():
            return "anthropic"
        return "ollama"
```

### DAG Visualization with Sugiyama Algorithm

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


class NodeState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class DAGNode:
    """Node in execution DAG."""
    id: str
    name: str
    type: str  # "llm_agent" or "script_agent"
    state: NodeState = NodeState.PENDING
    layer: int = 0  # Assigned by Sugiyama algorithm
    x: float = 0.0  # Final x coordinate
    y: float = 0.0  # Final y coordinate
    dependencies: list[str] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[dict] = None


@dataclass
class DAGEdge:
    """Edge in execution DAG."""
    source: str
    target: str


class DAGLayout:
    """
    Sugiyama-style hierarchical DAG layout.
    
    Algorithm:
    1. Assign layers (topological sort)
    2. Minimize edge crossings within layers
    3. Assign x coordinates
    """
    
    def __init__(self, nodes: list[DAGNode], edges: list[DAGEdge]):
        self.nodes = {n.id: n for n in nodes}
        self.edges = edges
        self.layers: list[list[str]] = []
    
    def layout(self, layer_height: float = 100, node_spacing: float = 150):
        """Compute layout coordinates for all nodes."""
        self._assign_layers()
        self._minimize_crossings()
        self._assign_coordinates(layer_height, node_spacing)
    
    def _assign_layers(self):
        """Assign nodes to layers via topological sort."""
        # Build adjacency and in-degree
        in_degree = {nid: 0 for nid in self.nodes}
        adj = {nid: [] for nid in self.nodes}
        
        for edge in self.edges:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1
        
        # Kahn's algorithm for topological sort
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        layer_assignment = {}
        
        while queue:
            # All nodes in queue go to current layer
            current_layer = []
            next_queue = []
            
            for nid in queue:
                layer_assignment[nid] = len(self.layers)
                current_layer.append(nid)
                
                for target in adj[nid]:
                    in_degree[target] -= 1
                    if in_degree[target] == 0:
                        next_queue.append(target)
            
            self.layers.append(current_layer)
            queue = next_queue
        
        # Update node layer assignments
        for nid, layer in layer_assignment.items():
            self.nodes[nid].layer = layer
    
    def _minimize_crossings(self):
        """
        Minimize edge crossings using barycenter heuristic.
        """
        for _ in range(4):  # Iterate a few times
            # Forward pass
            for i in range(1, len(self.layers)):
                self._reorder_layer(i, self.layers[i-1])
            
            # Backward pass
            for i in range(len(self.layers) - 2, -1, -1):
                self._reorder_layer(i, self.layers[i+1])
    
    def _reorder_layer(self, layer_idx: int, reference_layer: list[str]):
        """Reorder layer based on barycenter of connected nodes."""
        layer = self.layers[layer_idx]
        ref_positions = {nid: i for i, nid in enumerate(reference_layer)}
        
        # Compute barycenter for each node
        barycenters = {}
        for nid in layer:
            connected = []
            for edge in self.edges:
                if edge.source == nid and edge.target in ref_positions:
                    connected.append(ref_positions[edge.target])
                elif edge.target == nid and edge.source in ref_positions:
                    connected.append(ref_positions[edge.source])
            
            if connected:
                barycenters[nid] = sum(connected) / len(connected)
            else:
                barycenters[nid] = float('inf')
        
        # Sort by barycenter
        self.layers[layer_idx] = sorted(layer, key=lambda n: barycenters[n])
    
    def _assign_coordinates(self, layer_height: float, node_spacing: float):
        """Assign final x, y coordinates."""
        for layer_idx, layer in enumerate(self.layers):
            y = layer_idx * layer_height
            total_width = (len(layer) - 1) * node_spacing
            start_x = -total_width / 2
            
            for i, nid in enumerate(layer):
                self.nodes[nid].x = start_x + i * node_spacing
                self.nodes[nid].y = y
    
    def to_json(self) -> str:
        """Export layout to JSON for visualization."""
        return json.dumps({
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "type": n.type,
                    "state": n.state.value,
                    "x": n.x,
                    "y": n.y,
                    "layer": n.layer,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target}
                for e in self.edges
            ]
        }, indent=2)
```

### Real-Time Monitoring Server

```python
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import json


class ExecutionMonitor:
    """Real-time execution monitoring via WebSocket."""
    
    def __init__(self):
        self.connections: list[WebSocket] = []
        self.execution_state: dict = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        # Send current state
        await websocket.send_json(self.execution_state)
    
    def disconnect(self, websocket: WebSocket):
        self.connections.remove(websocket)
    
    async def broadcast(self, event: dict):
        """Broadcast event to all connected clients."""
        for connection in self.connections:
            try:
                await connection.send_json(event)
            except:
                pass  # Handle disconnected clients
    
    async def update_node(self, node_id: str, state: str, **kwargs):
        """Update node state and broadcast."""
        event = {
            "type": "node_update",
            "node_id": node_id,
            "state": state,
            **kwargs
        }
        self.execution_state[node_id] = event
        await self.broadcast(event)


monitor = ExecutionMonitor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await monitor.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            # Could handle commands here
    except:
        monitor.disconnect(websocket)


@app.get("/api/state")
async def get_state():
    """Get current execution state."""
    return monitor.execution_state


# Mount static files for visualization UI
# app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

### Reference Implementations

- [Burr](https://github.com/apache/burr) - State machine framework with telemetry UI
- [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) - Visual agent IDE
- [OpenLLMetry](https://github.com/traceloop/openllmetry) - Auto-instrumentation
- [d3-dag](https://github.com/erikbrinkman/d3-dag) - JavaScript DAG layout

---

## Priority 5: DSPy Prompt Optimization

**8-40% gains through Bayesian optimization**

### Background

DSPy (Declarative Self-improving Language Programs) replaces manual prompt engineering with **programmatic optimization**. MIPROv2, the flagship optimizer, jointly optimizes instructions and few-shot examples through Bayesian search, achieving significant gains especially on small models.

### Key Finding

> **An optimized Llama-3.2-3B achieved 3.57% better accuracy than an unoptimized Llama-3.1-8B** — optimization can overcome a 2.5x parameter disadvantage.

### Performance Benchmarks

| Task | Model | Before | After | Improvement |
|------|-------|--------|-------|-------------|
| MMLU | Llama 3.1 8B | 68.3% | **71.1%** | +2.8% |
| HotPotQA | GPT-4o-mini | 24% | **51%** | +27% |
| Custom RAG | 3B model | baseline | +40% | |

### MIPROv2 Algorithm Overview

```
MIPROv2 Algorithm:

1. BOOTSTRAP PHASE
   - Run program on training data
   - Keep outputs that pass metric validation
   - Generate max_bootstrapped_demos examples per predictor

2. INSTRUCTION PROPOSAL PHASE  
   - Generate candidate instructions using proposer
   - Proposer is dataset-aware, program-aware, and tip-aware
   - Creates diverse instruction candidates

3. BAYESIAN OPTIMIZATION PHASE
   - Use Optuna's TPE (Tree-Structured Parzen Estimator)
   - Search over instruction × few-shot combinations
   - Run num_trials trials
   - Evaluate on validation set
   - Return best configuration
```

### Implementation

```python
import dspy
from dspy.teleprompt import MIPROv2, BootstrapFewShot
from typing import Optional


# Configure DSPy with local model
def setup_dspy_local():
    """Configure DSPy with Ollama for local inference."""
    lm = dspy.LM(
        'ollama_chat/llama3.2',
        api_base='http://localhost:11434',
        temperature=0.7,
        max_tokens=1024
    )
    dspy.configure(lm=lm)
    return lm


def setup_dspy_teacher():
    """Configure larger teacher model for optimization."""
    return dspy.LM(
        'openai/gpt-4o',
        temperature=0.7,
        max_tokens=2048
    )


# Define a simple module
class QuestionAnswerer(dspy.Signature):
    """Answer questions accurately and concisely."""
    
    question: str = dspy.InputField(desc="The question to answer")
    context: str = dspy.InputField(desc="Relevant context")
    answer: str = dspy.OutputField(desc="The answer")


class RAGModule(dspy.Module):
    """Simple RAG module for optimization."""
    
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.answerer = dspy.ChainOfThought(QuestionAnswerer)
    
    def forward(self, question: str) -> str:
        context = self.retriever(question)
        result = self.answerer(question=question, context=context)
        return result.answer


# Metric function for optimization
def answer_metric(example, prediction, trace=None) -> float:
    """
    Metric for evaluating answer quality.
    
    During compilation (trace is not None): return boolean
    During evaluation: return numeric score
    """
    # Check if answer is present
    if not prediction or not prediction.answer:
        return False if trace else 0.0
    
    # Check correctness (simplified)
    is_correct = (
        example.answer.lower().strip() in 
        prediction.answer.lower().strip()
    )
    
    if trace:
        # During compilation: strict boolean
        return is_correct
    else:
        # During evaluation: numeric score
        return 1.0 if is_correct else 0.0


def optimize_with_mipro(
    module: dspy.Module,
    trainset: list,
    valset: list,
    teacher_lm: Optional[dspy.LM] = None,
    num_trials: int = 40,
    auto: str = "medium"
) -> dspy.Module:
    """
    Optimize module using MIPROv2.
    
    Args:
        module: DSPy module to optimize
        trainset: Training examples
        valset: Validation examples  
        teacher_lm: Larger model for generating demonstrations
        num_trials: Number of optimization trials
        auto: Preset - "light", "medium", or "heavy"
    
    Returns:
        Optimized module
    """
    optimizer = MIPROv2(
        metric=answer_metric,
        auto=auto,
        num_threads=4,
        # Use teacher for demo generation
        prompt_model=teacher_lm if teacher_lm else dspy.settings.lm,
    )
    
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        valset=valset,
        num_trials=num_trials,
        minibatch=True,
        minibatch_size=35,
        minibatch_full_eval_steps=5,
    )
    
    return optimized


def optimize_with_bootstrap(
    module: dspy.Module,
    trainset: list,
    teacher_lm: Optional[dspy.LM] = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 16
) -> dspy.Module:
    """
    Simpler optimization using BootstrapFewShot.
    Good for limited data (<50 examples).
    
    Args:
        module: DSPy module to optimize
        trainset: Training examples
        teacher_lm: Teacher model for generating demos
        max_bootstrapped_demos: Generated examples to keep
        max_labeled_demos: Examples from trainset to use
    
    Returns:
        Optimized module
    """
    optimizer = BootstrapFewShot(
        metric=answer_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=1,
        teacher_settings=dict(lm=teacher_lm) if teacher_lm else {}
    )
    
    return optimizer.compile(module, trainset=trainset)


def save_optimized_module(module: dspy.Module, path: str):
    """Save optimized module to JSON."""
    module.save(path)
    print(f"Saved optimized module to {path}")


def load_optimized_module(module_class, path: str) -> dspy.Module:
    """Load optimized module from JSON."""
    module = module_class()
    module.load(path)
    return module
```

### Integration with LLM-Orc

```python
"""
Strategy for integrating DSPy optimization with LLM-Orc ensembles.
"""

from dataclasses import dataclass
from pathlib import Path
import yaml
import json


@dataclass
class OptimizedAgent:
    """An LLM-Orc agent with DSPy-optimized prompts."""
    name: str
    model_profile: str
    original_system_prompt: str
    optimized_instruction: str
    few_shot_examples: list[dict]
    optimization_metadata: dict


def extract_dspy_optimization(
    compiled_module_path: str
) -> dict:
    """
    Extract optimized prompts from a compiled DSPy module.
    
    Returns dict with:
    - instruction: optimized system instruction
    - examples: few-shot examples
    - metadata: optimization info
    """
    with open(compiled_module_path) as f:
        data = json.load(f)
    
    return {
        "instruction": data.get("signature_instructions", ""),
        "examples": data.get("demos", []),
        "metadata": {
            "optimizer": data.get("optimizer", "unknown"),
            "metric_score": data.get("metric_score", None),
        }
    }


def generate_optimized_ensemble(
    original_ensemble_path: str,
    optimization_results: dict[str, dict],  # agent_name -> optimization
    output_path: str
):
    """
    Generate new ensemble YAML with DSPy-optimized prompts.
    
    Args:
        original_ensemble_path: Path to original ensemble YAML
        optimization_results: Mapping of agent names to optimization results
        output_path: Where to write optimized ensemble
    """
    with open(original_ensemble_path) as f:
        ensemble = yaml.safe_load(f)
    
    for agent in ensemble.get("agents", []):
        name = agent["name"]
        if name in optimization_results:
            opt = optimization_results[name]
            
            # Update system prompt with optimized instruction
            agent["system_prompt"] = opt["instruction"]
            
            # Add few-shot examples as context
            if opt["examples"]:
                examples_text = "\n\n".join([
                    f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
                    for i, ex in enumerate(opt["examples"][:4])
                ])
                agent["system_prompt"] = (
                    f"{opt['instruction']}\n\n"
                    f"Here are some examples:\n{examples_text}"
                )
            
            # Add optimization metadata as comment
            agent["_optimization"] = opt["metadata"]
    
    # Add optimization info to ensemble metadata
    ensemble["optimization"] = {
        "optimized_at": datetime.now().isoformat(),
        "agents_optimized": list(optimization_results.keys()),
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(ensemble, f, default_flow_style=False)
    
    print(f"Generated optimized ensemble: {output_path}")


# CLI integration concept
def llm_orc_optimize_command(ensemble_name: str, trainset_path: str):
    """
    llm-orc optimize <ensemble> --trainset <path>
    
    1. Load ensemble definition
    2. For each agent, create DSPy module from system_prompt
    3. Run MIPROv2 optimization with teacher model
    4. Save optimized prompts back to ensemble
    """
    pass  # Implementation would integrate with LLM-Orc CLI
```

### Optimization Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_trials` | 40 | Optimization iterations |
| `max_bootstrapped_demos` | 4 | Generated examples per predictor |
| `max_labeled_demos` | 4 | Examples from trainset |
| `auto` | "medium" | Preset: "light", "medium", "heavy" |
| `minibatch` | True | Use minibatch evaluation |
| `minibatch_size` | 35 | Samples per minibatch |
| `minibatch_full_eval_steps` | 5 | Full eval frequency |

### Cost Estimation

| Setting | Trials | Approx. Cost | Time |
|---------|--------|--------------|------|
| Light | ~20 | $2-3 USD | 15 min |
| Medium | ~40 | $5 USD | 35 min |
| Heavy | ~100 | $15 USD | 90 min |

### Reference Implementations

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [MIPROv2 Tutorial](https://dspy.ai/learn/optimization/optimizers/)

---

## Summary: Implementation Priorities

| Priority | Algorithm | Expected Gain | Effort | Key Dependencies |
|----------|-----------|---------------|--------|------------------|
| **1** | Self-Consistency | +6-18% accuracy | 1-2 weeks | None |
| **2** | Confidence Routing | 35-85% cost reduction | 2-3 weeks | Logprobs (Ollama 0.12.11+) |
| **3** | Task Routing | Prevents misuse | 1 week | None |
| **4** | DAG Telemetry | Debugging capability | 3-4 weeks | FastAPI, WebSocket |
| **5** | DSPy Optimization | 8-40% quality | 4-6 weeks | DSPy, Teacher model access |

### Key Insight for Local-First Deployments

**Optimized 3B models can exceed unoptimized 8B models on specific tasks.** The recommended workflow:

1. Run DSPy optimization once with cloud teacher models
2. Export compiled programs for local inference
3. Use confidence routing to selectively escalate difficult queries
4. Apply self-consistency for high-stakes decisions

This architecture achieves enterprise-grade quality with consumer-grade compute.
