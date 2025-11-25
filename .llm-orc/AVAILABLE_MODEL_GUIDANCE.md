# Available Model Guidance for LLM-ORC Ensembles

## Current Ollama Installation Analysis

**Ollama Version:** 0.11.10 (client: 0.9.6)
**Installation Path:** /opt/homebrew/bin/ollama

### Currently Installed Models

| Model | Size | Last Modified | Status |
|-------|------|---------------|--------|
| gemma3:1b | 815 MB | 4 weeks ago | ✅ Current |
| mistral:latest | 4.4 GB | 4 weeks ago | ✅ Current |
| qwen3:0.6b | 522 MB | 6 weeks ago | ✅ Current |
| gemma3:12b | 8.1 GB | 5 months ago | ⚠️ Large (>7B) |
| llama3-gradient:8b | 4.7 GB | 10 months ago | ⚠️ Large (>7B) |
| llama3:latest | 4.7 GB | 10 months ago | ⚠️ Large (>7B) |

**Note:** Older models (>6 months) should be considered for updates to newer versions.

## Recommended Small Models for Ensemble Creation

### Tier 1: Primary Recommendations (Under 7B Parameters)

#### For Code Review & Development
1. **qwen2.5-coder:7b** ⭐ **TOP CHOICE**
   - **Size:** ~4.5GB
   - **Strengths:** Outstanding code generation, supports 338 programming languages, context length 128K
   - **Performance:** Outperforms GPT-4o in some coding tasks, near-perfect implementations
   - **Use Cases:** Code review, debugging, complex algorithm implementation
   - **Install:** `ollama pull qwen2.5-coder:7b`

2. **codellama:7b**
   - **Size:** ~4.0GB
   - **Strengths:** Meta's specialized coding model, excellent syntax correctness (94.2%)
   - **Performance:** HumanEval 33.5%, MBPP 41.8%
   - **Use Cases:** Code generation, syntax checking, code completion
   - **Install:** `ollama pull codellama:7b`

#### For Content Generation & Analysis
1. **qwen2.5:7b** ⭐ **TOP CHOICE**
   - **Size:** ~4.2GB
   - **Strengths:** Multilingual (29 languages), excellent math/reasoning, 128K context
   - **Performance:** MMLU 85+, MATH 75.5, superior to Gemma2-9B-IT
   - **Use Cases:** Content creation, data analysis, multilingual tasks, reasoning
   - **Install:** `ollama pull qwen2.5:7b`

2. **mistral:7b-instruct**
   - **Size:** ~4.4GB
   - **Strengths:** Well-balanced general-purpose model, fast inference
   - **Performance:** Excellent balance of capability and resource usage
   - **Use Cases:** General assistance, document analysis, chat interfaces
   - **Install:** `ollama pull mistral:7b-instruct`

### Tier 2: Specialized & Lightweight Options

#### Ultra-Lightweight Models (Under 2B Parameters)
1. **phi3:3.8b** (Microsoft)
   - **Size:** ~2.3GB
   - **Strengths:** State-of-the-art compact model, excellent efficiency
   - **Use Cases:** Resource-constrained environments, fast responses
   - **Install:** `ollama pull phi3:3.8b`

2. **gemma2:2b**
   - **Size:** ~1.6GB
   - **Strengths:** Google's efficient model, good multilingual support
   - **Use Cases:** Edge deployment, quick prototyping
   - **Install:** `ollama pull gemma2:2b`

3. **tinyllama:1.1b**
   - **Size:** ~700MB
   - **Strengths:** Extremely lightweight, trained on 3T tokens
   - **Use Cases:** Minimal hardware, basic text tasks
   - **Install:** `ollama pull tinyllama:1.1b`

4. **smollm2:1.7b**
   - **Size:** ~1.0GB
   - **Strengths:** Optimized for edge deployment, good performance/size ratio
   - **Use Cases:** Mobile deployment, embedded systems
   - **Install:** `ollama pull smollm2:1.7b`

### Tier 3: Emerging & Specialized Models

#### For Specific Domains
1. **deepseek-coder:6.7b**
   - **Size:** ~3.8GB
   - **Strengths:** Trained on 2T code tokens, supports many languages
   - **Performance:** Competitive with larger models in coding tasks
   - **Use Cases:** Code analysis, documentation generation
   - **Install:** `ollama pull deepseek-coder:6.7b`

2. **starcoder2:7b**
   - **Size:** ~4.0GB
   - **Strengths:** Community-driven code model, good for specific coding tasks
   - **Use Cases:** Code completion, repository analysis
   - **Install:** `ollama pull starcoder2:7b`

## Use Case Recommendations

### Ensemble Composition Strategies

#### Code Review Ensemble
```yaml
primary_model: qwen2.5-coder:7b
supporting_models:
  - codellama:7b        # Syntax validation
  - deepseek-coder:6.7b # Alternative perspective
  - phi3:3.8b           # Quick checks
```

#### Content Generation Ensemble
```yaml
primary_model: qwen2.5:7b
supporting_models:
  - mistral:7b-instruct # General writing
  - gemma2:2b          # Quick drafts
  - tinyllama:1.1b     # Simple tasks
```

#### Data Analysis Ensemble
```yaml
primary_model: qwen2.5:7b
supporting_models:
  - phi3:3.8b          # Mathematical reasoning
  - mistral:7b-instruct # Interpretation
  - smollm2:1.7b       # Summarization
```

## Hardware Requirements

### Memory Usage Guidelines
- **1-2B models:** 2-4GB RAM
- **3-4B models:** 4-6GB RAM  
- **6-7B models:** 6-8GB RAM

### Performance Expectations
- **tinyllama:1.1b:** ~200+ tokens/sec (CPU)
- **phi3:3.8b:** ~150 tokens/sec (CPU)
- **qwen2.5:7b:** ~50-80 tokens/sec (CPU), ~150+ tokens/sec (GPU)

## Installation Priority Queue

### Immediate Additions (High Priority)
1. `ollama pull qwen2.5-coder:7b` - Best coding model
2. `ollama pull qwen2.5:7b` - Best general model
3. `ollama pull phi3:3.8b` - Efficient lightweight option

### Secondary Additions (Medium Priority)
4. `ollama pull gemma2:2b` - Ultra-lightweight backup
5. `ollama pull codellama:7b` - Code specialization
6. `ollama pull smollm2:1.7b` - Edge deployment

### Optional Additions (Low Priority)
7. `ollama pull deepseek-coder:6.7b` - Alternative coder
8. `ollama pull starcoder2:7b` - Community coder

## Model Update Recommendations

### Models to Update
- **gemma3:12b** → **gemma2:9b** (if needed) or remove (exceeds 7B limit)
- **llama3:latest** → **qwen2.5:7b** (better performance, smaller)
- **llama3-gradient:8b** → **qwen2.5-coder:7b** (specialized for code)

### Models to Remove (Outdated)
- wizard-vicuna-uncensored:latest (21 months old)
- dolphin2.2-mistral:7b-fp16 (21 months old)
- neural-chat:7b-v3.1-q8_0 (21 months old)
- All 2+ year old models

## Performance Benchmarks Summary

### Coding Performance (2024/2025 Data)
| Model | HumanEval | MBPP | Languages | Context |
|-------|-----------|------|-----------|---------|
| qwen2.5-coder:7b | 85+ | 85+ | 338 | 128K |
| codellama:7b | 33.5 | 41.8 | 20+ | 16K |
| deepseek-coder:6.7b | - | - | 338 | 128K |

### General Performance
| Model | MMLU | MATH | Tokens/sec |
|-------|------|------|------------|
| qwen2.5:7b | 85+ | 75.5 | ~80 |
| mistral:7b | 83.0 | 41.0 | ~100 |
| phi3:3.8b | 78.0 | 58.0 | ~150 |
| gemma2:2b | 51.0 | 26.0 | ~200 |

## Best Practices for Ensemble Creation

### Model Selection Criteria
1. **Diversity:** Mix specialized and general models
2. **Size Balance:** Include both efficient and capable models
3. **Use Case Alignment:** Match models to specific tasks
4. **Resource Constraints:** Consider available hardware

### Ensemble Patterns
- **Validator Pattern:** Use lightweight model for quick validation
- **Specialist Pattern:** Route specific tasks to domain experts
- **Consensus Pattern:** Multiple models vote on outputs
- **Fallback Pattern:** Efficient models as backup options

## Troubleshooting

### Common Issues
- **Out of Memory:** Use smaller models or reduce batch size
- **Slow Performance:** Check CPU/GPU utilization, consider model quantization
- **Poor Quality:** Ensure model is appropriate for task domain

### Model Health Checks
```bash
# Check model status
ollama list

# Test model response
ollama run qwen2.5:7b "Write a simple Python function"

# Monitor resource usage
htop
```

---

**Last Updated:** September 6, 2025
**Ollama Version:** 0.11.10
**Author:** ollama-model-advisor agent