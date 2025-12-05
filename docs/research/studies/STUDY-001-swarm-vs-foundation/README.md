# STUDY-001: Micro-Model Swarms vs Foundation Models for Structured Extraction

## Study Metadata

```yaml
study_id: STUDY-001
title: "Can Micro-Model Swarms Match Foundation Model Quality for Structured Extraction Tasks?"
authors: [TBD]
status: draft
created: 2025-12-05
updated: 2025-12-05
```

---

## 1. Research Question

### 1.1 Primary Question

> Can a swarm of small local models (< 7B parameters), augmented with script-based validation and retry mechanisms, achieve comparable extraction accuracy to a single foundation model (Claude) for structured document analysis tasks?

### 1.2 Hypotheses

| ID | Hypothesis | Type | Testable |
|----|------------|------|----------|
| H1 | Swarm extraction recall will be significantly lower than Claude baseline without validation gates | Alternative | Yes |
| H2 | Adding completeness validation gates will increase swarm recall to within 10% of Claude | Alternative | Yes |
| H3 | Adding retry mechanisms will further improve recall without significantly increasing latency | Alternative | Yes |
| H4 | Swarm precision will be comparable to Claude regardless of validation | Null | Yes |
| H5 | Swarm latency will be lower than Claude due to parallelism | Alternative | Yes |

### 1.3 Scope and Boundaries

**In Scope:**
- Structured extraction from Architecture Decision Records (ADRs)
- Extraction categories: technologies, risks, scope, phases, dependencies, alternatives
- Comparison of accuracy (recall, precision, F1)
- Comparison of latency and cost
- Effect of validation gates on accuracy

**Out of Scope:**
- Open-ended generation tasks
- Conversational or multi-turn tasks
- Fine-tuning of models
- Models > 7B parameters (focus on truly "micro" models)

**Assumptions:**
- ADRs have consistent structure (Status, Context, Decision, Consequences)
- Ground truth can be reliably established by human annotation
- Ollama provides consistent inference times for local models

---

## 2. Background and Motivation

### 2.1 Problem Statement

Foundation models like Claude excel at complex reasoning and extraction tasks but require API access, incur costs, and may not be available in all environments (air-gapped systems, privacy-sensitive contexts). Small local models are accessible and free but individually lack the capability for accurate complex extraction.

The question is whether architectural patterns—parallel execution, focused prompts, validation gates, and retry mechanisms—can compensate for individual model limitations.

### 2.2 Related Work

- **Mixture of Experts**: Sparse routing to specialized sub-models
- **Ensemble Methods in ML**: Voting, bagging, boosting
- **Multi-Agent LLM Systems**: AutoGen, CrewAI, LangGraph
- **Prompt Engineering**: Chain-of-thought, self-consistency
- **Model Cascading**: Smaller models as filters for larger ones

### 2.3 Contribution

This study provides:
1. Empirical comparison of swarm vs foundation model extraction accuracy
2. Quantification of validation gate effects on accuracy
3. Practical guidance for swarm architecture design
4. Open-source ensembles and scripts for replication

---

## 3. Methodology

### 3.1 Study Design

| Aspect | Choice | Justification |
|--------|--------|---------------|
| Design Type | Experimental (within-subjects) | Same documents analyzed by all conditions |
| Sampling | Purposive | ADRs selected to cover complexity range |
| Controls | Single foundation model (Claude) | Gold standard for comparison |

### 3.2 Variables

**Independent Variables:**

| Variable | Type | Levels | Operationalization |
|----------|------|--------|-------------------|
| Architecture | Categorical | 4 | (1) Claude baseline, (2) Basic swarm, (3) Swarm + validation, (4) Swarm + validation + retry |
| Document complexity | Ordinal | 3 | Low/Medium/High based on section count and length |

**Dependent Variables:**

| Variable | Type | Measurement | Validation |
|----------|------|-------------|------------|
| Recall | Continuous [0,1] | Extracted items / Ground truth items | Human verification |
| Precision | Continuous [0,1] | Correct items / Extracted items | Human verification |
| F1 Score | Continuous [0,1] | Harmonic mean of P and R | Calculated |
| Latency | Continuous (ms) | Time from invoke to completion | Artifact timestamps |
| Cost | Continuous ($) | API cost (Claude) or $0 (local) | Tracked |

**Confounding Variables:**

| Variable | Mitigation Strategy |
|----------|-------------------|
| Document length | Include as covariate, stratify sample |
| Model temperature | Fix at 0.7 for all conditions |
| Prompt variation | Use identical extraction prompts across swarm agents |
| Time of day | Randomize execution order |

### 3.3 Data Collection

**Sources:**
- llm-orc ADRs (docs/adrs/*.md) - 10 documents
- External ADRs from open-source projects - 10 documents

**Ground Truth Creation:**
1. Two human annotators independently extract all items per category
2. Calculate inter-rater reliability (Cohen's kappa)
3. Resolve disagreements through discussion
4. Final ground truth = consensus annotations

**Collection Method:**

```yaml
conditions:
  - name: claude-baseline
    ensemble: adr-review-claude
    model: claude-sonnet-4-20250514

  - name: swarm-basic
    ensemble: adr-swarm-review
    validation: none

  - name: swarm-validated
    ensemble: adr-swarm-validated
    validation: completeness-gate

  - name: swarm-retry
    ensemble: adr-swarm-retry
    validation: completeness-gate
    retry: on-validation-failure

sample_size: 20 ADRs
trials_per_condition: 3 (to account for stochasticity)
total_executions: 20 × 4 × 3 = 240
```

**Data Quality Checks:**
- [ ] All executions complete without errors
- [ ] Artifact JSON is parseable
- [ ] Extraction categories are consistent
- [ ] Timestamps are valid

### 3.4 Analysis Plan

**Statistical Methods:**

| Analysis | Tool/Script | Purpose |
|----------|-------------|---------|
| Descriptive stats | `descriptive_stats.py` | Summarize accuracy by condition |
| Repeated measures ANOVA | `anova.py` | Test main effect of architecture |
| Post-hoc pairwise comparisons | `t_test.py` with Bonferroni | Compare specific conditions |
| Effect size | `effect_size.py` | Quantify practical significance |
| Correlation | `correlation.py` | Relationship between complexity and accuracy |

**Significance Criteria:**
- Alpha level: 0.05
- Power target: 0.80
- Multiple comparison correction: Bonferroni (4 conditions = α/6 for pairwise)

---

## 4. Implementation

### 4.1 Ensembles Required

```yaml
ensembles:
  - name: adr-review-claude
    purpose: Baseline extraction with Claude
    status: NEEDED
    agents:
      - Single Claude agent with comprehensive extraction prompt

  - name: adr-swarm-review
    purpose: Basic swarm extraction (existing)
    status: EXISTS

  - name: adr-swarm-validated
    purpose: Swarm with completeness validation gate
    status: NEEDED
    agents:
      - All adr-swarm-review agents
      - validate-completeness (script)

  - name: adr-swarm-retry
    purpose: Swarm with validation and retry
    status: NEEDED
    agents:
      - All adr-swarm-validated agents
      - retry-on-failure logic
```

### 4.2 Scripts Required

| Script | Category | Purpose | Status |
|--------|----------|---------|--------|
| `parse_document.py` | preprocessing | Extract ADR sections | NEEDED |
| `completeness_validator.py` | validation | Check extraction against sections | NEEDED |
| `accuracy_scorer.py` | evaluation | Score against ground truth | NEEDED |
| `experiment_runner.py` | execution | Run all conditions | NEEDED |
| `metrics_collector.py` | analysis | Extract metrics from artifacts | NEEDED |
| `descriptive_stats.py` | analysis | Calculate summary statistics | NEEDED |
| `anova.py` | analysis | ANOVA test | NEEDED |
| `effect_size.py` | analysis | Cohen's d calculation | NEEDED |

### 4.3 Execution Plan

```yaml
phases:
  - name: ground_truth
    description: Create annotated ground truth for 20 ADRs
    deliverable: ground_truth.json

  - name: pilot
    description: Test methodology on 3 ADRs
    sample_size: 3
    success_criteria: All scripts run without error

  - name: tooling
    description: Implement missing scripts and ensembles
    deliverables:
      - completeness_validator.py
      - accuracy_scorer.py
      - experiment_runner.py
      - adr-swarm-validated ensemble
      - adr-swarm-retry ensemble

  - name: data_collection
    description: Full experimental runs
    sample_size: 20
    trials: 3 per condition

  - name: analysis
    description: Statistical analysis
    scripts: [descriptive_stats.py, anova.py, effect_size.py]

  - name: validation
    description: Peer review of methodology and findings
    method: human-in-loop-validation ensemble
```

---

## 5. Results

*[To be completed after data collection]*

### 5.1 Descriptive Statistics

| Condition | Recall (mean ± SD) | Precision (mean ± SD) | F1 (mean ± SD) | Latency (ms) |
|-----------|-------------------|----------------------|----------------|--------------|
| Claude baseline | | | | |
| Swarm basic | | | | |
| Swarm validated | | | | |
| Swarm retry | | | | |

### 5.2 Hypothesis Testing

| Hypothesis | Test | Result | Conclusion |
|------------|------|--------|------------|
| H1 | t-test | | |
| H2 | t-test | | |
| H3 | t-test | | |
| H4 | t-test | | |
| H5 | t-test | | |

### 5.3 Effect Sizes

| Comparison | Cohen's d | Interpretation |
|------------|-----------|----------------|
| Claude vs Swarm basic | | |
| Claude vs Swarm validated | | |
| Claude vs Swarm retry | | |

---

## 6. Discussion

*[To be completed after analysis]*

### 6.1 Interpretation
### 6.2 Limitations
### 6.3 Implications
### 6.4 Future Work

---

## 7. Artifacts

### 7.1 Data Files

| File | Description | Location |
|------|-------------|----------|
| `ground_truth.json` | Human-annotated ground truth | `data/` |
| `results_raw.json` | Raw experimental results | `data/` |
| `results_analyzed.json` | Analyzed results | `data/` |

### 7.2 Reproducibility

```bash
# 1. Create ground truth (manual step)
# 2. Run experiment
python scripts/experiment_runner.py \
  --conditions claude,swarm,swarm-validated,swarm-retry \
  --documents data/adrs/*.md \
  --trials 3 \
  --output data/results_raw.json

# 3. Analyze
python scripts/accuracy_scorer.py \
  --results data/results_raw.json \
  --ground-truth data/ground_truth.json \
  --output data/results_scored.json

python scripts/descriptive_stats.py \
  --input data/results_scored.json

python scripts/anova.py \
  --input data/results_scored.json \
  --factor architecture
```

---

## 8. Appendices

### A. Ground Truth Schema

```json
{
  "document_id": "ADR-010",
  "annotations": {
    "technologies": ["FastAPI", "Preact", "Dagre", "D3", "Chart.js", "uPlot", "Vite", "Tailwind CSS"],
    "risks": ["Maintenance burden", "Frontend complexity", "Potential divergence", "Duplication with MCP"],
    "scope_in": ["src/llm_orc/web/", "server.py", "api.py", "static/", "cli.py"],
    "scope_out": [],
    "phases": ["Core Server", "Basic Frontend", "Execution & Streaming", "Visualization", "Polish"],
    "dependencies": ["ADR-009", "ADR-008"],
    "alternatives": ["TUI", "VS Code Extension", "Electron App"]
  },
  "annotators": ["annotator1", "annotator2"],
  "agreement": 0.92
}
```

### B. Extraction Prompt (Swarm)

Each swarm agent receives a focused prompt for one category. Example for technologies:

```
Extract all technologies, frameworks, and libraries mentioned in this document.

Output format:
- technology1
- technology2
...

If none found, output: "None found."
```

### C. Extraction Prompt (Claude Baseline)

Claude receives a comprehensive prompt covering all categories:

```
Analyze this Architecture Decision Record and extract the following:

1. TECHNOLOGIES: All frameworks, libraries, tools mentioned
2. RISKS: All risks, concerns, negative consequences
3. SCOPE_IN: What is explicitly in scope
4. SCOPE_OUT: What is explicitly out of scope
5. PHASES: Implementation phases or timeline
6. DEPENDENCIES: Related ADRs or external dependencies
7. ALTERNATIVES: Alternative approaches considered

Output as JSON with these keys. Use empty arrays if nothing found.
```

---

## Checklist

### Pre-Study
- [ ] Research question is clear and testable
- [ ] Hypotheses are falsifiable
- [ ] Ground truth schema defined
- [ ] Required scripts identified
- [ ] Sample size justified (20 docs × 3 trials = 60 per condition)
- [ ] Inter-rater reliability planned

### Tooling (Blockers)
- [ ] `parse_document.py` implemented
- [ ] `completeness_validator.py` implemented
- [ ] `accuracy_scorer.py` implemented
- [ ] `experiment_runner.py` implemented
- [ ] `descriptive_stats.py` implemented
- [ ] `anova.py` implemented
- [ ] `adr-review-claude` ensemble created
- [ ] `adr-swarm-validated` ensemble created
- [ ] `adr-swarm-retry` ensemble created

### During Study
- [ ] Ground truth annotations complete
- [ ] Inter-rater reliability calculated (κ > 0.8)
- [ ] Pilot study successful
- [ ] All 240 executions complete
- [ ] Data quality checks passing

### Post-Study
- [ ] All hypotheses tested
- [ ] Effect sizes reported
- [ ] Limitations documented
- [ ] Results are reproducible
- [ ] Paper draft complete
