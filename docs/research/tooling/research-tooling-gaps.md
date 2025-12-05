# Research Tooling Gap Analysis

This document tracks the scripts, ensembles, and MCP capabilities needed for conducting rigorous research with llm-orc.

## Current Inventory

### Existing Research Scripts

| Script | Category | Capability | Limitations |
|--------|----------|------------|-------------|
| `t_test.py` | statistics | Welch's t-test | p-value approximation only, no exact calculation |

### Existing Research Ensembles

| Ensemble | Purpose | Usable for Research |
|----------|---------|---------------------|
| `literature-synthesizer` | Lit review synthesis | Yes - background research |
| `human-in-loop-validation` | Expert validation checkpoints | Yes - human validation gates |
| `validate-execution-metrics` | Execution metrics | Partial - needs enhancement |

### Existing Primitives Applicable to Research

| Script | Category | Research Use |
|--------|----------|--------------|
| `read_file.py` | file-ops | Data loading |
| `write_file.py` | file-ops | Results storage |
| `json_extract.py` | data-transform | Data extraction |
| `get_user_input.py` | user-interaction | Human-in-loop validation |
| `confirm_action.py` | user-interaction | Approval gates |
| `replicate_n_times.py` | control-flow | Repeated trials |

---

## Gap Analysis

### Statistical Analysis Scripts (Priority: HIGH)

| Script Needed | Purpose | Complexity | Status |
|---------------|---------|------------|--------|
| `descriptive_stats.py` | Mean, median, std, quartiles | Low | NEEDED |
| `chi_square.py` | Categorical variable testing | Medium | NEEDED |
| `anova.py` | Multi-group comparison | Medium | NEEDED |
| `correlation.py` | Pearson/Spearman correlation | Low | NEEDED |
| `mann_whitney.py` | Non-parametric comparison | Medium | NEEDED |
| `effect_size.py` | Cohen's d, Cliff's delta, etc. | Low | NEEDED |
| `confidence_interval.py` | CI calculation | Low | NEEDED |
| `power_analysis.py` | Sample size determination | Medium | NEEDED |

### Data Collection Scripts (Priority: HIGH)

| Script Needed | Purpose | Complexity | Status |
|---------------|---------|------------|--------|
| `experiment_runner.py` | Run ensemble N times, collect results | Medium | NEEDED |
| `metrics_collector.py` | Extract metrics from artifacts | Low | NEEDED |
| `data_logger.py` | Structured logging for experiments | Low | NEEDED |
| `sample_generator.py` | Generate test samples with parameters | Medium | NEEDED |

### Validation Scripts (Priority: MEDIUM)

| Script Needed | Purpose | Complexity | Status |
|---------------|---------|------------|--------|
| `inter_rater_reliability.py` | Cohen's kappa, Krippendorff's alpha | Medium | NEEDED |
| `completeness_validator.py` | Check extraction completeness | Low | NEEDED |
| `accuracy_scorer.py` | Score against ground truth | Medium | NEEDED |
| `consistency_checker.py` | Check for contradictions | Medium | NEEDED |

### Comparison Scripts (Priority: HIGH)

| Script Needed | Purpose | Complexity | Status |
|---------------|---------|------------|--------|
| `output_comparator.py` | Compare two outputs (semantic/exact) | Medium | NEEDED |
| `benchmark_runner.py` | Run same task on different configs | Medium | NEEDED |
| `baseline_comparator.py` | Compare against baseline (e.g., Claude) | High | NEEDED |

### Reporting Scripts (Priority: MEDIUM)

| Script Needed | Purpose | Complexity | Status |
|---------------|---------|------------|--------|
| `results_formatter.py` | Format results as tables/markdown | Low | NEEDED |
| `chart_generator.py` | Generate basic charts (ASCII or SVG) | Medium | NEEDED |
| `latex_exporter.py` | Export results for papers | Medium | NEEDED |

---

## MCP Tool Gaps

### Missing Capabilities

| Capability | Current State | Gap |
|------------|--------------|-----|
| Batch execution | `invoke` runs once | Need `invoke_batch` for N trials |
| Artifact comparison | Manual | Need `compare_artifacts` tool |
| Metrics aggregation | Basic in `analyze_execution` | Need statistical aggregation |
| Ground truth loading | None | Need `load_ground_truth` for validation |
| Experiment state | None | Need `save/load_experiment_state` |

### Implementation Priority

1. **Batch execution** - Essential for any statistical study
2. **Metrics aggregation** - Required for comparing conditions
3. **Artifact comparison** - Needed for A/B testing ensembles

---

## Research Ensemble Gaps

### Needed Ensembles

| Ensemble | Purpose | Agents Needed |
|----------|---------|---------------|
| `experiment-runner` | Run controlled experiments | sample-generator, batch-executor, metrics-collector, stats-analyzer |
| `accuracy-evaluator` | Evaluate against ground truth | ground-truth-loader, output-comparator, accuracy-scorer, reporter |
| `ablation-study` | Test component contributions | config-generator, batch-executor, comparator, stats-analyzer |
| `inter-model-comparison` | Compare models/ensembles | benchmark-runner, metrics-collector, stats-analyzer, reporter |

---

## Implementation Roadmap

### Phase 1: Core Statistics (Foundation)
```yaml
scripts:
  - descriptive_stats.py
  - effect_size.py
  - confidence_interval.py
```

### Phase 2: Experiment Infrastructure
```yaml
scripts:
  - experiment_runner.py
  - metrics_collector.py
  - data_logger.py
mcp_tools:
  - invoke_batch
```

### Phase 3: Validation & Comparison
```yaml
scripts:
  - accuracy_scorer.py
  - output_comparator.py
  - completeness_validator.py
mcp_tools:
  - compare_artifacts
```

### Phase 4: Reporting & Publication
```yaml
scripts:
  - results_formatter.py
  - latex_exporter.py
ensembles:
  - experiment-runner
  - accuracy-evaluator
```

---

## Immediate Blockers for Research

### For the "Swarm vs Foundation Model" Study

| Blocker | Type | Resolution |
|---------|------|------------|
| No batch execution | MCP gap | Implement `invoke_batch` or use script wrapper |
| No accuracy scoring | Script gap | Create `accuracy_scorer.py` |
| No ground truth format | Design gap | Define ground truth schema |
| No metrics aggregation | Script gap | Create `metrics_collector.py` + `descriptive_stats.py` |
| t_test.py uses approximation | Script limitation | Implement proper t-distribution CDF |

### Workarounds Available

1. **Batch execution**: Loop `invoke` in a script
2. **Metrics aggregation**: Parse artifact JSON manually
3. **Comparison**: Manual human evaluation initially

---

## Next Steps

1. [ ] Implement `descriptive_stats.py`
2. [ ] Implement `experiment_runner.py`
3. [ ] Implement `metrics_collector.py`
4. [ ] Define ground truth schema for ADR extraction
5. [ ] Create `experiment-runner` ensemble
6. [ ] Conduct pilot study with N=5 ADRs
