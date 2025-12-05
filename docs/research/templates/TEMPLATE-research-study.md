# Research Study Template

Use this template to structure research questions and studies conducted with llm-orc ensembles.

## Study Metadata

```yaml
study_id: STUDY-XXX
title: [Descriptive title]
authors: [Names]
status: [draft | in-progress | data-collection | analysis | review | published]
created: YYYY-MM-DD
updated: YYYY-MM-DD
```

---

## 1. Research Question

### 1.1 Primary Question
> State the primary research question clearly and concisely.

### 1.2 Hypotheses

| ID | Hypothesis | Type | Testable |
|----|------------|------|----------|
| H1 | [Statement] | Null/Alternative | Yes/No |
| H2 | [Statement] | Null/Alternative | Yes/No |

### 1.3 Scope and Boundaries

**In Scope:**
- [What this study addresses]

**Out of Scope:**
- [What this study does not address]

**Assumptions:**
- [Assumptions made]

---

## 2. Background and Motivation

### 2.1 Problem Statement
[Why is this research needed? What gap does it fill?]

### 2.2 Related Work
[Brief survey of related research. Use literature-synthesizer ensemble if needed.]

### 2.3 Contribution
[What novel contribution does this study make?]

---

## 3. Methodology

### 3.1 Study Design

| Aspect | Choice | Justification |
|--------|--------|---------------|
| Design Type | [Experimental/Quasi-experimental/Observational] | [Why] |
| Sampling | [Random/Stratified/Convenience] | [Why] |
| Controls | [Control conditions] | [Why] |

### 3.2 Variables

**Independent Variables:**
| Variable | Type | Levels/Range | Operationalization |
|----------|------|--------------|-------------------|
| [Name] | [Categorical/Continuous] | [Values] | [How measured] |

**Dependent Variables:**
| Variable | Type | Measurement | Validation |
|----------|------|-------------|------------|
| [Name] | [Type] | [How measured] | [How validated] |

**Confounding Variables:**
| Variable | Mitigation Strategy |
|----------|-------------------|
| [Name] | [How controlled] |

### 3.3 Data Collection

**Sources:**
- [Data source 1]
- [Data source 2]

**Collection Method:**
```yaml
ensemble: [ensemble-name]
input_format: [Format specification]
output_format: [Format specification]
sample_size: [N]
collection_period: [Dates]
```

**Data Quality Checks:**
- [ ] Completeness validation
- [ ] Format validation
- [ ] Outlier detection
- [ ] Consistency checks

### 3.4 Analysis Plan

**Statistical Methods:**
| Analysis | Tool/Script | Purpose |
|----------|-------------|---------|
| [Descriptive] | [Script] | [Why] |
| [Inferential] | [Script] | [Why] |
| [Effect Size] | [Script] | [Why] |

**Significance Criteria:**
- Alpha level: [0.05]
- Power analysis: [Details]
- Multiple comparison correction: [Method]

---

## 4. Implementation

### 4.1 Ensembles Used

```yaml
ensembles:
  - name: [ensemble-1]
    purpose: [What it does in this study]
    configuration: [Any modifications]

  - name: [ensemble-2]
    purpose: [What it does in this study]
```

### 4.2 Scripts Required

| Script | Category | Purpose | Status |
|--------|----------|---------|--------|
| [script.py] | [Category] | [Purpose] | [Exists/Needed] |

### 4.3 Execution Plan

```yaml
phases:
  - name: pilot
    description: Test methodology on small sample
    sample_size: [N]
    success_criteria: [Criteria]

  - name: data_collection
    description: Full data collection
    sample_size: [N]

  - name: analysis
    description: Statistical analysis
    scripts: [List]

  - name: validation
    description: Validate findings
    method: [Method]
```

---

## 5. Results

### 5.1 Descriptive Statistics

[Tables and summaries of data]

### 5.2 Hypothesis Testing

| Hypothesis | Test | Statistic | p-value | Effect Size | Conclusion |
|------------|------|-----------|---------|-------------|------------|
| H1 | [Test] | [Value] | [Value] | [Value] | [Supported/Rejected] |

### 5.3 Visualizations

[Charts, graphs, figures with captions]

---

## 6. Discussion

### 6.1 Interpretation
[What do the results mean?]

### 6.2 Limitations
[Study limitations and threats to validity]

### 6.3 Implications
[Practical and theoretical implications]

### 6.4 Future Work
[Suggested follow-up studies]

---

## 7. Artifacts

### 7.1 Data Files

| File | Description | Location |
|------|-------------|----------|
| [file.json] | [Description] | [Path] |

### 7.2 Execution Artifacts

| Artifact ID | Ensemble | Purpose |
|-------------|----------|---------|
| [ID] | [Ensemble] | [Purpose] |

### 7.3 Reproducibility

```bash
# Commands to reproduce this study
llm-orc invoke [ensemble] "[input]"
```

---

## 8. Appendices

### A. Raw Data Samples
### B. Full Statistical Output
### C. Ensemble Configurations

---

## Checklist

### Pre-Study
- [ ] Research question is clear and testable
- [ ] Hypotheses are falsifiable
- [ ] Methodology is appropriate for question
- [ ] Required scripts exist or are identified
- [ ] Sample size is justified
- [ ] Data collection plan is complete

### During Study
- [ ] Data quality checks passing
- [ ] Execution artifacts are being stored
- [ ] Anomalies are documented

### Post-Study
- [ ] All hypotheses tested
- [ ] Effect sizes reported
- [ ] Limitations documented
- [ ] Results are reproducible
- [ ] Artifacts are archived
