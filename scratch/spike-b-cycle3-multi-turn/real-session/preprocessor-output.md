# DETERMINISTIC PREPROCESSING (script-agent context for cheap-with-script arm)

This block is prepended deterministically by a preprocessor before the LLM orchestrator
begins. It collects the codebase anchors the agent would otherwise need to discover via
multiple Read calls. Parallels Spike A arm 2 (A2 + script input) at multi-turn.

## Naming conventions
- Ensemble files: `<name>.yaml` in `.llm-orc/ensembles/<tier>/` (or anywhere addressable)
- Script files: `<name>.py` (typically in `.llm-orc/scripts/` for project tier)
- Ensemble.script paths: relative to repo root (cwd at invoke time)

## Script-agent contract (load-bearing)
Read JSON from stdin shaped as one of:
  - Legacy: {"dependencies": {agent_name: {"response": "..."}}, "parameters": {...}}
  - ScriptAgentInput (newer): {"agent_name": str, "input_data": Any,
    "dependencies": {agent_name: dict_or_str}, "parameters": dict}

Both formats may appear; defensive parsing recommended.

Output to stdout: a JSON object with at least `success: bool` for downstream
agents to react to. Other fields are free-form per agent purpose.

## Stdlib-only constraint
Do not pip install. Use only Python stdlib (json, sys, re, etc.) for script logic.

## Reference files (already verified to exist at these paths)

### Reference ensemble — minimal two-agent pattern
- **Path:** `.llm-orc/ensembles/testing/test-script-agents.yaml` (exists)
- **Content (first 30 lines):**
  ```yaml
  name: test-script-agents
  description: Test ensemble to verify script agent functionality
  
  agents:
    - name: script-test
      script: scripts/test-script.py
      parameters:
        test_param: "hello world"
        number: 42
      
    - name: analyze-script-output
      model_profile: default-sonnet
      system_prompt: "You are an analyzer. Summarize the script output you receive."
      depends_on: [script-test]
  ```
- **Notes for this reference:**
  - Look for: agents list with 'name', 'script' (path) for script agents; 'name', 'model_profile', 'system_prompt' for LLM agents; 'depends_on' for dependency wiring.

### Reference script-agent — canonical contract
- **Path:** `.llm-orc/scripts/aggregator.py` (exists)
- **Content (first 20 lines):**
  ```python
  #!/usr/bin/env python3
  """
  aggregator.py - Collect and structure multiple agent outputs for synthesis.
  
  Usage:
    As script agent in ensemble:
      script: aggregator.py
      depends_on: [extractor-1, extractor-2, ...]
      parameters:
        format: "markdown"  # or "json"
  """
  import json
  import sys
  
  
  def main() -> None:
      if not sys.stdin.isatty():
          config = json.loads(sys.stdin.read())
      else:
          config = {}
  ```
- **Notes for this reference:**
  - Reads JSON config from stdin: {'dependencies': {agent_name: output_dict}, 'parameters': {...}}.
  - 'dependencies' contains output of each upstream agent (the LLM that wrote the haiku in our case).
  - Extract 'response' or 'data' field from each agent_output dict.
  - Write structured output (JSON or markdown) to stdout.

### Reference script-agent — simpler example
- **Path:** `.llm-orc/scripts/test-script.py` (exists)
- **Content (first 20 lines):**
  ```python
  #!/usr/bin/env python3
  """Test script for script agent functionality."""
  
  import json
  import sys
  
  def main():
      # Read JSON input from stdin
      if not sys.stdin.isatty():
          try:
              input_data = json.loads(sys.stdin.read())
          except json.JSONDecodeError:
              input_data = {}
      else:
          input_data = {}
      
      # Process the input
      result = {
          "success": True,
          "message": "Hello from test script!",
  ```
- **Notes for this reference:**
  - Demonstrates the simplest valid script-agent: stdlib-only, stdin JSON, stdout output.

### Production code-review ensemble — for richer multi-agent reference
- **Path:** `.llm-orc/ensembles/development/code-review.yaml` (exists)
- **Content (first 30 lines):**
  ```yaml
  name: code-review
  description: Comprehensive code review ensemble mixing local and cloud expertise for thorough analysis
  
  default_task: |
    Review this code submission for a production system. Provide a thorough analysis covering:
  
    **SCOPE:** Full production readiness assessment
    **FOCUS:** Security, performance, maintainability, and team standards
    **OUTPUT:** Actionable feedback with specific recommendations
  
    Please analyze the code and provide detailed recommendations for improvement.
  
  agents:
    - name: security-auditor
      model_profile: security-auditor
  
    - name: performance-engineer
      model_profile: performance-engineer
  
    - name: senior-reviewer
      model_profile: senior-reviewer
  
    - name: tech-lead-synthesizer
      model_profile: tech-lead
      depends_on: [security-auditor, performance-engineer, senior-reviewer]
      timeout_seconds: 120
      system_prompt: |
        Synthesize the code review feedback into a comprehensive assessment.
        
        **CODE REVIEW SUMMARY**
  ```
- **Notes for this reference:**
  - Shows multiple agents with system prompts and a synthesizer depending on others.

---
End of deterministic preprocessing context. Task description follows below.
