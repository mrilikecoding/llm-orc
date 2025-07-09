# PR Review Ensemble

The PR Review Ensemble demonstrates script-based agent integration with LLM agents to provide comprehensive GitHub PR analysis.

## Overview

This ensemble combines:
- **GitHub Fetcher Script**: Retrieves PR data using GitHub CLI
- **Security Reviewer**: Analyzes code for security vulnerabilities
- **Performance Reviewer**: Evaluates performance implications
- **Readability Reviewer**: Assesses code quality and maintainability

## Usage

### Basic Usage

```bash
# Review a GitHub PR
echo "https://github.com/owner/repo/pull/123" | llm-orc invoke pr_review_github --output-format json

# Review with text output
echo "https://github.com/owner/repo/pull/123" | llm-orc invoke pr_review_github
```

### Prerequisites

1. **GitHub CLI**: Install and authenticate with GitHub CLI
   ```bash
   # Install GitHub CLI
   brew install gh  # macOS
   # or follow instructions at https://cli.github.com/

   # Authenticate
   gh auth login
   ```

2. **Repository Access**: Ensure you have access to the repository containing the PR

### Configuration

The ensemble is configured in `~/.llm-orc/ensembles/pr_review_github.yaml`:

```yaml
name: pr_review_github
description: Multi-perspective PR review with GitHub integration
agents:
  - name: github_fetcher
    type: script
    command: |
      # Parse GitHub PR URL and fetch data
      PR_URL="$INPUT_DATA"
      REPO=$(echo "$PR_URL" | sed -E 's|.*github.com/([^/]+/[^/]+)/pull/.*|\1|')
      PR_NUM=$(echo "$PR_URL" | sed -E 's|.*github.com/.*/pull/([0-9]+).*|\1|')
      
      echo "=== PR INFORMATION ==="
      gh pr view "$PR_NUM" --repo "$REPO" --json title,body,number,additions,deletions,files
      
      echo "=== PR DIFF ==="
      gh pr diff "$PR_NUM" --repo "$REPO" | head -200
    
  - name: security_reviewer
    type: llm
    role: security_analyst
    model: claude-3-sonnet
    prompt: |
      You are a cybersecurity expert. Review the PR data for security vulnerabilities...
    
  # ... other agents
```

## Output Format

The ensemble provides structured output with:

### JSON Output

```json
{
  "ensemble": "pr_review_github",
  "status": "completed",
  "results": {
    "github_fetcher": {
      "response": "=== PR INFORMATION ===\n...",
      "status": "success"
    },
    "security_reviewer": {
      "response": "Security analysis...",
      "status": "success"
    },
    "performance_reviewer": {
      "response": "Performance analysis...",
      "status": "success"
    },
    "readability_reviewer": {
      "response": "Code quality analysis...",
      "status": "success"
    }
  },
  "synthesis": "Comprehensive PR review combining all perspectives...",
  "metadata": {
    "duration": "60.86s",
    "agents_used": 4,
    "usage": { ... }
  }
}
```

### Text Output

- Executive summary
- Critical issues (CRITICAL/HIGH priority)
- Important improvements (MEDIUM priority)
- Minor suggestions (LOW priority)
- Positive aspects

## Architecture

This ensemble demonstrates the hybrid script + LLM agent approach:

1. **Script Agent** (`github_fetcher`):
   - Executes GitHub CLI commands
   - Parses PR URL to extract repo and PR number
   - Fetches PR metadata and diff
   - Provides structured data to LLM agents

2. **LLM Agents** (security, performance, readability):
   - Receive GitHub data as input
   - Apply domain-specific analysis
   - Generate structured feedback
   - Rate findings by priority

3. **Synthesis**:
   - Combines all agent outputs
   - Provides prioritized recommendations
   - Formats as structured JSON

## Script-Based Agent Features

The GitHub fetcher demonstrates script agent capabilities:

- **Environment Variables**: Access to `$INPUT_DATA` (PR URL)
- **Shell Commands**: Execute `gh` CLI commands
- **Error Handling**: Proper timeout and error reporting
- **Data Processing**: Parse URLs and format output

## Extending the Ensemble

### Adding New Agents

Add new agents to the `agents` array in the configuration:

```yaml
agents:
  - name: accessibility_reviewer
    type: llm
    role: accessibility_analyst
    model: claude-3-sonnet
    prompt: |
      Review the PR for accessibility issues and WCAG compliance...
```

### Customizing Script Agents

Modify the GitHub fetcher script to fetch additional data:

```yaml
- name: github_fetcher
  type: script
  command: |
    # Fetch PR comments
    gh pr view "$PR_NUM" --repo "$REPO" --json comments
    
    # Fetch PR checks
    gh pr checks "$PR_NUM" --repo "$REPO"
```

### Integration with Other Tools

Script agents can integrate with any command-line tool:

```yaml
- name: security_scanner
  type: script
  command: |
    # Clone repo and run security scan
    git clone "$REPO_URL" temp_repo
    cd temp_repo
    semgrep --config=auto .
```

## Future Enhancements

This ensemble establishes the foundation for:

1. **Context Engineering** (Issue #31): Sophisticated context transformation between agents
2. **MCP Integration** (Issue #25): Dynamic tool access for agents
3. **Advanced Workflows**: Multi-step PR review processes
4. **Custom Templates**: Configurable review templates for different project types

## Performance

- **Total Duration**: ~60-90 seconds for comprehensive review
- **Parallel Execution**: LLM agents run concurrently
- **Resource Usage**: Minimal for script agents, standard for LLM agents
- **Scalability**: Easily add more reviewers without architectural changes

## Troubleshooting

### GitHub CLI Issues

```bash
# Check authentication
gh auth status

# Re-authenticate if needed
gh auth login

# Test repository access
gh repo view owner/repo
```

### Script Agent Errors

- Check GitHub CLI installation and authentication
- Verify repository access permissions
- Review script syntax in ensemble configuration
- Check ensemble execution logs for detailed error messages

## Related Documentation

- [Script-Based Agents](./script_agents.md)
- [Ensemble Configuration](./ensemble_config.md)
- [Context Engineering](./context_engineering.md)
- [CLI Usage](./cli_usage.md)