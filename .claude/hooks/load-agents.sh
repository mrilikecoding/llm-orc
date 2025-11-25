#!/bin/bash

# Session start hook to load agent descriptions into Claude's context
# This enables proactive use of project-specific agents

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$SCRIPT_DIR/../agents"

# Build the context string
context="# Project Agents

The following specialized agents are available for proactive use in this llm-orc development session:
"

# Function to extract agent info from markdown files
extract_agent_info() {
    local file="$1"
    
    # Extract name and description from YAML frontmatter
    local name=$(grep "^name:" "$file" | sed 's/name: //')
    local description=$(grep "^description:" "$file" | sed 's/description: //')
    
    if [[ -n "$name" && -n "$description" ]]; then
        echo "## $name

**Usage**: $description

**Activation**: Use proactively when tasks match the agent's expertise area.

"
    fi
}

# Process all agent files and build context
for agent_file in "$AGENTS_DIR"/*.md; do
    if [[ -f "$agent_file" ]]; then
        context="$context$(extract_agent_info "$agent_file")"
    fi
done

# Output properly formatted JSON
cat << EOF
{
  "additionalContext": "$context"
}
EOF