#!/usr/bin/env python3
"""
PR Review Panel Example with GitHub CLI Integration

Demonstrates practical PR review using Ollama with real GitHub PR data.

Usage:
    python examples/pr_review_with_gh_cli.py <pr_url>
    python examples/pr_review_with_gh_cli.py https://github.com/owner/repo/pull/123

Requirements:
    - Ollama running locally with llama3 model available
    - GitHub CLI (gh) installed and authenticated
    - Access to the GitHub repository containing the PR

Example:
    ollama pull llama3
    ollama serve
    gh auth login
    python examples/pr_review_with_gh_cli.py https://github.com/anthropics/claude-code/pull/42
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_orc.models import OllamaModel
from llm_orc.orchestration import Agent, PRReviewOrchestrator
from llm_orc.roles import RoleDefinition


def extract_repo_and_pr_from_url(pr_url: str) -> tuple[str, str, int]:
    """Extract owner, repo, and PR number from GitHub PR URL."""
    try:
        parsed = urlparse(pr_url)
        if parsed.hostname != "github.com":
            raise ValueError("URL must be from github.com")

        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) < 4 or path_parts[2] != "pull":
            raise ValueError(
                "URL must be a GitHub PR URL: https://github.com/owner/repo/pull/123"
            )

        owner = path_parts[0]
        repo = path_parts[1]
        pr_number = int(path_parts[3])

        return owner, repo, pr_number
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid GitHub PR URL: {e}") from e


def fetch_pr_data_with_gh_cli(owner: str, repo: str, pr_number: int) -> dict:
    """Fetch PR data using GitHub CLI."""
    try:
        # Get PR details
        pr_cmd = [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--repo",
            f"{owner}/{repo}",
            "--json",
            "title,body,number,additions,deletions,files,headRefName,baseRefName",
        ]
        pr_result = subprocess.run(pr_cmd, capture_output=True, text=True, check=True)
        pr_data = json.loads(pr_result.stdout)

        # Get PR diff
        diff_cmd = ["gh", "pr", "diff", str(pr_number), "--repo", f"{owner}/{repo}"]
        diff_result = subprocess.run(
            diff_cmd, capture_output=True, text=True, check=True
        )
        diff_content = diff_result.stdout

        # Extract filenames from file data
        files_changed = [file_info["path"] for file_info in pr_data.get("files", [])]

        return {
            "title": pr_data["title"],
            "description": pr_data.get("body", "No description provided"),
            "number": pr_data["number"],
            "additions": pr_data["additions"],
            "deletions": pr_data["deletions"],
            "files_changed": files_changed,
            "head_branch": pr_data["headRefName"],
            "base_branch": pr_data["baseRefName"],
            "diff": diff_content[:5000],  # Limit diff size for LLM processing
            "repo": f"{owner}/{repo}",
        }

    except subprocess.CalledProcessError as e:
        if "not found" in e.stderr.lower():
            raise ValueError(
                f"PR #{pr_number} not found in {owner}/{repo}. "
                "Make sure the repository exists and you have access."
            ) from e
        elif "authentication" in e.stderr.lower():
            raise ValueError(
                "GitHub CLI authentication required. Run: gh auth login"
            ) from e
        else:
            raise ValueError(f"Failed to fetch PR data: {e.stderr}") from e
    except json.JSONDecodeError as e:
        raise ValueError("Failed to parse PR data from GitHub CLI") from e


async def create_specialist_agents() -> tuple[Agent, Agent, Agent]:
    """Create specialist reviewer agents using Ollama."""

    # Create Ollama models for each specialist
    senior_dev_model = OllamaModel(model_name="llama3", host="http://localhost:11434")
    security_model = OllamaModel(model_name="llama3", host="http://localhost:11434")
    ux_model = OllamaModel(model_name="llama3", host="http://localhost:11434")

    # Senior Developer Role
    senior_dev_role = RoleDefinition(
        name="senior_developer",
        prompt=(
            "You are a senior software developer with 10+ years of experience. "
            "Review code for: clean architecture, design patterns, naming conventions, "
            "error handling, testing needs, performance considerations, "
            "and maintainability. "
            "Provide constructive, specific feedback in 3-4 sentences. "
            "Focus on code quality and best practices."
        ),
        context={
            "specialties": ["code_quality", "best_practices", "architecture", "testing"]
        },
    )

    # Security Expert Role
    security_expert_role = RoleDefinition(
        name="security_expert",
        prompt=(
            "You are a cybersecurity expert specializing in secure coding practices. "
            "Review code for: security vulnerabilities, authentication flaws, "
            "data validation, secret management, encryption standards, "
            "injection attacks, and access controls. "
            "Identify specific security risks and provide mitigation strategies. "
            "Use CRITICAL/HIGH/MEDIUM/LOW severity ratings for issues found."
        ),
        context={
            "specialties": [
                "security",
                "encryption",
                "authentication",
                "vulnerabilities",
            ]
        },
    )

    # UX/Product Reviewer Role
    ux_reviewer_role = RoleDefinition(
        name="ux_reviewer",
        prompt=(
            "You are a UX specialist focused on user experience and product impact. "
            "Review code changes for: user experience implications, "
            "accessibility considerations, error messaging clarity, usability impact, "
            "and product requirements alignment. "
            "Consider how technical changes affect end users. "
            "Provide user-centered feedback in 3-4 sentences."
        ),
        context={
            "specialties": ["user_experience", "accessibility", "usability", "product"]
        },
    )

    # Create agents
    senior_dev = Agent("senior_dev", senior_dev_role, senior_dev_model)
    security_expert = Agent("security_expert", security_expert_role, security_model)
    ux_reviewer = Agent("ux_reviewer", ux_reviewer_role, ux_model)

    return senior_dev, security_expert, ux_reviewer


async def main():
    """Run PR review with real GitHub data using Ollama."""
    if len(sys.argv) != 2:
        print("❌ Usage: python pr_review_with_gh_cli.py <github_pr_url>")
        print(
            "   Example: python pr_review_with_gh_cli.py https://github.com/owner/repo/pull/123"
        )
        return 1

    pr_url = sys.argv[1]

    print("🔍 GitHub PR Review Panel")
    print("=" * 50)

    try:
        # Extract repo info from URL
        print(f"📋 Parsing PR URL: {pr_url}")
        owner, repo, pr_number = extract_repo_and_pr_from_url(pr_url)
        print(f"   Repository: {owner}/{repo}")
        print(f"   PR Number: #{pr_number}")

        # Fetch PR data using GitHub CLI
        print("\n📡 Fetching PR data using GitHub CLI...")
        pr_data = fetch_pr_data_with_gh_cli(owner, repo, pr_number)

        print("✅ PR Data Retrieved:")
        print(f"   Title: {pr_data['title']}")
        print(f"   Files: {len(pr_data['files_changed'])} changed")
        print(f"   Changes: +{pr_data['additions']} -{pr_data['deletions']} lines")
        print(f"   Branch: {pr_data['head_branch']} → {pr_data['base_branch']}")

        # Create specialist agents
        print("\n🤖 Creating specialist reviewer agents...")
        senior_dev, security_expert, ux_reviewer = await create_specialist_agents()
        print("   ✓ Senior Developer (Code Quality)")
        print("   ✓ Security Expert (Vulnerability Analysis)")
        print("   ✓ UX Reviewer (User Experience)")

        # Create PR review orchestrator
        print("\n🎭 Setting up PR review orchestrator...")
        pr_orchestrator = PRReviewOrchestrator()
        pr_orchestrator.register_reviewer(senior_dev)
        pr_orchestrator.register_reviewer(security_expert)
        pr_orchestrator.register_reviewer(ux_reviewer)

        # Conduct the review
        print("\n📝 Conducting multi-agent PR review...")
        print("   (This may take 30-60 seconds with Ollama...)")

        review_results = await pr_orchestrator.review_pr(pr_data)

        # Display results
        print("\n" + "=" * 50)
        print("📊 PR REVIEW RESULTS")
        print("=" * 50)
        print(f"PR: {review_results['pr_title']}")
        print(f"Repository: {pr_data['repo']}")
        print(f"Reviewers: {review_results['total_reviewers']}")

        print("\n🔎 DETAILED REVIEWS:")
        print("-" * 50)

        for review in review_results["reviews"]:
            reviewer_name = review["reviewer"].replace("_", " ").title()
            specialization = ", ".join(review["specialization"])

            print(f"\n👤 {reviewer_name} ({specialization}):")
            print(f"   {review['feedback']}")

        print("\n📋 CONSOLIDATED SUMMARY:")
        print("-" * 50)
        print(f"{review_results['summary']}")

        print("\n✨ Review complete! Consider addressing the feedback before merging.")

    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Ensure Ollama is running: ollama serve")
        print("   - Ensure llama3 model is available: ollama pull llama3")
        print("   - Ensure GitHub CLI is authenticated: gh auth login")
        print("   - Ensure you have access to the repository")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
