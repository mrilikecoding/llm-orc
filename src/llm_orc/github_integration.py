"""GitHub API integration for PR analysis."""

import json
import subprocess
from typing import Any
from urllib.parse import urlparse


class GitHubPRFetcher:
    """Fetches PR data using GitHub CLI."""

    def __init__(self) -> None:
        """Initialize the GitHub PR fetcher."""
        self._check_gh_cli()

    def _check_gh_cli(self) -> None:
        """Check if GitHub CLI is available and authenticated."""
        try:
            subprocess.run(
                ["gh", "--version"], capture_output=True, text=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ValueError(
                "GitHub CLI (gh) is not installed or not in PATH. "
                "Install it from https://cli.github.com/"
            ) from None

        # Check if authenticated
        try:
            result = subprocess.run(
                ["gh", "auth", "status"], capture_output=True, text=True, check=True
            )
            if "logged in" not in result.stderr.lower():
                raise ValueError("GitHub CLI is not authenticated. Run: gh auth login")
        except subprocess.CalledProcessError:
            raise ValueError(
                "GitHub CLI is not authenticated. Run: gh auth login"
            ) from None

    def parse_pr_url(self, pr_url: str) -> tuple[str, str, int]:
        """Parse a GitHub PR URL and extract owner, repo, and PR number."""
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

    def fetch_pr_data(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
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
                "title,body,number,additions,deletions,files,headRefName,baseRefName,url",
            ]
            pr_result = subprocess.run(
                pr_cmd, capture_output=True, text=True, check=True
            )
            pr_data = json.loads(pr_result.stdout)

            # Get PR diff
            diff_cmd = ["gh", "pr", "diff", str(pr_number), "--repo", f"{owner}/{repo}"]
            diff_result = subprocess.run(
                diff_cmd, capture_output=True, text=True, check=True
            )
            diff_content = diff_result.stdout

            # Extract filenames from file data
            files_changed = [
                file_info["path"] for file_info in pr_data.get("files", [])
            ]

            return {
                "url": pr_data["url"],
                "title": pr_data["title"],
                "description": pr_data.get("body", "No description provided"),
                "number": pr_data["number"],
                "additions": pr_data["additions"],
                "deletions": pr_data["deletions"],
                "files_changed": files_changed,
                "head_branch": pr_data["headRefName"],
                "base_branch": pr_data["baseRefName"],
                "diff": diff_content[:8000],  # Limit diff size for LLM processing
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

    def fetch_pr_from_url(self, pr_url: str) -> dict[str, Any]:
        """Fetch PR data from a GitHub PR URL."""
        owner, repo, pr_number = self.parse_pr_url(pr_url)
        return self.fetch_pr_data(owner, repo, pr_number)

    def format_pr_for_analysis(self, pr_data: dict[str, Any]) -> str:
        """Format PR data for LLM analysis."""
        formatted = f"""# PR Analysis: {pr_data["title"]}

**Repository:** {pr_data["repo"]}
**PR Number:** #{pr_data["number"]}
**URL:** {pr_data["url"]}

**Description:**
{pr_data["description"]}

**Changes Summary:**
- Files changed: {len(pr_data["files_changed"])}
- Additions: +{pr_data["additions"]} lines
- Deletions: -{pr_data["deletions"]} lines
- Branch: {pr_data["head_branch"]} → {pr_data["base_branch"]}

**Files Changed:**
{", ".join(pr_data["files_changed"])}

**Code Diff:**
```diff
{pr_data["diff"]}
```
"""
        return formatted
