"""Spike D Arm B1 trial — staged orchestration via direct model factory.

Bypasses the opencode CLI path (which stalls on substantial code prompts)
and uses llm-orc's production ConfigurationManager + ModelFactory directly.
This is the same path Spike B's tau_shape_harness used and that worked
reliably for multi-call cheap-tier workflows.

Workflow:
  Stage 1: load spike-c-code-review review output (already produced in Spike C)
  Stage 2: call MiniMax directly with original file + compact issue list,
           ask for fixed file in a code block
  Stage 3: extract code block, save to fixed-files/arm-b1-trial1.py
  Stage 4: invoke spike-d-fix-verifier on the fixed file
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_orc.cli_library.template_provider import LibraryTemplateProvider
from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.models.model_factory import ModelFactory


HERE = Path(__file__).resolve().parent
ORIGINAL_FILE = REPO_ROOT / "scratch/spike-c-cycle3-architecture-comparison/fixture/session_budget.py"
ISSUE_LIST = """ISSUES TO FIX (from ensemble code review):

1. SECURITY: logger.warning emits 'API key: {self.api_key}' — credential leak. Remove the api_key from the log message (hash, mask, or omit).

2. OFF-BY-ONE: check_limit uses 'if total_used() > limit' — should be '>=' so the at-limit case correctly returns False.

3. TYPE ANNOTATION: 'def register(self, ..., limit: int = None)' — None is not int. Should be 'limit: int | None = None'. Also fix the truthy guard: 'limit if limit is not None else DEFAULT_BUDGET_LIMIT' (so limit=0 is preserved).

4. CROSS-FILE DRIFT: DEFAULT_BUDGET_LIMIT = 100_000 in this file but DEFAULT_MAX_TOKEN_LIMIT = 50_000_000 in src/llm_orc/agentic/orchestrator_config.py — 500x mismatch. Import the value: 'from llm_orc.agentic.orchestrator_config import DEFAULT_MAX_TOKEN_LIMIT'.

5. TEST GAP: Module ships without tests. Acknowledge this in the module docstring at minimum.
"""


def extract_code_block(text: str) -> str | None:
    """Extract the LAST python code block from response text."""
    matches = re.findall(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if matches:
        return matches[-1]
    return None


async def main() -> int:
    print(f"[{time.strftime('%H:%M:%S')}] Stage 1: Loading existing spike-c-code-review output...")
    review_path = HERE / "stage1-review-output.txt"
    if not review_path.exists():
        print(f"FATAL: stage1-review-output.txt not found at {review_path}")
        return 1
    print(f"  Review available at {review_path} ({review_path.stat().st_size} chars)")

    print(f"\n[{time.strftime('%H:%M:%S')}] Stage 2: Calling MiniMax via model factory to generate fix...")
    config_manager = ConfigurationManager(template_provider=LibraryTemplateProvider())
    credential_storage = CredentialStorage(config_manager)
    model_factory = ModelFactory(config_manager, credential_storage)
    model = await model_factory.load_model_from_agent_config(
        {"model_profile": "orchestrator-minimax-m25-free"}
    )

    original_code = ORIGINAL_FILE.read_text()
    user_prompt = f"""Apply the listed fixes to this Python file and output the complete fixed file.
Output ONLY the fixed Python code in a single fenced code block (```python ... ```). Nothing before or after the fenced block.

ORIGINAL FILE:
```python
{original_code}
```

{ISSUE_LIST}

Output the complete fixed file now (single ```python ... ``` block, no other text):"""

    role_prompt = "You are a code-fix generator. Produce only the fixed file in a single fenced code block."

    print(f"  Prompt length: {len(user_prompt)} chars; system: {len(role_prompt)} chars")

    start = time.time()
    try:
        response = await asyncio.wait_for(
            model.generate_response(message=user_prompt, role_prompt=role_prompt),
            timeout=300,  # 5 min
        )
    except asyncio.TimeoutError:
        print(f"  TIMEOUT after 300s — model did not respond")
        return 2
    duration = time.time() - start

    usage = model.get_last_usage() or {}
    print(f"  Response received in {duration:.1f}s; "
          f"in/out tokens: {usage.get('input_tokens', '?')}/{usage.get('output_tokens', '?')}; "
          f"response chars: {len(response)}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Stage 3: Extracting code block + saving fixed file...")
    fixed_code = extract_code_block(response)
    if not fixed_code:
        print(f"  FATAL: No code block found in response. First 500 chars:")
        print(response[:500])
        return 3

    fixed_path = HERE / "fixed-files" / "arm-b1-trial1.py"
    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_path.write_text(fixed_code)
    print(f"  Saved {len(fixed_code)} chars to {fixed_path}")

    # Save the full response for record
    full_response_path = HERE / "trials" / "arm-b1-trial1-stage2-response.txt"
    full_response_path.parent.mkdir(parents=True, exist_ok=True)
    full_response_path.write_text(response)
    print(f"  Full Stage 2 response saved to {full_response_path}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Workflow complete (Stage 4 verifier runs separately).")
    print(f"\nNext: uv run llm-orc invoke spike-d-fix-verifier -f {fixed_path} --output-format json --no-streaming")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
