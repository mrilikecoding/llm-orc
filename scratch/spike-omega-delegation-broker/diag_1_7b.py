"""Spike ω — labeled diagnostics d1/d2 (non-viability-bearing, P2-D).

d1: qwen3:1.7b, thinking re-enabled, canonical delegate case (psi-c01).
    Classifies whether think=false masked tool-call capability.
d2: qwen3:1.7b, think=false (verdict-arm setting), quoted client prompt
    ablated from the composed input. Classifies the delegation suppression:
    data-leakage from the quoted prompt vs intrinsic read-bias.
"""

import json
from pathlib import Path

from omega_cases import all_cases
from omega_lib import (
    BROKER_TOOLS,
    compose_broker_input,
    extract_decision,
    ollama_chat,
)

MODEL = "qwen3:1.7b"
N = 3
RESULTS = Path(__file__).parent / "results"

case = next(c for c in all_cases() if c["id"] == "psi-c01")
out: dict[str, list] = {"d1_think_enabled": [], "d2_client_ablated": []}

# d1 — thinking re-enabled, full composed input
msgs = compose_broker_input(case["request"])
for i in range(N):
    resp = ollama_chat(MODEL, msgs, tools=BROKER_TOOLS, think=True)
    dec = extract_decision(resp)
    rec = {
        "iter": i,
        "decision": dec["decision"],
        "carry_kind": (dec.get("args") or {}).get("kind"),
        "wall_s": round(resp.get("wall_s", -1), 2),
        "eval_tokens": resp.get("eval_count"),
    }
    out["d1_think_enabled"].append(rec)
    print(f"[d1] #{i} {dec['decision']}/{rec['carry_kind']} "
          f"({rec['wall_s']}s ev={rec['eval_tokens']})")

# d2 — client prompt ablated, verdict-arm think=false
msgs = compose_broker_input(case["request"], client_prompt_override="")
for i in range(N):
    resp = ollama_chat(MODEL, msgs, tools=BROKER_TOOLS, think=False)
    dec = extract_decision(resp)
    rec = {
        "iter": i,
        "decision": dec["decision"],
        "carry_kind": (dec.get("args") or {}).get("kind"),
        "wall_s": round(resp.get("wall_s", -1), 2),
        "eval_tokens": resp.get("eval_count"),
    }
    out["d2_client_ablated"].append(rec)
    print(f"[d2] #{i} {dec['decision']}/{rec['carry_kind']} "
          f"({rec['wall_s']}s ev={rec['eval_tokens']})")

(RESULTS / "diag-qwen3-1_7b.json").write_text(json.dumps(out, indent=2))
print("-> results/diag-qwen3-1_7b.json")
