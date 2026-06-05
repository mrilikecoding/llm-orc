"""Spike θ hosted secondary arms (portability annotation, NOT verdict-bearing).

Pre-registered: run and read only AFTER the local verdict is recorded
(methods review P2-D). Local verdict recorded 2026-06-05 (round 2: Form
B-enriched adopted). These arms mirror the winning form on
zen:minimax-m2.7: θ.h1 = B-enriched work-complete (theta3b composition),
θ.h2 = B-enriched work-remaining E4 (theta4b composition). n=10 each,
~20 calls, ~$0.03 at ω's measured per-call cost.

Usage: python theta_hosted.py thetah1|thetah2 [n]
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"

_spec_t = importlib.util.spec_from_file_location("theta", HERE / "theta.py")
theta = importlib.util.module_from_spec(_spec_t)
_spec_t.loader.exec_module(theta)

_spec_o = importlib.util.spec_from_file_location(
    "omega_lib", HERE.parent / "spike-omega-delegation-broker" / "omega_lib.py"
)
omega_lib = importlib.util.module_from_spec(_spec_o)
_spec_o.loader.exec_module(omega_lib)

MODEL = "minimax-m2.7"

ARMS = {
    "thetah1": ("theta3b", "COMPLETE"),
    "thetah2": ("theta4b", "REMAINING"),
}


def main() -> None:
    arm = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    local_arm, expected = ARMS[arm]
    req = theta.ARMS[local_arm]()  # same composition; model swapped at call
    records = []
    correct = 0
    for i in range(n):
        t0 = time.time()
        data = omega_lib.zen_chat(MODEL, req["messages"])
        if "error" in data:
            rec = {
                "verdict": None,
                "correct": False,
                "flip": False,
                "error": data["error"][:200],
                "status": data.get("status"),
            }
        else:
            # zen_chat normalizes to native-Ollama shape; adapt to OpenAI.
            adapted = {"choices": [{"message": data["message"]}]}
            rec = theta.classify(local_arm, adapted)
            rec["correct"] = rec["verdict"] == expected
        rec["run"] = i + 1
        rec["elapsed_s"] = round(time.time() - t0, 1)
        records.append(rec)
        correct += bool(rec["correct"])
        print(
            f"{arm} run {i + 1}/{n}: verdict={rec.get('verdict')} "
            f"correct={rec['correct']} ({rec['elapsed_s']}s)",
            flush=True,
        )
    out = RESULTS / f"{arm}.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"\n{arm}: correct {correct}/{n} — recorded {out}")


if __name__ == "__main__":
    main()
