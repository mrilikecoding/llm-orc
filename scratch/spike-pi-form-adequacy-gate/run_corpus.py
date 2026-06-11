"""Spike π — run the gate arms over the labeled corpus (Forks 1, 2, 4)."""

from __future__ import annotations

from corpus import CORPUS, PASS, REFUSE, RESIDUAL
from gates import GATES, FormRefusedError

GATE_ORDER = ["A_passthrough", "B_parse_check", "C_fence_only", "D_marker"]


def run_gate(gate, item) -> str:
    try:
        gate(item.content, item.destination_path)
        return "PASS"
    except FormRefusedError:
        return "REFUSE"


def verdict(ground_truth: str, result: str) -> str:
    """Score a (ground_truth, result) pair into a labeled outcome."""
    if ground_truth == REFUSE:
        return "catch" if result == "REFUSE" else "MISS"
    if ground_truth == PASS:
        return "FALSE-POS" if result == "REFUSE" else "ok"
    # RESIDUAL
    return "over-reach" if result == "REFUSE" else "residual"


def main() -> None:
    # Per-item x per-gate matrix
    print("## Confusion matrix (result | verdict)\n")
    header = "| item | cat | seam | path | truth | " + " | ".join(GATE_ORDER) + " |"
    sep = "|" + "---|" * (5 + len(GATE_ORDER))
    print(header)
    print(sep)
    rows = []
    for item in CORPUS:
        cells = []
        for name in GATE_ORDER:
            result = run_gate(GATES[name], item)
            v = verdict(item.ground_truth, result)
            cells.append((name, result, v))
        rows.append((item, cells))
        cell_str = " | ".join(f"{r}/{v}" for _, r, v in cells)
        print(
            f"| {item.cid} | {item.category} | {item.seam} | "
            f"{item.destination_path} | {item.ground_truth} | {cell_str} |"
        )

    # Per-gate summary
    print("\n## Per-gate summary\n")
    refuse_items = [i for i in CORPUS if i.ground_truth == REFUSE]
    pass_items = [i for i in CORPUS if i.ground_truth == PASS]
    residual_items = [i for i in CORPUS if i.ground_truth == RESIDUAL]

    print(
        "| gate | catch (C1-5 det.) | false-pos (controls) | "
        "residual-pass (C6) | viable? | both-seams? |"
    )
    print("|---|---|---|---|---|---|")
    for name in GATE_ORDER:
        gate = GATES[name]
        caught = [i for i in refuse_items if run_gate(gate, i) == "REFUSE"]
        fps = [i for i in pass_items if run_gate(gate, i) == "REFUSE"]
        res_pass = [i for i in residual_items if run_gate(gate, i) == "PASS"]
        caught_ids = {i.cid for i in caught}
        viable = len(fps) == 0 and "C1" in caught_ids
        both_seams = "C4" in caught_ids and "C5" in caught_ids
        fp_ids = ",".join(i.cid for i in fps) or "—"
        miss_ids = ",".join(i.cid for i in refuse_items if i.cid not in caught_ids) or "—"
        print(
            f"| {name} | {len(caught)}/{len(refuse_items)} "
            f"(miss: {miss_ids}) | {len(fps)}/{len(pass_items)} ({fp_ids}) | "
            f"{len(res_pass)}/{len(residual_items)} | "
            f"{'YES' if viable else 'no'} | {'YES' if both_seams else 'no'} |"
        )

    # Fork 2 verdict on B specifically
    print("\n## Fork 2 (unification) — Arm B miss-set\n")
    b = GATES["B_parse_check"]
    b_miss = [i.cid for i in refuse_items if run_gate(b, i) == "PASS"]
    b_fp = [i.cid for i in pass_items if run_gate(b, i) == "REFUSE"]
    b_res = [i.cid for i in residual_items if run_gate(b, i) == "PASS"]
    print(f"- B catches deterministic failures C1-C5: miss-set = {b_miss or '∅'}")
    print(f"- B false-positives on controls: {b_fp or '∅'}")
    print(f"- B passes the residual C6a/b/c: {b_res} (expected all 3)")
    holds = not b_miss and not b_fp and len(b_res) == len(residual_items)
    print(
        f"\n**Unification (B miss-set = residual exactly, 0 FP): "
        f"{'HOLDS' if holds else 'DOES NOT HOLD'}**"
    )


if __name__ == "__main__":
    main()
