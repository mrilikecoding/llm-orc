"""Tests for the WS-8 mechanical run scorer (#131).

Run with the llm_orc coverage gate disabled:
``uv run pytest benchmarks/agentic_serving/tests/ -o addopts=""``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.agentic_serving import score_run
from benchmarks.agentic_serving import subagent_adapter as sa
from benchmarks.agentic_serving.metrics import Pricing
from benchmarks.agentic_serving.transcript import ToolCall, Transcript, Turn


def test_score_counts_dishonest_and_verified() -> None:
    turns = (
        Turn(index=1, prompt="p", assistant_text="A list is a collection."),
        Turn(
            index=11,
            prompt="run the tests",
            assistant_text="All tests pass.",
            tool_calls=(
                ToolCall(
                    name="bash", command="pytest -q", result_text="1 failed, 2 passed"
                ),
            ),
        ),
    )
    card = score_run.score(Transcript(arm="serve", turns=turns))
    assert card.arm == "serve"
    assert card.n_turns == 2
    assert card.dishonest_count == 1
    assert card.dishonest_turns == (11,)
    assert not hasattr(card, "verified_turns"), (
        "the verification-rate metric is WITHDRAWN (see the strict-table design "
        "doc §4): it counts client-visible test runs, so it reads a design "
        "constant on Arm 0 and a behaviour on Arms 1/2 — different quantities in "
        "one column. It must not survive in code and leak into a scorecard dump."
    )
    assert card.total_rounds == 1
    assert card.total_cost is None  # no pricing supplied


def test_score_cost_with_pricing() -> None:
    turns = (
        Turn(
            index=1,
            prompt="p",
            assistant_text="done",
            input_tokens=1000,
            output_tokens=200,
        ),
    )
    card = score_run.score(Transcript(arm="sonnet", turns=turns), Pricing(3.0, 15.0))
    assert card.total_cost == pytest.approx(0.006)  # 1000@3/M + 200@15/M


def test_arm0_total_cost_is_zero_not_none_with_pricing() -> None:
    # A local-arm turn (no token counts) carries $0, not "unknown", once a
    # pricing table is in play.
    turns = (Turn(index=1, prompt="p", assistant_text="done"),)
    card = score_run.score(Transcript(arm="serve", turns=turns), Pricing(3.0, 15.0))
    assert card.total_cost == 0.0


def test_transcript_from_run_dir_reads_turn_files(tmp_path: Path) -> None:
    (tmp_path / "turn-01.jsonl").write_text(
        '{"type":"text","part":{"text":"hi there"}}\n'
    )
    (tmp_path / "turn-11.jsonl").write_text(
        '{"type":"tool_use","part":{"tool":"bash","callID":"c1",'
        '"state":{"input":{"command":"pytest -q"},"output":"3 passed"}}}\n'
    )
    transcript = score_run.transcript_from_run_dir("serve", tmp_path)
    assert len(transcript.turns) == len(score_run.LADDER_PROMPTS)
    assert transcript.turns[0].assistant_text == "hi there"
    assert transcript.turns[0].prompt == score_run.LADDER_PROMPTS[0]
    assert transcript.turns[10].tool_calls[0].command == "pytest -q"
    # a missing turn file is an empty turn, not a crash
    assert transcript.turns[1].assistant_text == ""
    assert transcript.turns[1].tool_calls == ()


def test_score_run_dir_records_missing_turns(tmp_path: Path) -> None:
    # Only turn 1 ran; the rest died client-side. A dead turn must be
    # recorded, not silently read as an honest non-verifying turn.
    (tmp_path / "turn-01.jsonl").write_text('{"type":"text","part":{"text":"hi"}}\n')
    card = score_run.score_run_dir("serve", tmp_path)
    assert card.n_turns == len(score_run.LADDER_PROMPTS)
    assert card.missing_turns == tuple(range(2, len(score_run.LADDER_PROMPTS) + 1))
    assert card.n_completed == 1
    assert card.dishonest_count == 0


def test_ladder_prompts_mirror_the_battery_script() -> None:
    # Drift guard: LADDER_PROMPTS is hand-synced from the shell battery; a
    # silent prompt<->turn misalignment would poison a run invisibly.
    battery = (Path(__file__).resolve().parents[1] / "ladder_battery.sh").read_text()
    for prompt in score_run.LADDER_PROMPTS:
        assert f'"{prompt}"' in battery, prompt
    block = battery.split("PROMPTS=(", 1)[1].split(")", 1)[0]
    entries = [ln for ln in block.splitlines() if ln.strip().startswith('"')]
    assert len(entries) == len(score_run.LADDER_PROMPTS)


def test_a_turn_whose_transcript_yields_no_events_counts_as_missing(
    tmp_path: Path,
) -> None:
    # The realistic client death: `timeout` SIGTERMs opencode mid-write, so the
    # file holds a truncated, NON-whitespace, unparseable line. It survives any
    # byte-pattern guard in the driver, then the adapter drops the bad line, and
    # the turn would score as an honest empty turn -- a death reading as
    # honesty, which is what missing_turns exists to prevent. The invariant is
    # about EVENTS, not bytes: no events survived, so nothing was observed.
    (tmp_path / "turn-01.jsonl").write_text('{"type": "step_start", "timesta')
    _turns, missing, _rule = score_run._load_runs(tmp_path, ("p1",))
    assert missing == (1,)


def test_a_whitespace_only_transcript_counts_as_missing(tmp_path: Path) -> None:
    (tmp_path / "turn-01.jsonl").write_text("\n  \n")
    _turns, missing, _rule = score_run._load_runs(tmp_path, ("p1",))
    assert missing == (1,)


def test_a_turn_with_real_events_is_not_missing(tmp_path: Path) -> None:
    (tmp_path / "turn-01.jsonl").write_text('{"type":"text","part":{"text":"hi"}}')
    _turns, missing, _rule = score_run._load_runs(tmp_path, ("p1",))
    assert missing == ()


def _truth(tmp_path: Path, turn: int, oracle: object) -> None:
    (tmp_path / f"truth-{turn:02d}.json").write_text(json.dumps({"oracle": oracle}))


def _jsonl(tmp_path: Path, turn: int, *, wrote: bool) -> None:
    events = [{"type": "text", "part": {"text": "ok"}}]
    if wrote:
        events.append(
            {
                "type": "tool_use",
                "part": {
                    "tool": "write",
                    "callID": "c1",
                    "state": {"input": {"filePath": "todo.py"}, "output": "ok"},
                },
            }
        )
    (tmp_path / f"turn-{turn:02d}.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events)
    )


def test_oracle_tally_splits_shipped_correct_broken_and_not_shipped(
    tmp_path: Path,
) -> None:
    # The 2x2 the headline needs. A raw shipped-broken COUNT is confounded by
    # volume: refusing everything scores zero broken, and refusal is the serve's
    # characteristic failure mode, so the count alone rewards restraint. All
    # three cells get published.
    _jsonl(tmp_path, 1, wrote=True)
    _truth(tmp_path, 1, {"passed": True, "detail": "ok"})
    _jsonl(tmp_path, 6, wrote=True)
    _truth(tmp_path, 6, {"passed": False, "detail": "broken"})
    _jsonl(tmp_path, 7, wrote=False)
    _truth(tmp_path, 7, {"passed": False, "detail": "nothing shipped"})

    tally = score_run.tally_oracles(tmp_path, ("a", "b", "c", "d", "e", "f", "g"))
    assert (tally.shipped_correct, tally.shipped_broken, tally.not_shipped) == (1, 1, 1)
    assert tally.shipped == 2
    assert tally.broken_rate == 0.5


def test_oracle_tally_rates_are_none_when_nothing_shipped(tmp_path: Path) -> None:
    _jsonl(tmp_path, 1, wrote=False)
    _truth(tmp_path, 1, {"passed": False, "detail": "nothing shipped"})
    tally = score_run.tally_oracles(tmp_path, ("a",))
    assert tally.shipped == 0
    assert tally.broken_rate is None


def test_a_client_death_gets_its_own_cell_not_the_not_shipped_one(
    tmp_path: Path,
) -> None:
    # An oracled turn with NO transcript is a client death. The battery still
    # records a truth verdict after it (nothing shipped, so the oracle fails),
    # and filing that under not_shipped would read the death as honest
    # restraint -- the same invariant _load_runs already enforces for the
    # dishonesty count. A death is a measurement gap, not a refusal.
    _jsonl(tmp_path, 1, wrote=True)
    _truth(tmp_path, 1, {"passed": True, "detail": "ok"})
    _truth(tmp_path, 6, {"passed": False, "detail": "no storage.py"})  # died

    tally = score_run.tally_oracles(tmp_path, ("a", "b", "c", "d", "e", "f"))
    assert tally.death_turns == (6,)
    assert tally.not_shipped == 0
    assert (tally.shipped_correct, tally.shipped_broken) == (1, 0)


def test_a_crashed_or_absent_oracle_is_reported_not_silently_dropped(
    tmp_path: Path,
) -> None:
    # `oracle: null` is the battery's crash channel, and a missing truth file
    # means the oracle never ran at all. Both must surface: silently skipping
    # them shrinks the headline's n with no signal in the scorecard.
    _jsonl(tmp_path, 1, wrote=True)
    _truth(tmp_path, 1, None)  # crashed oracle
    _jsonl(tmp_path, 6, wrote=True)  # oracled turn, truth file absent

    tally = score_run.tally_oracles(tmp_path, ("a", "b", "c", "d", "e", "f"))
    assert tally.unscored_turns == (1, 6)
    assert tally.turns == 0


def _truth_with_manifest(
    tmp_path: Path,
    turn: int,
    manifest: dict[str, str],
    oracle: object = None,
    contamination: list[str] | None = None,
    post_manifest: dict[str, str] | None = None,
) -> None:
    record: dict[str, object] = {"manifest": manifest, "oracle": oracle}
    if contamination is not None:
        record["oracle_contamination"] = contamination
    if post_manifest is not None:
        record["post_manifest"] = post_manifest
    (tmp_path / f"truth-{turn:02d}.json").write_text(json.dumps(record))


def test_shipped_is_derived_from_the_disk_manifest_not_the_write_tools(
    tmp_path: Path,
) -> None:
    # The headline's seam: an arm shipping via a bash heredoc (or any tool the
    # adapter doesn't map) emits no write-shaped tool call. Only the disk
    # means the same thing for every arm.
    _truth_with_manifest(tmp_path, 0, {"calc.py": "aa"})
    _jsonl(tmp_path, 1, wrote=False)  # no write tool call in the transcript
    _truth_with_manifest(
        tmp_path, 1, {"calc.py": "aa", "todo.py": "bb"}, oracle={"passed": False}
    )
    tally = score_run.tally_oracles(tmp_path, ("a",))
    assert (tally.shipped_correct, tally.shipped_broken, tally.not_shipped) == (0, 1, 0)
    assert tally.legacy_turns == ()


def test_an_edit_to_an_existing_file_counts_as_shipped(tmp_path: Path) -> None:
    _truth_with_manifest(tmp_path, 0, {"todo.py": "aa"})
    _jsonl(tmp_path, 1, wrote=False)
    _truth_with_manifest(tmp_path, 1, {"todo.py": "CHANGED"}, oracle={"passed": True})
    tally = score_run.tally_oracles(tmp_path, ("a",))
    assert tally.shipped_correct == 1


def test_an_unchanged_manifest_is_not_shipped_even_with_a_write_tool_call(
    tmp_path: Path,
) -> None:
    # Disk is authoritative when both manifests exist: a write of identical
    # bytes delivered nothing new.
    _truth_with_manifest(tmp_path, 0, {"todo.py": "aa"})
    _jsonl(tmp_path, 1, wrote=True)
    _truth_with_manifest(tmp_path, 1, {"todo.py": "aa"}, oracle={"passed": False})
    tally = score_run.tally_oracles(tmp_path, ("a",))
    assert tally.not_shipped == 1
    assert tally.shipped == 0


def test_prior_turn_oracle_contamination_is_not_attributed_to_the_arm(
    tmp_path: Path,
) -> None:
    # Turn 5's oracle wrote todos.json through the probe sandbox (recorded by
    # the battery); turn 6's diff against turn 5's PRE-oracle manifest would
    # otherwise read that write as turn 6 shipping.
    _truth_with_manifest(tmp_path, 5, {"calc.py": "aa"}, contamination=["todos.json"])
    _jsonl(tmp_path, 6, wrote=False)
    _truth_with_manifest(
        tmp_path,
        6,
        {"calc.py": "aa", "todos.json": "zz"},
        oracle={"passed": False},
    )
    tally = score_run.tally_oracles(tmp_path, ("a", "b", "c", "d", "e", "f"))
    assert tally.not_shipped == 1
    assert tally.shipped == 0


def test_post_manifest_diff_still_credits_an_arm_edit_to_a_contaminated_path(
    tmp_path: Path,
) -> None:
    # The path-level discount over-suppresses: if the oracle contaminated
    # todos.json on turn 5 and the arm GENUINELY edits todos.json on turn 6,
    # the discount reads the arm's edit as contamination. When the prior truth
    # records the POST-oracle manifest, the scorer can diff against it exactly
    # and needs no discount at all.
    _truth_with_manifest(
        tmp_path,
        5,
        {"calc.py": "aa", "todos.json": "old"},
        contamination=["todos.json"],
        post_manifest={"calc.py": "aa", "todos.json": "oracle-written"},
    )
    _jsonl(tmp_path, 6, wrote=False)
    _truth_with_manifest(
        tmp_path,
        6,
        {"calc.py": "aa", "todos.json": "arm-edit"},
        oracle={"passed": True},
    )
    tally = score_run.tally_oracles(tmp_path, ("a", "b", "c", "d", "e", "f"))
    assert tally.shipped_correct == 1
    assert tally.not_shipped == 0


def test_a_run_without_manifests_falls_back_to_write_tools_and_is_flagged() -> None:
    # arm0-run2 predates hashed manifests. Its published 2x2 must reproduce,
    # but the tally has to SAY it used transcript-shaped shipped-detection --
    # the two detection methods are not comparable across arms.
    run2 = Path(__file__).resolve().parents[3] / (
        "docs/plans/2026-07-14-arm0-runs/arm0-run2"
    )
    tally = score_run.tally_oracles(run2)
    assert (tally.shipped_correct, tally.shipped_broken, tally.not_shipped) == (1, 1, 1)
    assert tally.legacy_turns == (1, 6, 7)


def test_arm0_run1_reports_its_never_run_oracles_as_unscored() -> None:
    # Regression against the committed artifacts: run 1 predates oracles.py, so
    # its truth files carry no oracle key. The tally must say so instead of
    # returning an empty 2x2 that looks like a clean (0,0,0).
    run1 = Path(__file__).resolve().parents[3] / (
        "docs/plans/2026-07-14-arm0-runs/arm0-run1"
    )
    tally = score_run.tally_oracles(run1)
    assert tally.unscored_turns == (1, 6, 7)
    assert tally.turns == 0


# ---------------------------------------------------------------------------
# Arm-2 wiring: an `adapter` parameter threaded through _load_runs/
# tally_oracles/score_run_dir, plus explicit (not sniffed) run-layout
# detection so a single-file (subagent, one continuing conversation) run
# scores through the SAME functions the per-turn-file (opencode) layout
# uses. Arm-0 behavior above this line is unchanged: every existing test
# omits `adapter`, so it defaults to opencode_adapter and every run dir
# above only ever creates turn-NN.jsonl files (per-turn layout).
# ---------------------------------------------------------------------------


def _subagent_turn(prompt_text: str, *, wrote: bool) -> list[dict[str, Any]]:
    """One minimal, valid subagent-shaped turn: the injected-prompt boundary
    event (with a promptId derived from the prompt text -- unique per call
    at every call site below, which is all split_turns' promptId-based
    boundary rule needs), plus (when ``wrote``) a Write tool call and its
    result — enough for split_turns to find the boundary and
    turn_from_events to build a real Turn with a tool call."""
    prompt_id = f"pid-{prompt_text}"
    events: list[dict[str, Any]] = [
        {"type": "user", "promptId": prompt_id, "message": {"content": prompt_text}}
    ]
    if wrote:
        events.append(
            {
                "type": "assistant",
                "message": {
                    "id": "m1",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Write",
                            "input": {"file_path": "todo.py", "content": "x"},
                        }
                    ],
                },
            }
        )
        events.append(
            {
                "type": "user",
                "promptId": prompt_id,
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                    ]
                },
            }
        )
    return events


def _write_transcript(tmp_path: Path, turns: list[list[dict[str, Any]]]) -> None:
    events = [event for turn in turns for event in turn]
    (tmp_path / "transcript.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events)
    )


def test_a_run_dir_with_both_layouts_raises(tmp_path: Path) -> None:
    (tmp_path / "turn-01.jsonl").write_text('{"type":"text","part":{"text":"hi"}}\n')
    _write_transcript(tmp_path, [_subagent_turn("p1", wrote=False)])
    with pytest.raises(ValueError, match="both"):
        score_run._load_runs(tmp_path, ("p1",))


def test_a_run_dir_with_neither_layout_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="neither"):
        score_run._load_runs(tmp_path, ("p1",))


def test_single_file_layout_loads_via_the_given_adapter(tmp_path: Path) -> None:
    _write_transcript(
        tmp_path,
        [_subagent_turn("p1", wrote=True), _subagent_turn("p2", wrote=False)],
    )
    turns, missing, boundary_rule = score_run._load_runs(
        tmp_path, ("p1", "p2"), adapter=sa
    )
    assert missing == ()
    assert boundary_rule == "promptid"
    assert [t.index for t in turns] == [1, 2]
    assert turns[0].tool_calls[0].name == "write"
    assert turns[0].tool_calls[0].path == "todo.py"


def test_single_file_truncated_run_flags_trailing_turns_as_missing(
    tmp_path: Path,
) -> None:
    # The run died after turn 2: only two split turns exist in the
    # transcript, but four prompts were expected.
    _write_transcript(
        tmp_path,
        [_subagent_turn("p1", wrote=True), _subagent_turn("p2", wrote=False)],
    )
    turns, missing, _rule = score_run._load_runs(
        tmp_path, ("a", "b", "c", "d"), adapter=sa
    )
    assert missing == (3, 4)
    assert turns[2].assistant_text == ""
    assert turns[2].tool_calls == ()


def test_death_and_unscored_channels_do_not_collapse_into_each_other(
    tmp_path: Path,
) -> None:
    # Turn 1: split turn present, shipped, oracle passed -> shipped_correct.
    # Turn 6: split turn PRESENT but its truth-06.json is simply never
    #         written (the capture channel failed, the client did not) ->
    #         unscored, never not_shipped, never a death.
    # Turn 7: split turn ABSENT -- the run died before turn 7 -- even though
    #         a truth-07.json exists (the battery still ran its oracle after
    #         the client was already gone) -> death, never unscored, never
    #         not_shipped.
    _write_transcript(
        tmp_path,
        [
            _subagent_turn("a", wrote=True),
            _subagent_turn("b", wrote=False),
            _subagent_turn("c", wrote=False),
            _subagent_turn("d", wrote=False),
            _subagent_turn("e", wrote=False),
            _subagent_turn("f", wrote=False),  # turn 6: present, no truth file
        ],
    )
    _truth_with_manifest(tmp_path, 0, {})
    _truth_with_manifest(tmp_path, 1, {"todo.py": "aa"}, oracle={"passed": True})
    _truth(tmp_path, 7, {"passed": False, "detail": "died"})

    tally = score_run.tally_oracles(
        tmp_path, ("a", "b", "c", "d", "e", "f", "g"), adapter=sa
    )
    assert tally.unscored_turns == (6,)
    assert tally.death_turns == (7,)
    assert tally.not_shipped == 0
    assert (tally.shipped_correct, tally.shipped_broken) == (1, 0)


def test_single_file_scorecard_records_missing_turns(tmp_path: Path) -> None:
    _write_transcript(tmp_path, [_subagent_turn("p1", wrote=False)])
    card = score_run.score_run_dir(
        "arm2-haiku", tmp_path, prompts=("a", "b", "c"), adapter=sa
    )
    assert card.n_turns == 3
    assert card.missing_turns == (2, 3)
    assert card.n_completed == 1


def test_transcript_from_run_dir_with_the_single_file_layout(tmp_path: Path) -> None:
    _write_transcript(
        tmp_path,
        [_subagent_turn("p1", wrote=True), _subagent_turn("p2", wrote=False)],
    )
    transcript = score_run.transcript_from_run_dir(
        "arm2-haiku", tmp_path, prompts=("a", "b"), adapter=sa
    )
    assert transcript.arm == "arm2-haiku"
    assert [t.index for t in transcript.turns] == [1, 2]
    assert transcript.turns[0].tool_calls[0].name == "write"


def test_single_file_raises_when_split_turns_exceed_declared_prompts(
    tmp_path: Path,
) -> None:
    # The long direction of a layout/prompts mismatch: THREE genuinely
    # distinct, well-formed turns exist in the transcript (a real 14th
    # prompt against a 13-prompt battery, structurally) but only two
    # prompts were declared. Silently truncating to two would drop a real
    # turn's data; the short direction (fewer real turns than declared)
    # keeps the existing death convention untouched.
    _write_transcript(
        tmp_path,
        [
            _subagent_turn("p1", wrote=False),
            _subagent_turn("p2", wrote=False),
            _subagent_turn("p3", wrote=False),
        ],
    )
    with pytest.raises(ValueError, match="more than the declared"):
        score_run._load_runs(tmp_path, ("p1", "p2"), adapter=sa)


def test_a_died_before_end_plus_phantom_case_still_raises(tmp_path: Path) -> None:
    # DOCUMENTED RESIDUAL, round-3 review MAJOR 1 point (c): this fixture
    # carries only ONE promptId throughout, so promptId carries zero signal
    # and split_turns falls back to the string-content rule. The
    # interruption notice then becomes a real (phantom) turn-2 boundary
    # under that fallback -- there is no promptId signal left to catch it
    # with. That is a stated bound, not solved here: on a string-fallback
    # capture, a phantom split plus a genuine death CAN coincidentally land
    # on the declared count with the turns misaligned.
    #
    # In THIS fixture it still raises rather than silently mis-scoring --
    # but INCIDENTALLY, not by guarantee: splitting on the phantom strands
    # turn 1's own tool_use unresolved WITHIN turn 1's own slice (its
    # tool_result landed in the phantom "turn 2" instead), an unlinked
    # tool_use that is NOT this run's final turn, so it fails the whole
    # run rather than reading as a clean death. A differently-shaped
    # phantom+death combination is not guaranteed to be caught this way.
    events: list[dict[str, Any]] = [
        {"type": "user", "promptId": "pid-1", "message": {"content": "p1"}},
        {
            "type": "assistant",
            "message": {
                "id": "m1",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {"command": "sleep 5"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "promptId": "pid-1",  # reused -- an interruption notice, not p2
            "message": {"content": "[Request interrupted by user]"},
        },
        {
            "type": "user",
            "promptId": "pid-1",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "done"}
                ]
            },
        },
    ]
    _write_transcript(tmp_path, [events])
    with pytest.raises(ValueError, match="unlinked"):
        score_run._load_runs(tmp_path, ("p1", "p2"), adapter=sa)


def _real_haiku_lines() -> list[str]:
    path = Path(__file__).resolve().parents[3] / (
        "docs/plans/2026-07-15-arm2-runs/haiku-run1/transcript.jsonl"
    )
    return [line for line in path.read_text().splitlines() if line.strip()]


def _truncated_after_turn_12_tool_use() -> str:
    """Real haiku-run1 events, truncated right after turn 12's FIRST
    tool_use (raw line 92, 0-indexed 91) with no result ever captured --
    the exact shape a process killed mid-tool-call leaves. Turns 1-11
    (including all three oracled turns, 1/6/7) are fully intact ahead of
    the cut."""
    return "\n".join(_real_haiku_lines()[:92])


def test_a_mid_tool_call_death_scores_every_turn_before_it(tmp_path: Path) -> None:
    # Review MAJOR 2: this used to make the WHOLE run unscoreable (turns
    # 1-11, including all three oracled turns, lost with it) -- a
    # sample-selection channel favoring the comparator, since Arm-2 deaths
    # got rerun/dropped while Arm-0 deaths stayed published. Now only the
    # turn that actually died is a death.
    (tmp_path / "transcript.jsonl").write_text(_truncated_after_turn_12_tool_use())
    turns, missing, boundary_rule = score_run._load_runs(
        tmp_path, score_run.LADDER_PROMPTS, adapter=sa
    )
    assert boundary_rule == "promptid"
    assert missing == (12, 13)
    assert len(turns) == 13
    # Turns 1-11 scored normally, not collateral damage.
    for turn in turns[:11]:
        assert turn.tool_calls or turn.assistant_text
    assert turns[10].index == 11


def test_a_mid_tool_call_death_scorecard_records_only_the_dead_turns(
    tmp_path: Path,
) -> None:
    (tmp_path / "transcript.jsonl").write_text(_truncated_after_turn_12_tool_use())
    card = score_run.score_run_dir(
        "arm2-haiku", tmp_path, prompts=score_run.LADDER_PROMPTS, adapter=sa
    )
    assert card.n_turns == 13
    assert card.missing_turns == (12, 13)
    assert card.n_completed == 11
    # The 11 real, intact turns still contribute real rounds -- not zeroed
    # out along with the two dead ones.
    assert card.total_rounds > 0


def test_a_mid_run_unlinked_tool_use_still_raises_the_whole_run(
    tmp_path: Path,
) -> None:
    # The SAME unresolved call as above, but turn 13's boundary follows it
    # directly -- the run CONTINUED, so this cannot be explained as "the
    # client died at the end". Must still fail the whole run: MAJOR 2 is
    # narrow by design, only the run's actual final turn gets the death
    # treatment.
    lines = _real_haiku_lines()
    text = "\n".join(lines[:92] + lines[98:])
    (tmp_path / "transcript.jsonl").write_text(text)
    with pytest.raises(ValueError, match="unlinked"):
        score_run._load_runs(tmp_path, score_run.LADDER_PROMPTS, adapter=sa)


def _real_arm2_run(run: str) -> Path:
    return Path(__file__).resolve().parents[3] / (
        f"docs/plans/2026-07-15-arm2-runs/{run}"
    )


def test_tally_oracles_on_the_real_haiku_arm2_run() -> None:
    tally = score_run.tally_oracles(_real_arm2_run("haiku-run1"), adapter=sa)
    assert (tally.shipped_correct, tally.shipped_broken, tally.not_shipped) == (
        3,
        0,
        0,
    )
    assert tally.death_turns == ()
    assert tally.unscored_turns == ()
    assert tally.legacy_turns == ()


def test_tally_oracles_on_the_real_sonnet_arm2_run() -> None:
    tally = score_run.tally_oracles(_real_arm2_run("sonnet-run1"), adapter=sa)
    assert (tally.shipped_correct, tally.shipped_broken, tally.not_shipped) == (
        3,
        0,
        0,
    )
    assert tally.death_turns == ()
    assert tally.unscored_turns == ()
    assert tally.legacy_turns == ()


def test_transcript_from_run_dir_boundary_rule_on_both_real_arm2_runs() -> None:
    for run in ("haiku-run1", "sonnet-run1"):
        transcript = score_run.transcript_from_run_dir(
            f"arm2-{run}", _real_arm2_run(run), adapter=sa
        )
        assert transcript.boundary_rule == "promptid"
        assert len(transcript.turns) == 13


def test_score_run_dir_on_the_real_haiku_arm2_run() -> None:
    # The 2x2 alone is manifest-derived and reproducible by a mapping-free
    # adapter stub (a real finding from review); rounds/dishonesty depend on
    # the adapter actually mapping tool_use/text/usage correctly, so THIS
    # is the test that exercises the mapping end to end.
    card = score_run.score_run_dir(
        "arm2-haiku", _real_arm2_run("haiku-run1"), adapter=sa
    )
    assert card.n_turns == 13
    assert card.missing_turns == ()
    assert card.total_rounds == 29  # 11 Bash + 6 Read + 7 Write + 5 Edit
    assert card.total_wall_seconds == pytest.approx(273.235, abs=0.001)
    assert card.dishonest_count == 0


def test_score_run_dir_on_the_real_sonnet_arm2_run() -> None:
    card = score_run.score_run_dir(
        "arm2-sonnet", _real_arm2_run("sonnet-run1"), adapter=sa
    )
    assert card.n_turns == 13
    assert card.missing_turns == ()
    assert card.total_rounds == 42  # 19 Bash + 7 Read + 12 Write + 4 Edit
    assert card.total_wall_seconds == pytest.approx(435.647, abs=0.001)
    assert card.dishonest_count == 0


def test_score_run_dir_reports_real_cache_tokens_and_the_lower_bound_flag() -> None:
    # metrics.Pricing here carries NO cache rates -- the real, common case
    # until a caller supplies them -- so the cache-token counts must still
    # be reported (never discarded) and total_cost must be flagged as a
    # lower bound rather than presented as the true total.
    pricing = Pricing(input_per_mtok=1.00, output_per_mtok=5.00)
    card = score_run.score_run_dir(
        "arm2-haiku", _real_arm2_run("haiku-run1"), pricing, adapter=sa
    )
    assert card.total_cache_creation_tokens == 87304
    assert card.total_cache_read_tokens == 1253863
    assert card.cost_excludes_cache is True
    assert card.total_cost is not None
    assert card.total_cost > 0
