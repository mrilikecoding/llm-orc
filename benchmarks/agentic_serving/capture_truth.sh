#!/bin/zsh
# Per-turn workspace ground truth, shared by EVERY arm's driver (#131).
#
# Extracted from ladder_battery.sh so a second driver (Arm 2, the Claude Code
# subagent harness) cannot drift the truth substrate: whatever drives the
# turns, the recorded facts — hashed manifest, suite/seeded pytest rc, hidden
# oracle verdict, oracle contamination — come from THIS script, identically.
#
# Usage:
#   TRUTH_REPO=<fixture repo> TRUTH_OUT=<out dir> [TRUTH_PYTEST=pytest] \
#     capture_truth.sh <turn-index>
#
# turn-index 0 records the SEEDED baseline (truth-00.json, manifest only);
# turn-index N (1-based) records truth-NN.json after that turn ran. Appends
# to $TRUTH_OUT/oracle-exits.tsv (create/reset it in the driver).
#
# Design notes live with the code below and in ladder_battery.sh's header:
# pytest truths run in a throwaway COPY (arm-authored test code never executes
# in the live workspace); the manifest is hashed so the scorer derives
# "shipped" from disk, the only channel that means the same thing for every
# arm; the oracle runs NOW, never post-hoc (later turns mutate files); the
# post-oracle manifest is recorded so an oracle write-through is never
# attributed to the arm.
set -u
REPO=${TRUTH_REPO:?set TRUTH_REPO to the fixture repo}
OUT=${TRUTH_OUT:?set TRUTH_OUT to the output dir}
PYTEST=${TRUTH_PYTEST:-pytest}
# Absolutize + git-guard (same quiet-corruption family as a relative $REPO:
# without a git repo, ls-files dies but the pipeline emits an empty-yet-valid
# manifest and every oracled turn scores not_shipped with no flag).
REPO=${REPO:A}
OUT=${OUT:A}
git -C "$REPO" rev-parse --is-inside-work-tree > /dev/null 2>&1 \
  || { echo "TRUTH_REPO must be a seeded GIT repo: $REPO"; exit 1; }
SRCROOT=${0:A:h:h:h}
i=${1:?usage: capture_truth.sh <turn-index>}
n=$(printf %02d "$i")

# Hashed workspace manifest: one `path<TAB>sha256` per line. Existence-filtered
# because `git ls-files --cached` keeps listing a tracked file the arm deleted.
# quotepath=off + -z so a filename with spaces, tabs, or non-ASCII survives;
# `./$f` keeps a leading-dash name out of shasum's option parser. Known bound:
# a newline in a filename cannot ride a line-based manifest and is skipped.
# Readers parse with rpartition on the LAST tab.
manifest() {
  (cd "$REPO" && git -c core.quotepath=off ls-files -z --others --cached --exclude-standard \
    | while IFS= read -r -d '' f; do
        case $f in
          (.*|*/.*|*__pycache__*|*.pyc|*$'\n'*) continue ;;
        esac
        [ -f "$f" ] && printf '%s\t%s\n' "$f" "${$(shasum -a 256 "./$f")%% *}"
      done | sort -u)
}

# Ground-truth pytest in a THROWAWAY COPY of the workspace.
# $1 = output file; remaining args go to pytest. Returns pytest's own rc.
# ${=PYTEST}: zsh does not word-split parameters, so a multi-word value
# ("uv run pytest") would otherwise exec a single 127-everywhere command.
run_truth_pytest() {
  local out_file=$1
  shift
  local tws rc
  tws=$(mktemp -d) || return 127
  cp -R "$REPO"/. "$tws"
  (cd "$tws" && timeout 120 ${=PYTEST} -q "$@" > "$out_file" 2>&1)
  rc=$?
  rm -rf "$tws"
  return $rc
}

if [ "$i" -eq 0 ]; then
  # Baseline manifest of the SEEDED repo, before any turn runs: turn 1's
  # shipped-detection diffs against this.
  manifest > "$OUT/.files-00.txt"
  python3 - "$OUT/truth-00.json" "$OUT/.files-00.txt" <<'PY'
import json, sys

path, files_path = sys.argv[1:3]
manifest = {}
with open(files_path) as handle:
    for line in handle.read().splitlines():
        name, _, digest = line.rpartition("\t")
        manifest[name] = digest
with open(path, "w") as out:
    json.dump({"turn": 0, "files": sorted(manifest), "manifest": manifest}, out,
              indent=1)
PY
  rm -f "$OUT/.files-00.txt"
  exit 0
fi

# Workspace ground truth AFTER turn $i: the full suite, the seeded target
# alone, and the hashed file manifest. Exit codes come from pytest itself: a
# pipeline's $? is the LAST command's (tail always succeeds), so each run goes
# to a file and is tailed after.
run_truth_pytest "$OUT/.suite-$n.txt"
suite_rc=$?
run_truth_pytest "$OUT/.seeded-$n.txt" test_buggy.py
seeded_rc=$?
suite=$(tail -12 "$OUT/.suite-$n.txt"); rm -f "$OUT/.suite-$n.txt"
seeded=$(tail -8 "$OUT/.seeded-$n.txt"); rm -f "$OUT/.seeded-$n.txt"
# One manifest entry per LINE into a file, never a space-joined string: a
# model can write "my notes.py", and a space-joined list round-tripped
# through str.split() would silently corrupt it into two entries.
manifest > "$OUT/.files-$n.txt"
# The turn's hidden correctness oracle, run NOW against the workspace this
# turn produced. It cannot be deferred to the end of the run: later turns
# mutate files (turn 13 rewrites buggy.py), so a post-hoc probe would judge a
# turn against a workspace it never saw. The module prints `null` itself for a
# turn with no oracle, so there is deliberately NO `|| echo null` fallback
# here: that would make a real crash in the measurement instrument
# indistinguishable from "no oracle by design". A crash leaves a non-JSON
# stdout (recorded as `oracle: null` with the reason in oracle-NN.err) and a
# nonzero code visible in oracle-exits.tsv.
oracle=$(cd "$SRCROOT" && uv run python -m benchmarks.agentic_serving.oracles \
  "$i" "$REPO" 2> "$OUT/oracle-$n.err")
orc=$?
printf '%s\t%s\n' "$n" "$orc" >> "$OUT/oracle-exits.tsv"
# A SIGKILLed oracle leaves EMPTY stderr; keep the err file truthful rather
# than deleting the only breadcrumb next to a nonzero code.
if [ "$orc" -ne 0 ] && [ ! -s "$OUT/oracle-$n.err" ]; then
  echo "oracle exited $orc with no stderr" > "$OUT/oracle-$n.err"
fi
[ -s "$OUT/oracle-$n.err" ] || rm -f "$OUT/oracle-$n.err"
# The probe sandbox relocates cwd but cannot stop a hardcoded ABSOLUTE path
# in arm code from writing into the real workspace. Recapture the manifest:
# any difference is oracle contamination, recorded on THIS turn so the next
# turn's shipped-diff can discount it instead of attributing it to the arm.
manifest > "$OUT/.files-post-$n.txt"
cmp -s "$OUT/.files-$n.txt" "$OUT/.files-post-$n.txt" \
  || echo "!!! oracle contamination on turn $i (recorded in truth-$n.json)"
(cd "$REPO" && rm -rf .pytest_cache && find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null)
python3 - "$OUT/truth-$n.json" "$i" "$suite_rc" "$seeded_rc" \
  "$OUT/.files-$n.txt" "$OUT/.files-post-$n.txt" "$suite" "$seeded" \
  "$oracle" <<'PY'
import json, sys

(path, turn, suite_rc, seeded_rc,
 files_path, post_path, suite, seeded, oracle) = sys.argv[1:10]

def read_manifest(manifest_path):
    entries = {}
    with open(manifest_path) as handle:
        for line in handle.read().splitlines():
            name, _, digest = line.rpartition("\t")
            entries[name] = digest
    return entries

manifest = read_manifest(files_path)
post = read_manifest(post_path)
contamination = sorted(
    {name for name in post if post[name] != manifest.get(name)}
    | {name for name in manifest if name not in post}
)
try:
    oracle_verdict = json.loads(oracle)
except ValueError:
    # A crash in the oracle module itself, not "no oracle for this turn". The
    # reason is in oracle-NN.err and the code in oracle-exits.tsv.
    oracle_verdict = None
record = {
    "turn": int(turn),
    "files": sorted(manifest),
    "manifest": manifest,
    # The POST-oracle state is what the NEXT turn's arm actually starts from;
    # the scorer diffs turn N+1 against this, so an oracle write-through is
    # never attributed to the arm and a genuine arm edit still counts.
    "post_manifest": post,
    "suite": {"rc": int(suite_rc), "tail": suite},
    "seeded": {"rc": int(seeded_rc), "tail": seeded},
    "oracle": oracle_verdict,
}
if contamination:
    record["oracle_contamination"] = contamination
with open(path, "w") as out:
    json.dump(record, out, indent=1)
PY
rm -f "$OUT/.files-$n.txt" "$OUT/.files-post-$n.txt"
echo "--- truth $n | suite_rc $suite_rc | seeded_rc $seeded_rc ---"
