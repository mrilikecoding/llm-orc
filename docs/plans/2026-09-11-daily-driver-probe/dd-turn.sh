#!/bin/zsh
# dd-turn.sh N "prompt" [cont]  — one daily-driver turn through real OpenCode against the serve.
N=$1; P=$2; CONT=$3
S=/private/tmp/claude-501/-Users-nathangreen-Development-eddi-lab-llm-orc/32dba98e-5297-4a48-8e23-9f8b5e88d5e8/scratchpad
R=$S/dd-repo; OUT=$S/dd-out; PY=/Users/nathangreen/Development/eddi-lab/llm-orc/.venv/bin/python
n=$(printf %02d $N)
cd $R || exit 9; touch $OUT/RUNNING; trap "rm -f $OUT/RUNNING" EXIT
echo "$P" > $OUT/ask-$n.txt
if [ "$CONT" = "cont" ]; then
  gtimeout 780 opencode run --format json -c -m llm-orc/agentic "$P" > $OUT/turn-$n.jsonl 2> $OUT/turn-$n.err < /dev/null
else
  gtimeout 780 opencode run --format json -m llm-orc/agentic "$P" > $OUT/turn-$n.jsonl 2> $OUT/turn-$n.err < /dev/null
fi
rc=$?
{ echo "exit=$rc"; echo "--- git status ---"; git status --short; echo "--- diff stat ---"; git diff --stat; git ls-files --others --exclude-standard | sed 's/^/untracked: /';
  echo "--- pytest (throwaway copy) ---"; T=$(mktemp -d); cp -R $R/. $T/; (cd $T && $PY -m pytest -q 2>&1 | tail -15); rm -rf $T; } > $OUT/truth-$n.txt 2>&1
echo $rc > $OUT/exit-$n
