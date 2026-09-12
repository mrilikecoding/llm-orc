#!/bin/zsh
S=/private/tmp/claude-501/-Users-nathangreen-Development-eddi-lab-llm-orc/32dba98e-5297-4a48-8e23-9f8b5e88d5e8/scratchpad
O=$S/think-ab; touch $S/BENCH-RUNNING; trap "rm -f $S/BENCH-RUNNING" EXIT
until curl -s -m 3 localhost:8777/v1/models >/dev/null; do sleep 2; done
TOOLS=$(python3 -c "import json; print(json.dumps(json.load(open('/Users/nathangreen/Development/eddi-lab/llm-orc/docs/plans/2026-08-30-168-live-gate/168-ask.json'))['tools']))")
run() { # port ask tag
  local port=$1 ask=$2 tag=$3
  local body=$(python3 -c "import json,sys; print(json.dumps({'model':'agentic','messages':[{'role':'user','content':sys.argv[1]}],'tools':json.loads(sys.argv[2])}))" "$ask" "$TOOLS")
  local t0=$(date +%s.%N)
  curl -s -m 900 localhost:$port/v1/chat/completions -H 'content-type: application/json' -d "$body" > $O/$tag.json
  local t1=$(date +%s.%N)
  printf '%s\t%s\t%.1f\t%s\n' "$tag" "$port" "$(echo "$t1 - $t0" | bc)" "$(python3 -c "import json,sys; d=json.load(open('$O/$tag.json')); m=d['choices'][0]['message']; print(m.get('tool_calls') and 'tool_calls:'+m['tool_calls'][0]['function']['name'] or 'stop:'+(m.get('content') or '')[:60].replace(chr(10),' '))" 2>&1)" >> $O/results.tsv
}
A1="write a function that adds two numbers in add.py"
A2="create storage.py with save_todos and load_todos functions using json"
run 8765 "$A1" s1-main-thinkon
run 8777 "$A1" s1-thinkoff
run 8777 "$A2" s2-thinkoff
run 8765 "$A2" s2-main-thinkon
run 8765 "$A1" s1b-main-thinkon
run 8777 "$A1" s1b-thinkoff
echo DONE >> $O/results.tsv
