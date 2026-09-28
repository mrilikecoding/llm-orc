"""Fails on its first invocation, succeeds on every one after.

Loop iteration_failures pin fixture (lf2): the counter lives at the path
named by LLM_ORC_FLAKY2_COUNTER, which the test sets to a fresh pytest
tmp_path each run — never a fixed or hardcoded scratch path, and never
state that survives between test runs.
"""

import os
import sys

sys.stdin.read()

path = os.environ["LLM_ORC_FLAKY2_COUNTER"]
count = int(open(path).read()) if os.path.exists(path) else 0
with open(path, "w") as handle:
    handle.write(str(count + 1))

if count == 0:
    sys.stderr.write("first iteration boom\n")
    sys.exit(2)

print('{"ok": true}')
