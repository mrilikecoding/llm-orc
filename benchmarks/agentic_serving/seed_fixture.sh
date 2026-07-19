#!/bin/sh
# Seed a fresh ladder fixture repo (#131), identical for every arm.
#
# The four seed files live in fixture_seed/ and are hash-pinned below to the
# EXACT bytes of the Arm-0 runs' fixtures (verified against the committed
# truth-01 manifests of arm0-run3/4, docs/plans/2026-07-15-arm0-runs/), so a
# fixture seeded by this script is byte-identical to the one the recorded
# Arm-0 column ran against. If a seed file is ever edited, the pin fails loud
# here rather than silently forking the fixture between arms.
#
# Usage: seed_fixture.sh <target-dir>   (created; must not already exist)
set -eu
TARGET=${1:?usage: seed_fixture.sh <target-dir>}
[ -e "$TARGET" ] && { echo "target already exists: $TARGET" >&2; exit 1; }
SRCDIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/fixture_seed

expected="\
ad48e6dac4cafb2a3342bc3b1d60910a562317b7c04b4c6888ee5bc83e02b7e9  buggy.py
9b4886a7d55129066d955cc4d606da83bdcea42c883d1c3e4305711f9bedb7a4  calc.py
b1c829eadce845cd99f644c1a9c042053d04ac88c3a40e204e75fde59a713050  metrics.py
fdf2354a945f96a905d9a321fe537de73777d489d9eceeea28f90620054bf6e9  test_buggy.py"
actual=$(cd "$SRCDIR" && sha256sum buggy.py calc.py metrics.py test_buggy.py)
[ "$expected" = "$actual" ] || {
  echo "fixture_seed/ hashes diverged from the pinned Arm-0 fixture:" >&2
  echo "$actual" >&2
  exit 1
}

mkdir -p "$TARGET"
cp "$SRCDIR"/buggy.py "$SRCDIR"/calc.py "$SRCDIR"/metrics.py \
  "$SRCDIR"/test_buggy.py "$TARGET"/
git -C "$TARGET" init -q
git -C "$TARGET" add -A
git -C "$TARGET" -c user.email=ladder@fixture -c user.name=ladder \
  commit -qm "seed ladder fixture"
echo "seeded: $TARGET"
