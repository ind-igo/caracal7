#!/bin/sh
# Runs every tests/test_*.mojo and builds every bench, warnings as errors; exits non-zero on the first failure.
# Extra arguments go to mojo. The E16 field switch is the constant in src/caracal7/core/field.mojo.
set -e
for t in tests/test_*.mojo; do
  echo "== $t $*"
  uv run mojo run --Werror "$@" -I src "$t"
done
for b in bench/*.mojo; do
  echo "== build $b"
  uv run mojo build --Werror "$@" -I src "$b" -o /dev/null
done
