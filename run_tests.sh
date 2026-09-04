#!/bin/sh
# Runs every tests/test_*.mojo; exits non-zero on the first failure.
set -e
for t in tests/test_*.mojo; do
  echo "== $t"
  mojo run -I src "$t"
done
