#!/bin/sh
# Runs every tests/test_*.mojo and builds every bench, warnings as errors, JOBS files at a time (default 4);
# exits non-zero if any fails and prints that file's log. Extra arguments go to mojo (e.g. -D CARACAL_NVIDIA_MMA).
# BENCH=0 skips the bench builds. The E16 field switch is the constant in src/caracal7/core/field.mojo.
# The cost of a file is its distinct Params shapes: every grid instantiates the whole kernel set again.
set -u
JOBS=${JOBS:-4}
LOG=$(mktemp -d)
one() {   # $1 = test or bench file; the rest = mojo args
  f=$1; shift
  t0=$(date +%s)
  case "$f" in
    tests/*) uv run mojo run --Werror "$@" -I src "$f" ;;
    *)       uv run mojo build --Werror "$@" -I src "$f" -o /dev/null ;;
  esac > "$LOG/$(basename "$f").log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then echo "ok   $f  ${dt}s"; else echo "FAIL $f  ${dt}s"; cat "$LOG/$(basename "$f").log"; echo "$f" >> "$LOG/failed"; fi
}
export -f one 2>/dev/null || true
FILES=$(ls tests/test_*.mojo)
[ "${BENCH:-1}" = 0 ] || FILES="$FILES $(ls bench/*.mojo)"
for f in $FILES; do
  while [ "$(jobs -p | wc -l)" -ge "$JOBS" ]; do sleep 0.2; done
  one "$f" "$@" &
done
wait
if [ -f "$LOG/failed" ]; then echo "FAILED: $(tr '\n' ' ' < "$LOG/failed")"; rm -rf "$LOG"; exit 1; fi
rm -rf "$LOG"; echo "all passed"
