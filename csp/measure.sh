#!/usr/bin/env bash
# Sizes: the proof file; preprocessing 0 (the compiled statement is derived at run time, nothing is
# persisted). Circuit size: committed witness cells, printed by the prover.
set -euo pipefail
: "${STATE_JSON:?STATE_JSON is required}"
: "${SIZES_JSON:?SIZES_JSON is required}"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$("$DIR/prove.sh" 2>/dev/null)"
PROOF="$(jq -r .proof "$STATE_JSON")"
jq -n --argjson p "$(stat -f %z "$PROOF" 2>/dev/null || stat -c %s "$PROOF")" '{proof_size: $p, preprocessing_size: 0}' > "$SIZES_JSON"
CELLS="$(printf '%s\n' "$OUT" | sed -n 's/.* cells \([0-9]*\).*/\1/p')"
SIZES="$DIR/circuit_sizes.json"; [[ -f "$SIZES" ]] || echo '{}' > "$SIZES"
jq --arg t "$(jq -r .target "$STATE_JSON")" --arg n "$(jq -r .size "$STATE_JSON")" --argjson c "$CELLS" '.[$t][$n] = $c' "$SIZES" > "$SIZES.tmp" && mv "$SIZES.tmp" "$SIZES"
cat "$SIZES_JSON"
