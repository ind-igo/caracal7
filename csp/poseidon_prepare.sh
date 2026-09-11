#!/usr/bin/env bash
# The prover hashes over Mersenne-31 (the expander's Poseidon instance). The utils generator prints
# 254-bit elements; each is reduced mod 2^31 - 1 until the generator grows an m31 flag.
set -euo pipefail
: "${UTILS_BIN:?UTILS_BIN is required}"
ELEMS=()
while read -r x; do ELEMS+=("$(echo "$x % 2147483647" | bc)"); done < <("$UTILS_BIN" poseidon -n "$INPUT_SIZE")
exec "$(dirname "$0")/common_prepare.sh" poseidon "${ELEMS[@]}"
