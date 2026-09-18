#!/usr/bin/env bash
set -euo pipefail
: "${UTILS_BIN:?UTILS_BIN is required}"
GEN="$("$UTILS_BIN" keccak -n "$INPUT_SIZE")"
exec "$(dirname "$0")/common_prepare.sh" keccak "$(sed -n '1p' <<< "$GEN")" "$(sed -n '2p' <<< "$GEN")"
