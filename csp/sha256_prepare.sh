#!/usr/bin/env bash
set -euo pipefail
: "${UTILS_BIN:?UTILS_BIN is required}"
MSG="$("$UTILS_BIN" sha256 -n "$INPUT_SIZE" | sed -n '1p')"
exec "$(dirname "$0")/common_prepare.sh" sha256 "$MSG"
