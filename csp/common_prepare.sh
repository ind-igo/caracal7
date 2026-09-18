#!/usr/bin/env bash
# Writes STATE_JSON for prove.sh / verify.sh: {target, size, proof, args: [...]}.
# Usage: common_prepare.sh <target> <arg>...   (env: INPUT_SIZE, STATE_JSON)
set -euo pipefail
: "${INPUT_SIZE:?INPUT_SIZE is required}"
: "${STATE_JSON:?STATE_JSON is required}"
TARGET="$1"; shift
PROOF="${TMPDIR:-/tmp}/caracal7_${TARGET}_${INPUT_SIZE}.proof"
jq -nc --arg t "$TARGET" --argjson n "$INPUT_SIZE" --arg p "$PROOF" --args '{target:$t, size:$n, proof:$p, args:$ARGS.positional}' -- "$@" > "$STATE_JSON"
