#!/usr/bin/env bash
set -euo pipefail
set -f
: "${STATE_JSON:?STATE_JSON is required}"
DIR="$(cd "$(dirname "$0")" && pwd)"
set -- $(jq -r '.target, .size, .proof, .args[]' "$STATE_JSON")     # one jq call: the fields are hex and decimals, no spaces
exec "$DIR/target/caracal7" prove "$@"
