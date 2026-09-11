#!/usr/bin/env bash
set -euo pipefail
: "${STATE_JSON:?STATE_JSON is required}"
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/target/caracal7" prove "$(jq -r .target "$STATE_JSON")" "$(jq -r .size "$STATE_JSON")" "$(jq -r .proof "$STATE_JSON")" $(jq -r '.args[]' "$STATE_JSON")
