#!/usr/bin/env bash
# Builds the prover binary next to these scripts. Needs uv (https://docs.astral.sh/uv/) and an Apple GPU.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${CARACAL7_REPO:-$DIR/..}"
mkdir -p "$DIR/target"
(cd "$REPO" && uv run mojo build -I src cli/main.mojo -o "$DIR/target/caracal7")
