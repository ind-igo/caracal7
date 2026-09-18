#!/usr/bin/env bash
# The prover checks secp256k1 signatures; the utils generator's `ecdsa` command is secp256r1, so the
# vector is fixed here (digest, x_Q, y_Q, r||s: the same lines the generator prints) until it grows a curve flag.
set -euo pipefail
V=()
while read -r x; do V+=("$x"); done < "$(dirname "$0")/ecdsa_secp256k1_vector.txt"
exec "$(dirname "$0")/common_prepare.sh" ecdsa "${V[@]}"
