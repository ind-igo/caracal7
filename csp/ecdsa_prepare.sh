#!/usr/bin/env bash
# The prover checks secp256k1 signatures; the utils generator's `ecdsa` command is secp256r1, so the
# vector is fixed here (digest, x_Q, y_Q, r||s: the same lines the generator prints) until it grows a curve flag.
set -euo pipefail
DIR="$(dirname "$0")"
mapfile -t V < "$DIR/ecdsa_secp256k1_vector.txt"
exec "$DIR/common_prepare.sh" ecdsa "${V[0]}" "${V[1]}" "${V[2]}" "${V[3]}"
