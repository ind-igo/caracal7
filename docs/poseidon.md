# Poseidon-M31 workload

The fourth `Workload` (`workloads/poseidon.mojo`): the expander's Poseidon over Mersenne-31 (width 16, rate 8,
8 full and 14 partial rounds, x^5, circulant MDS of small constants, keccak-derived round constants), the
Poseidon entry of the csp-benchmarks suite at 2 to 16 field elements. 78 bit columns, 4 Horner accumulators,
one chain-end identity, 24 opening points, 5 dense public columns and about 70 tiny periodic ones.

## The idea

A round is: add the constants, apply the matrix, apply the S-box (every lane in a full round, lane 0 in a
partial one). A chain is one lane of one round, 64 rows of bit weights (row `r` holds bit `62 - r`, row 63
idle because the Horner scan never ingests the last row). The grid's `h2 = 16 R` chains are 16 segments of `R`
rounds, segment `s` round `t` at chain `s R + t`: the next round is the next chain, the other lanes of a round
sit `R` chains apart, cyclically. Three things are new against SHA-256.

- **Products on a bit layout.** The S-box needs `mf^2`, `mf^4`, `mf^5`: three bit convolutions, each
  certified by a Horner fingerprint as in mulmod (coefficient bits at their own weight, 5 bits each because
  operands below `2^31 + 5` keep every coefficient below 32), rippled with 5-bit carries and folded twice by
  `2^31 = 1`. The three coefficient sets share one accumulator weighted by `rho^k`, so the chain end is one
  identity, `R_mf^2 + rho R_p2^2 + rho^2 R_p4 R_mf = R_C`, and Z is 4 accumulators (64 columns).
- **An anti-circulant matrix across chains.** `M[i][j] = row[(i + j) % 16]`, so the coefficient of lane
  `j` in output `i` depends on `i + j`, not `j - i`, and a uniform read offset cannot carry a constant
  coefficient. Round `t` stores lane `sigma_t s` at segment `s` with `sigma_t = (-1)^t` and computes at
  segment `s` the output lane the next round wants there: the read at chain offset `e R` then has the
  coefficient `row[(sigma_t e) % 16]`, a function of the offset and the round's parity only. Period-2
  public bit columns `g_e_k` supply it; six shifted copies `u << k` on the source chain turn the
  bit-by-bit sum into reads at `k1 = 0`, so the matrix costs 15 opening points, not 105. The pile splits
  into even and odd offsets (23 and 17 bits per row at most) so 5-bit carries suffice.
- **The absorb and the digest ride on existing families.** The chunk word enters the constant-addition
  pile as `ab w` (a bounded witness column; the selector is on the `n` input slots of rounds 1 and 23 only, so the padding is the selector's zero, not the prover's); the digest
  is the S-box output of the last round, `ld v = dg` against a dense public column; the transition
  `x[next] = v` is cut at segment ends by `nl`. Round 0 is idle with `z0 x = 0`; rounds past the last
  permutation continue as garbage rounds nobody reads.

## Decisions

- **Mersenne-31, not BN254.** The suite fixes the hash and the input count, not the field: circom and
  provekit hash BN254 elements, plonky2 Goldilocks, expander M31. BN254 would need a product op with a
  witness quotient (no sparse fold) and a linear-combination op for the dense MDS; M31 needs neither, the
  existing fold shape does it, and the prover work is about 20x smaller. The digest matches the expander
  entry (its test vectors are the tests).
- **Lanes along chains, not rows.** With lanes along rows one Horner over the chain would mix the lanes'
  operands and the product identity picks up cross terms; the fingerprint must be per lane, so a lane is a
  chain. The price is the cross-chain matrix, paid as above.
- **Canonical state.** The last fold of the matrix output subtracts the modulus under a chain-constant bit
  `b` (`lo + hi = mf + b M31` as integers, so `b = 1` only when the sum reaches the modulus), the honest
  state is canonical every round and the digest column is a plain equality; a prover choosing `b = 0`
  keeps `mf < 2^31 + 2`, inside every bound.
- **Exactness by masks, no zero rows.** Every ripple's carry into weight 0 is cut by the row selector `nb`
  (zero on rows 62 and 63), the folds read the high half through `lo` (weights 0..30) and `hs` (0..31), and
  the idle row's piles are masked to zero, so its free cells (a product's coefficient and carry bits) feed
  nothing live.
- **Grid.** `CLIENT.grid(64, 16 (23 + k))`: one chunk (up to 8 inputs) needs `R >= 23`, so 368 chains pad
  to 384 (`R = 24`); two chunks need `R >= 45`, 720 pads to 896 (`R = 56`). `R` must be even for the
  parity columns (`h2 % 32 == 0`).

## Column order

`x s sc0 sc1` the state and the constant sum with its carries, `u u1..u6` the folded sum and its shifted
copies, `uc`, `ha hac0..4 hb hbc0..4` the two matrix piles, `y yc mo moc mf mfc b` the sum, its folds and
the modulus bit, then per product `p2 p4 p5`: `c0..c4` coefficient bits, `y0..y4` carries, `r o oc f fc`;
`v` the S-box selection, `w` the chunk word.

## Measured (M1 Pro)

| inputs | grid | proof | warm prove | verify |
|---|---|---|---|---|
| 2 | 64 x 384 | 421 KB | 100 ms | 32 ms |
| 8 | 64 x 384 | 421 KB | 89 ms | 34 ms |
| 16 | 64 x 896 | 457 KB | 152 ms | 40 ms |

## Tests

`test_poseidon`: the expander's two test vectors (8 and 16 copies of 114514); every family without a
challenge on the host at 384 chains (3 and 8 inputs) and 896 (16), with the last round's `v` against the
reference digest; the prover round trip for one and two chunks; a proof against a wrong digest rejected; a trace with
coefficients off by one on the S-box-free lanes (every row family holds) rejected by the chain-end identity;
the public data refusing a non-canonical digest word.
