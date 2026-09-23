# Soundness

Status: **conditional analysis, not certification**. The numbers below are what
`bench/bench_soundness.mojo` computes from the compiled statements; the conditions they rest on
are listed at the end. Nothing here is a measured attack cost.

## The numbers

Every benchmark statement, 22 cases (the 16 csp-benchmarks cases on their 12 grids, P-256, the
125-hash chain grid, RSA-2048, SOD, DSC and the passport), at `E = F_(127^20)`,
112-bit query target per level, 20 grinding bits:

| | all 22 cases | ECDSA |
|---|---:|---:|
| interactive bits, ideal challenges, no credit for grinding | 87.50 to 88.37 | 88.21 |
| work bits, the query error discounted by the 20 grinding bits | 107.12 to 108.18 | 108.03 |

The two rows are the same error split two ways:

```
Q = 127^20      A = sum of the field numerators      P = query error per attempt
interactive bits = -log2(P + A/Q)
work bits        = -log2(P / 2^20 + A/Q)
```

The interactive row is the bound on a prover that gets fresh uniform challenges and cannot
grind. The work row credits the 20 bits of proof of work on every query seed; that credit
assumes an adversary must pay one hash per attempt, which is a statement about the
Fiat-Shamir compilation and is not proved (P5 below). On every case the query term sets both
rows: undiscounted it is about `2^-88`, discounted about `2^-108`, and the field term `A/Q` lies
between `2^-108` and `2^-117`. More queries would raise the interactive row; on the SOD and DSC
the field term already exceeds the discounted query term, so their work row is capped near 108.

These are the executable's `johnson_dkt26_joint_list_conditional` row, the least conservative of
the rows it prints. With only the numerator swapped for Haböck's (ePrint 2025/2110, Theorem 2) it
gives 105.7 to 106.5 work bits; with Haböck's numerator and the older per-column list caps (16 to
30 instead of 4 to 7) 81.0 to 99.5. The ledger covers these 22 statements only: it rejects a
statement with Herder lookup or permutation accumulators, split codewords or another tail arity.

## What is charged

The prover commits Reed-Solomon codewords over `F4` at rate at most 1/4 (level 1) and over `E`
at rate at most 1/8 (the tail), opens `s_i` rows per level at the Johnson radius `1 - sqrt(rate) - 1/16`, and
folds the tail three binary digits per level. The ledger charges, per case:

- **Queries.** Per level, `(sqrt(K_i/n_i) + 1/16)^s_i`, times the list cap of the next
  committed level (one after the last). The query count `s_i` is sized so the level-1 term is
  below `2^-92`; the levels sum to `P`.
- **Correlated agreement.** Level 1: `4 (C + 1)`, `C` the DKT26 Theorem 5.12 numerator, one
  per conjugate code of the `E tensor F4` alphabet; each tail level: `3 C`, one per tensor factor. The numerator is Lemma 5.3's, at the compiled integer radius.
- **Lists.** The joint block-list cap `L` per level, a sufficient integer cap (not a claim that
  this many close codewords exist): `L = ceil(n (T - D) / (T^2 - n D))` with `D = K - 1` the
  degree and `T = ceil(sqrt(n K) + n/16)` the agreement threshold; 4 to 7 on these cases.
  Bind (two list members colliding at the opening point): `opened * L(L-1)/2 * (h1+h2-2)`.
  The opening batch `2L`, and the batching and sumcheck allowances at each tail root times that
  root's cap.
- **Relations.** The Schwartz-Zippel degrees of every check the verifier makes: the alpha
  batch, the grid identity (`2h1 + 2h2`), the small grid (`3h2`), the boundary and restriction
  lines, the Horner chain-end identities (challenge degree through the recurrence), the wiring
  fingerprints and the grand product (`slots * h2 + public factors`, and `2F` for zero factors).
  Each is one bad-identity event at its own challenge barrier, times `L - 1` for the list in the
  executable (the union argument says `L`; the difference is far below the display precision).

The verifier rejects every byte it reads as a field coordinate (opened rows, openings, sumcheck
messages, the clear vector, derived public data) unless it is canonical, before any arithmetic.
Roots, digests and length words stay opaque, and the bounds the optimized field arithmetic
assumes are covered by tests, not audited.

## What is assumed

The bounds are conditional on these obligations, none of which is closed:

- **P1.** The packed level-1 alphabet (one `F4` symbol per column per row, `E`-valued columns
  opened as one value with weights `beta * b_t`) admits the scalar Reed-Solomon guarantee in the
  block metric. The descent and joint-list extraction are own proofs; the grouped-weight bridge
  for `E` columns is not written.
- **P2.** The two-grid quotient identities imply every intended row and chain-end constraint
  under the adaptive choice of `Z` after stage 1 and of `Q` after `alpha`.
- **P3.** The workload traces (SHA-256, Keccak, Poseidon, mulmod, ECDSA, RSA, the passport
  groups) are faithful: a false semantic claim gives a nonzero polynomial at the barrier where its
  challenges are sampled. Degree counting alone does not establish this.
- **P4.** The tail is the analyzed scalar tensor-fold protocol, including the four-coordinate
  first batch and the last clear check.
- **P5.** The Blake3 Fiat-Shamir transcript, the Merkle binding and the grinding preserve the
  interactive bound; the adversary model and the losses are not stated. The grinding search
  probes 2^32 nonces per state and can exhaust them, so termination is not guaranteed; a bounded
  search with a reseed is open. No quantum claim.

Own proofs, not theorems in the cited papers: the reduction from lines to affine spaces with no
dimension factor (BCIKS20 section 6.3 with BCHKS25 Theorem 1.5 as the line input), and the
list-regime argument (descent, Bind, backward extraction through the tail). The list caps are
checked by certificates in the separate `caracal7-fv` repository.

## Sources

- Dao, Kominers, Thaler, *Reed-Solomon Codes Beyond Johnson* (DKT26): Theorem 5.12 (p. 53) and
  Lemma 5.3 (pp. 40-43) for the level numerators; Corollaries 7.2, 7.7 and 7.8 for the affine,
  interleaved and fold transfers. Source: [quangvdao/rs-beyond-johnson](https://github.com/quangvdao/rs-beyond-johnson).
- Ben-Sasson, Carmon, Haböck, Kopparty, Saraf (BCHKS25): Theorem 1.5 (p. 9), the Johnson-radius
  line bound the query count is sized by.
- Haböck, [ePrint 2025/2110](https://eprint.iacr.org/2025/2110), Theorem 2 (p. 4): the
  conservative mutual correlated agreement numerator.
- Jo, [ePrint 2026/891](https://eprint.iacr.org/2026/891), Corollary 4.6 and Theorem 4.7: line
  MCA is preserved under interleaving and by tensor factors.
- Ben-Sasson, Carmon, Ishai, Kopparty, Saraf (BCIKS20), section 6.3: lines to affine spaces.
- Diamond and Gruen, [ePrint 2024/1351](https://eprint.iacr.org/2024/1351), Theorem 3.6:
  interleaved proximity.

## Run

```sh
uv run mojo run --Werror -I src bench/bench_soundness.mojo
```

CPU only; compiles every benchmark statement, prints its parameters, query counts, each field
numerator by name, and the two bit counts per case. It runs inside `./run_tests.sh`. Recompute
after any change to the field, a grid, a rate, a query count or a workload.
