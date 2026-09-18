# The matmul relation: design

Status: withdrawn, kept as the record of a dead end. Both reviews (Opus, Codex) found that the relation
below proves `C = A B mod 127` only: the field has characteristic 127, so one int8 product already wraps,
and a base-127 carry family is identically zero (`127 = 0`). Integer accumulation in this field needs
every partial sum in small digits with range checks, O(T k n) cells. Summary and consequences:
`docs/zkml.md`, "The matmul relation: why there is none". The reviews also found: the Frobenius pairing
couples inner and outer digits, so the contraction must be defined on stored slots with a direct tensor
functional (a `Unit`), not on coefficient indices; the multilinear points multiply the verifier's unit
count by the odd-digit multiplicity (about 1,800 units per point at the milestone grid); the round
kernels fold three digits per level, so `D` must be a multiple of 3; padding must be a checked
constraint; the point bound is `2 (h1 + h2) / |E|` in all, plus a column coordinate when a matrix spans
columns; the matvec milestone cannot go below two cells per MAC because of B's two limbs. The text
below is unchanged. Spec references are to `wiki/projects/caracal7/specs/caracal-prover.md` section 9.

## Claim

For each matmul `m` in a statement, integer matrices `A_m` (T x k), `B_m` (k x n), `C_m` (T x n) with
`C_m = A_m B_m` over the integers. The prover computes `C_m` natively and commits all three as base-127
limb columns in the W tree. No product is committed.

The relation proves, for a batching challenge `lambda` and points `r` (over T), `s` (over n):

    sum_m lambda^m sum_kappa A_m(r, kappa) B_m(kappa, s) = sum_m lambda^m C_m(r, s)          (1)

where `A_m(r, kappa)` is the contraction of `A_m` against the point functional over its T digits with
the inner index `kappa` free, and likewise for `B_m` and `C_m`. The sum over `kappa` is a sumcheck over
the inner digits; it ends in openings at multilinear points. Everything else is the existing protocol.

## Layout in the grid

A committed column has `N = h1 h2` slots, `a1 + a2` binary digits and one odd digit of size `m1 m2`
(spec 9.1). The binary digits are monomial digits of the column polynomial; the odd digit is a value
digit on `mu_1 x mu_2`.

Each matrix occupies a set of columns. Its index digits are partitioned per matrix into:

- **inner digits**: binary slot digits only. The odd digit is never an inner digit (it does not fold).
  For `A_m` and `B_m` the inner index `kappa` runs over the same number of inner digits `d_m`; both are
  zero-padded to `2^d_m`.
- **outer digits**: the remaining slot digits (binary and odd) plus the column index. `T` for `A_m`,
  `n` for `B_m`, both for `C_m`.

When `kappa` needs more digits than one column has binary digits, the inner index also runs over
columns: the sum over `kappa` is then a sum over the per-column claims, which the verifier adds after
the rounds (the round messages are for the slot digits; the column part is verifier arithmetic). For
the model weights this puts `k / 2^(a1 + a2)` columns per matrix, which is what proof size pays.

Limbs: each of `A_m`, `B_m` is a separate column set per limb; `C_m` has one column set per limb pair
`(a, b)` holding `A_(m,a) B_(m,b)` without carries. One degree-1 family per output entry folds the
limb pairs into the canonical limbs of `C_m` with carries (the carry columns are ordinary witness
columns; the family is a quotient-checked local relation). The relation (1) runs once per limb pair,
batched by `lambda` like everything else.

## Protocol

Transcript order (spec 9.4 with two insertions, marked `+`):

```
 commit  tree W (witness incl. A, B, C limbs)   -> stage-1 challenges; + lambda, r, s
 +       matmul rounds: for d = 0 .. D - 1, send s_d (degree 2, 3 elements) -> rho_d
 commit  tree Z; Z2                              -> alpha
 commit  tree Q; Q3                              -> z
 send    openings at P points; + the matmul points (r, rho_bar), (rho_bar, s), (r, s)
                                                 -> beta, gamma
 tail as today
```

`r` and `s` are points in the sense of 9.1 on the outer digits: a monomial point on the binary
digits, the Lagrange vector on the odd digit. `rho_bar = (x) (1 - rho_d, rho_d)` is a multilinear
point on the inner digits. `D` is the largest `d_m` in the statement; a matmul with fewer inner digits
is padded with zeros so that the rounds are shared.

**Prover.** After the stage-1 challenges the prover builds, per matmul and limb pair, the two vectors
of length `2^D` over E:

    a_m(kappa) = A_m(r, kappa) = <w_r, A_m(., kappa)>,     b_m(kappa) = B_m(kappa, s) = <w_s, B_m(kappa, .)>

by contracting the stored columns against the point functionals over the outer digits (one pass over
the matrix; the same slot weights `open` uses, with the inner digits left free). The product does not
distribute over the batch, so the pairs stay separate and the rounds run on the batched product:

    s_d(X) = sum_m lambda^m sum_(kappa above d) a_m(rho_0 .. rho_(d-1), X, kappa) b_m(rho_0 .. rho_(d-1), X, kappa)

which is what `k_round_partial` computes for one pair (`w_tilde := a_m`, `y := b_m`, three digits per
level), followed by `k_fold8` on both vectors at `rho`. The matmul rounds reuse those two kernels and
`k_round_sum`; they run on prover-held vectors and commit nothing. Per level the prover does `M` pairs
and adds the partials with `lambda^m`. Cost: `O(sum_m 2^d_m)` E-work per round, negligible next to the
encoder.

**Verifier.** Reads the `3 D` round elements, checks `s_d(0) + s_d(1) = s_(d-1)(rho_(d-1))` with
`s_(-1)(.) := sum_m lambda^m C_m(r, s)`, samples `rho_d`, and at the end checks

    s_(D-1)(rho_(D-1)) = sum_m lambda^m A_m(r, rho_bar) B_m(rho_bar, s)

with both sides taken from the opening list once it arrives. Both are deferred equalities; the round
checks themselves are the same loop as `_Tail.level` (`quadratic_at`, `list_e`).

## Multilinear points in the opening machinery

This is the one new piece of machinery. Today an opening point is `z g^dj` or a fixed coordinate
(`relations/ir.mojo`, POINT = (dj1, dj2)), and its functional `w_z` is built by `slot_weight`: a
monomial product over the binary digits times the Lagrange vector on the odd digit, with the
Frobenius-real pairing folded in as `Mon(x) + Par(x, r)` where `Par(x, r) = z^xbar r^s(x)` is the
weight of the conjugate partner `xbar = (2^a - x) mod 2^a`. That is cheap because `z^(2^a - x)` is
again a product over the digits of `x`.

The matmul points `(r, rho_bar)` and `(rho_bar, s)` have multilinear factors `(1 - rho_d, rho_d)` on the
inner digits. Their partner weight `eq(xbar, rho)` is not a product over the digits of `x`, because
`2^a - x` carries. It is a short sum of products: with `p` the lowest set bit of `x`, the bits of
`xbar` are 0 below `p`, 1 at `p`, and the complement of `x` above `p`, so

    eq(xbar, rho) = [x = 0] prod_d (1 - rho_d) + sum_p [x_d = 0 for d < p] [x_p = 1] prod_(d<p) (1 - rho_d) rho_p prod_(d>p) eq(1 - x_d, rho_d)

one term per `p` plus the zero term: at most `a + 1` tensor terms per axis, each a product of per-digit
pairs and deltas. `pcs/tensor.mojo` already expresses exactly that (`Unit.pair`, `Unit.delta`): a
multilinear point becomes a list of at most `(a1 + 1)(a2 + 1)` units instead of the twelve of a monomial
point. The verifier folds units per level as it does now, so its cost per matmul point is a few dozen
units.

Prover side, `slot_weight` is monomial-specific. The prover materializes `w` for a matmul point from
its unit list with a generic units-to-vector kernel, the same job `k_materialize_tail` does for the
tail's batch (`ptab` power tables, one pass over N). Two new points per proof, so two extra passes.

Descriptor: a new POINT kind whose coordinates name a sampled challenge (`r`, `s`) or the round vector
`rho_bar` instead of a shift of `z`; `Shape.point_list` carries it, `required_points` demands it when
the statement has a matmul, and `prefix_bytes` binds it like every descriptor.

## Soundness (ledger terms)

All terms are over `|E| = 127^20`, union-bounded, and go in `docs/soundness.md` next to the relation
numerators. Conditional on the openings being the evaluations of the committed columns (the existing
commitment-layer bound):

- batching over matmuls and limb pairs: `(M - 1) / |E|` for `lambda` (degree `M - 1` in `lambda`);
- the points `r`, `s`: (1) is a polynomial identity in the coordinates of `r` and `s` of degree below
  `h1` and `h2` per coordinate after the binding of `lambda`; if `C_m != A_m B_m` for some `m` it is a
  nonzero polynomial, so `(h1 + h2) / |E|` per axis coordinate, four coordinates;
- the rounds: `2 / |E|` per round, `2 D` in all;
- the multilinear points as opening points: no new term, the openings are bound by the tail as today.

Integer faithfulness is not a field term: it follows from the limb-pair columns being exact products
below 127^2 (no reduction) and the carry family being a quotient-checked local relation.

## Costs

Per proof: `3 D e` bytes of round messages, two extra opening points (`2 columns e` bytes of
openings plus their tail units), and the columns of A, B, C. Per matmul the committed cells are the
matrices themselves, `T k + k n + T n` limb cells, plus the carry columns of `C`.

Proof size scales with committed cells divided by `N`, at about 600 to 800 bytes per column (level-1
query rows and openings). This is the weak point against a GKR prover and is addressed by segments
and aggregation, not here.

## Milestone: int8 matvec 256 x 256

- Grid: `h1 = 144` (`a1 = 4, m1 = 9`), `h2 = 128` (`a2 = 7, m2 = 1`). Inner index over the 7 binary
  digits of axis 2 and 1 binary digit of axis 1; the other 3 binary digits and the odd digit of axis 1
  are outer.
- `A`: 1 x 256, int8, 2 limbs: two columns. `B`: 256 x 256, int8, 2 limbs: `n = 256` over the outer
  digits (72 values per column) and columns: 4 columns per limb, 8 in all. `C`: four limb-pair column
  sets plus carry columns and the fold family.
- Measure: prove time, committed cells per MAC (expected below 1), proof size, verify time, on the M1.
- Test: a wrong product in one entry of `C` must fail the rounds; a wrong carry must fail the fold family.

## Open items

1. The inner index over columns: the verifier sums per-column final claims. Confirm that batching those
   under one `lambda` power per matmul keeps the `lambda` term as stated.
2. `T` larger than the outer digits of one column (long sequences): `A_m` and `C_m` then span columns
   over `T`, and the point `r` extends over the column index as another digit group. Same mechanism as
   the inner index over columns; write it out when the block workload needs it.
3. Where the matmul stage sits in the prover layout (`ST_*` groups, bump allocation of the `a_m`, `b_m`
   vectors: `2 M 2^D e` bytes).
4. Whether `r`, `s` should be sampled with the stage-1 challenges (as above) or after Z. Above is
   earliest; nothing in the rounds needs Z or Q.
