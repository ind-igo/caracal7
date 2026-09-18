# Soundness ledger for the CSP configuration

Status: **conditional analysis, not certification**. Reviewed against prover commit `8f94cc7` on
2026-09-12; parameters updated 2026-09-13 (`E = F_(127^20)`, query target 112) and 2026-09-14 (Johnson
regime, tail rate 1/8, decisions.md). The
executable ledger compiles the current source; the conclusions below require the proof obligations
and implementation gaps in this note to be closed. A successful calculation or test run does not
establish a cryptographic security level.

The benchmark metadata says `security_bits: 112`. In `core/params.mojo`, 112 is the per-level
**query target**, met as 92 bits of queries at the Johnson radius `1 - sqrt(rate) - 1/16` (BCHKS25
Theorem 1.5 below) plus 20 bits of grinding on every query seed, with
`E = F_(127^20)` (20 coordinates; `E16` in `core/field.mojo` rebuilds the old
16-coordinate tower for measurement). It is not the total soundness budget. Do not use the metadata
as a verified claim. The upstream
[eligibility rule](https://github.com/ethereum/csp-benchmarks/blob/main/CONTRIBUTING.md#benchmark-eligibility)
requires at least 96 bits; the project's own goal is more than 100 bits.

## Run and interpret

```sh
uv run mojo run --Werror -I src bench/bench_soundness.mojo
```

The program runs its arithmetic self-checks, compiles all 16 benchmark cases, and prints their
actual `Params`, `Shape`, tail levels, query error, and named field-error numerators. It neither
constructs a GPU context nor generates a witness or proof. SHA-256 and Keccak input bytes,
Poseidon elements, and ECDSA signature values do not affect the compiled statement at a fixed grid.
The ECDSA report calls the workload's `statement()` and its fixed `Ecdsa.circuit()` without a live walk.

The case whitelist mirrors `cli/ffi.mojo` and `cli/main.mojo`. Those routes must be compared again
when adding sizes or changing routing. All dimensions, descriptors, and counts after routing come
from production compilation; there is no copied table of shape counts or reimplementation of the
domain/tail planner. The ledger rejects Herder lookup/permutation accumulators (the four benchmark
workloads have none), other tail arities, and challenge-dependent ordinary families that would need
another argument. This is scoped to these workloads, not a generic IR security checker.

Output:

- `field_numerator NAME A` means the candidate error contribution `A / 127^e`.
- `query_error_per_attempt` is the sum over **all committed levels**, including level 1, of the miss
  probability of one transcript attempt; `query_error` divides it by `2^grind_bits`, the hashes one
  attempt costs the prover (see Grinding below).
- `capacity_conjecture ...` and `johnson_bchks25_1.5_projection ...` are projections of the other two regimes at the
  same geometry: the queries each level would need at the same per-level target, the query bits at those
  queries, the `pcs_gap` numerator of that regime, and the `conditional_iop_bits` the switch would compile.
  `johnson_bchks25_4.2_4.6_projection ...` is the Johnson regime charged as BCHKS25 section 4 states it (J1, J2
  below): the doubled `m` and the factor `M = words - 1` per code. `Profile.regime` selects the regime;
  `CLIENT` compiles `REGIME_JOHNSON` since 2026-09-14.
  - `REGIME_CAPACITY` (radius `1 - rate - eta`, miss `rate + eta`) is the unproven up-to-capacity
    conjecture. It has no proven field term; the projection keeps the unique `pcs_gap` as a placeholder,
    so its `conditional_iop_bits` is the query term only. Nothing in this note certifies it.
  - **Historical pairs charge (superseded by J2 below):** `REGIME_JOHNSON` (radius `1 - sqrt(rate) - eta`, miss `sqrt(rate) + eta`) charges the correlated
    agreement error of BCHKS25 Theorem 1.5 ([[raw/papers/proximity-gaps-rs-codes]], any domain, any field):
    `a / |E|` per code with `m = max(ceil(sqrt(rho) / (2 eta)), 3)` and
    `a = (2 (m + 1/2)^5 + 3 (m + 1/2) gamma rho) / (3 rho^1.5) * n + (m + 1/2) / sqrt(rho)`, linear in `n`
    where BCIKS20 1.2 had `n^2`. The theorem is stated for pairs (lines); the ledger applies it in the
    same places as the unique term (level 1 uniform fold, three tensor elements per tail level). The
    affine-space (batched columns) form (J1), the mutual form (Haböck 2025, cited there; J2), and the
    list-regime rewrite of the level 1 close case (spec 12.2; J3) are open obligations before this regime
    is proven for caracal7; see "Open obligations of the Johnson regime (J1-J3)" below. The constant grows
    as `rho^-1.5`, so the low-rate tail levels dominate the term.
- `conditional_iop_bits` is `-log2(query_error + sum(A)/127^e)` at the compiled `e`, rounded down to
  two decimals.
- `projected_e16_...` and `projected_e20_...` change only the denominator to `127^16` or `127^20` at
  the same geometry and queries. They are not performance estimates or security certifications.

Calculations use Float64 for diagnostic sizing, not interval arithmetic or a machine-checked proof.
Exit zero means the calculation succeeded. The program always reports unresolved obligations and
never emits `security_bits` or a certification pass. Fiat-Shamir, hash binding, quantum security,
and the gaps below are not assigned zero error; they are **outside this conditional total**.

### Current results

After the J2 numerator correction, ECDSA has **88.52 conditional IOP bits**
(before J3's Bind term). This is 19.13 bits below the former 107.65 projection.
Its `pcs_gap` numerator is 2,676,623,914,570,924, instead of 4,096,994,528.
The old section-4 curve projection is 95.99 bits. All these totals still assume
J1/J3 and P1-P5 as specified below. The table below records the earlier charges;
J3 will replace the current column after its list terms are added.

These are diagnostic bounds under the assumptions below, not measured attack costs or certified
security levels. Poseidon sizes sharing a grid have identical ledgers.

| Workload | Input | Grid | Historical pairs projection, e = 20, Johnson, tail rate 1/8 (2026-09-14) | unique, tail rate 1/32 (2026-09-13) | e = 16, lambda' 103 (before 2026-09-13) |
|---|---:|---:|---:|---:|---:|
| SHA-256 | 128 B | 32 x 224 | 109.93 | 110.46 | 95.00 |
| SHA-256 | 256 B | 32 x 336 | 109.68 | 110.70 | 94.51 |
| SHA-256 | 512 B | 32 x 672 | 109.27 | 110.46 | 93.44 |
| SHA-256 | 1024 B | 32 x 1152 | 108.51 | 110.26 | 92.64 |
| SHA-256 | 2048 B | 32 x 2688 | 107.65 | 110.21 | 90.71 |
| Keccak | 128 B | 64 x 24 | 110.19 | 110.70 | 97.08 |
| Keccak | 256 B | 64 x 48 | 110.11 | 110.54 | 96.11 |
| Keccak | 512 B | 64 x 96 | 109.92 | 110.54 | 95.17 |
| Keccak | 1024 B | 64 x 192 | 109.33 | 110.26 | 94.20 |
| Keccak | 2048 B | 64 x 384 | 108.86 | 110.15 | 93.20 |
| Poseidon | 2, 4, 8 elements | 64 x 384 | 108.86 | 110.15 | 93.20 |
| Poseidon | 12, 16 elements | 64 x 896 | 107.77 | 110.11 | 91.66 |
| ECDSA | 1 signature | 144 x 576 | 107.65 | 110.58 | 90.69 |

At `e = 16` only two of the sixteen cases reached 96 bits: the dominant contribution was the PCS
proximity gap, particularly the first committed tail's large code domain. For ECDSA,
`A_pcs_gap = 161280 + 3*(645120 + 43008 + 5376) = 2241792`; the full numerator is 2282533, which
over `127^20` is about 2^-118.7. Increasing query counts cannot reduce that field term; the field
did. At `e = 20` the query term binds (101 bits at the old target of 103 per level, since four levels
sum), so the per-level target is 112, met as 92 bits of queries and 20 of grinding, and the total sits
at 110 to 111 bits. The remaining proof,
arithmetic, and Fiat-Shamir obligations are unchanged; the table does not by itself justify submitting.

## Claim and composition

The desired claim is that an adversarial prover cannot make the verifier accept a false workload
statement, except with bounded probability. The adversary may write every witness and proof byte;
it need not run the honest trace generator, respect its advice, or call its validation routines.
The statement and public data must be those reconstructed by `verify_workload`.

First analyze the interactive protocol with independent uniform verifier challenges and binding
oracles. Each bound must hold conditional on the preceding transcript, including adaptive Z and Q
commitments. Sum the bad-event probabilities; do not sum bit counts, take only the largest term,
or charge only the first tail level. Ligerito's composition follows this pattern in
[section 6.3](https://angeris.github.io/papers/ligerito.pdf#page=14).

Then establish the noninteractive claim for the implemented Blake3 Fiat-Shamir transcript, stating
the random-oracle model, adversary query/work budget, and reduction losses. The IOP total is not
automatically a 96-bit computational-security claim. `is_pq: true` also does not establish the same
bit count against a quantum adversary. Zero knowledge is not claimed (`is_zk: false`).

## Commitment-layer bound

Let `Q = |E|`, `L_i` be the code length, `k_i` the message dimension, `s_i` the query count, and
`r_i = k_i/L_i`. Level 1 has `k_1 = N/4`; subsequent levels use `TailLevel.rows`.

```
epsilon_queries = sum_i ((1 + r_i) / 2)^s_i
A_pcs_gap       = L_1 + 3 sum_{i >= 2} L_i
A_sumcheck      = 6 * number_of_committed_tail_levels
A_opening_batch = 2
```

Queries are sampled independently with replacement by `transcript.sample`; duplicate rows are
deduplicated only for Merkle authentication. Therefore the original draw count is the exponent.
The miss probability follows from RS distance `d = L-k+1` at radius `t = floor((d-1)/2)`:
at least `t+1` positions fail in the relevant far/incorrect-fold case, and
`1-(t+1)/L <= (1+r)/2`. This also avoids the inconsistent proximity factors printed in some
equations of the May 2025 Ligerito paper; it agrees with its section 6.4 distance-based calculation.

**Scalar citation checked.** Ben-Sasson, Carmon, Ishai, Kopparty, and Saraf's
[July 7, 2020 paper](https://sites.math.rutgers.edu/~sk1233/GS-FRI.pdf#page=3), Theorem 1.2,
gives error `L/Q` in the unique-decoding regime. Theorem **1.7** in this version supplies
correlated agreement for affine spaces with the same error, independent of the dimension of the
space. Theorem 1.6 in this version is about parameterized curves. Pin the version with the theorem
number instead of treating the old note's number as universal.

Uniform coefficients in `E^columns` induce a uniform element of their span even when columns are
dependent. The scalar theorem therefore needs no column-count multiplier. The separate transfer
to Caracal's four-coordinate `E tensor_F F4` alphabet is the lemma in vault spec section 12.1:
split into four conjugate RS codes over E, descend a unique nearby word to F4, and use the block
metric to check all coordinates at the same position. **This transfer and its composition with
opening claims remain review obligations**, not consequences of finding the citation.

For the tail, three tensor challenges give `3 L_i/Q`, conditional on the interleaved/tensor
proximity result; see Diamond and Gruen,
[Proximity Gaps in Interleaved Codes](https://eprint.iacr.org/2024/1351), Theorems 3.1 and 3.6,
Corollary 3.7, and the Ligerito construction. The last committed code is included even though its
folded message is sent in the clear. There is no extra code or sumcheck after that clear message.

**Historical configuration decision (2026-09-14; pairs projection).** `CLIENT` compiles `REGIME_JOHNSON` with `eta = 1/16` and the
tail rate rule `tail_rate_inv = 8` in place of spec 9.5's 1/32. The ledger's ECDSA sweep over the tail
rate (all at 74 level-1 queries) is why: at 1/32 the tail levels sit near rate 1/60, where the
`rho^-1.5` constant of Theorem 1.5 is large, and the field term caps the case at 101.45 bits with
74/39/45/45 queries; 1/16 gives 105.24 with 74/51/55/55; 1/8 gives 107.65 with 74/70/72/72; 1/4 gives
108.04 with 74/108/108/108. The 1/8 rule keeps the bits within three of the unique ledger with 288
queries against 401, and its tail domains are seven times smaller (level 2: 92,160 against 645,120),
which the prover's tail encode and the verifier's openings both see. ECDSA measures 545,892 B of proof,
warm prove 369 ms, verify 116 ms (`bench_ecdsa`, from about 705 KB / 400 ms / 135 ms). The ledger runs
107.65 to 110.19 bits across the cases. Those figures used the pairs form of BCHKS25 Theorem 1.5, p. 9.
The current J2 charge and J1-J3 status below replace that analysis.
A per-level `eta` (the tail at rate 1/8 could take `eta = 1/24` with `m` still 3) is a small further
lever not taken.

### Open obligations of the Johnson regime (J1-J3)

These are the three items the paragraph above names. The J2 update below supersedes the historical charge and status above. Each records what the ledger charges today, what
the cited source actually proves, what is missing, and what evidence would close it. None is a code
defect; all three are analysis gaps that the `CLIENT` profile now depends on.

**J1. Affine-space (batched columns) form: reduction checked.** BCHKS25
**Theorem 1.5, p. 9**, states the line case. BCIKS20 **section 6.3,
printed pp. 28-29 (PDF pp. 29-30)** gives a reduction from lines to affine
spaces. The detailed independent check below confirms that Theorem 1.5 can
replace its line theorem with no dimension factor. This is **own proof,
needs review**, not a theorem stated verbatim in either paper.

BCHKS25 **Theorem 4.2, p. 27**, instead uses a power curve, with a factor
`M` and the doubled `m`. **Theorem 4.5, p. 28**, constrains the decoded
codewords to a subspace of the code. Neither is the required statement
about a uniform affine space of input words. The J1 reduction supplies
ordinary correlated agreement. J3 needs the stronger same-set property
and must pay for it separately; J1 alone does not justify the old total.

**J2. Mutual correlated agreement: public source and transfer.** The public source is
Haböck, *A note on mutual correlated agreement for Reed-Solomon codes*,
[ePrint 2025/2110](https://eprint.iacr.org/2025/2110), version of 2025-11-17.
The author calls it a proof outline. **Theorem 2, p. 4** applies to
`RS[F_q, D, k]` of dimension `k + 1`, on any set of distinct points `D ⊆ F_q`.
There is no subgroup, characteristic, or random-domain condition. Set `n = |D|`,
`rho = k/n`, `m >= 3` an integer, and `gamma_m = 1 - (1 + 1/(2m)) sqrt(rho)`.
For all but at most

```
a_H = (m + 1/2)^7 n^2 / (3 rho^(3/2))
```

scalars `z`, **every** set `A` of at least `(1-gamma_m)n` positions on which
`f0 + z f1` restricts to a codeword also makes both inputs restrict to codewords.
This is the same-set form, not only existence of one common agreement set.
For a target `gamma = 1 - sqrt(K/n) - eta`, where `K` is the code dimension,
use `rho = (K-1)/n`, `eta_H = 1 - sqrt(rho) - gamma`, and
`m = max(ceil(sqrt(rho)/(2 eta_H)), 3)`. Then `gamma <= gamma_m` and the theorem
bounds the target event. Hab25 does **not** use the doubled `m`. Its numerator
is quadratic in `n`. The note says that multilinear and power combinations can
be treated in the same way (p. 2, unnumbered remark); an explicit bound for them
is **not stated** there.

BCHKS25 **Theorem 4.6, pp. 28-29**, states the stronger linear-in-`n` bound
`M a_B`, with

```
m_B = max(ceil(sqrt(rho)/(1-sqrt(rho)-gamma)), 3)
a_B = (2(m_B+1/2)^5 + 3(m_B+1/2)gamma rho)n/(3 rho^(3/2))
      + (m_B+1/2)/sqrt(rho).
```

Here `rho=(K-1)/n`; the historical projection instead retains `K/n`.
It is a theorem for the power curve `sum_{j=0}^M z^j u_j`; its proof is a sketch
that uses the factor argument from Hab25. Hab25's public proof does not state
this improved numerator. At nominal rate 1/4 and `eta=1/16`, Hab25 has `m=4`
and Theorem 4.6 has `m_B=8`; at rate 1/8 they have 3 and 6. The ledger now uses
`a_H`, with the degree/dimension distinction above. This is a conservative choice
of a public bound, not a refutation of Theorem 4.6. It retains the old pairs
charge and the old `section4=True` charge as labelled projections. The latter
still includes `M = columns-1` at level 1, as a power-curve comparison only.
The protocol still samples independent column coefficients. Neither projection
is a proof for that sampler. Historical projections retain their old rate
normalization. The main Hab25 calculation pads Float64 upward by `1e-12` before
rounding; this is diagnostic sizing, not interval certification.

**Transfer source found.** Diamond and Gruen **Theorem 3.1, p. 6**, and
**Corollary 3.7, p. 9**, restrict the radius to unique decoding. The proof uses
uniqueness in **Lemma 3.2, p. 7** to identify every nearby word with one fixed
word. That step fails for lists. **Theorem 3.6, p. 9**, itself has no unique-radius
restriction: it assumes the required line gap for every interleaving. The former
note incorrectly put the restriction on both theorems.

Jo's supplied [ePrint 2026/1432](https://eprint.iacr.org/2026/1432), p. 5,
points to a separate public result: *Interleaving Stability for Mutual Correlated
Agreement and Curve Decodability*, [ePrint 2026/891](https://eprint.iacr.org/2026/891).
Its **Theorem 4.4 and Corollaries 4.5-4.6, p. 9**, give exact preservation of
line MCA under any row interleaving, at every radius. Its **Theorem 4.7,
pp. 10-11**, bounds polynomial-generator MCA by the sum of the univariate
factor errors. Jo quotes the tensor closure from BCGM25 Lemma 4.4 and
the linear-change lemma 4.1 on p. 10; this citation uses Jo's stated theorem.
Thus the three binary tensor factors of an RS code cost
`3 a_H / |E|`, with no interleaving-width factor. The factors `(1-r,r)` and
`(1,r)` differ by an invertible linear change of the two input words. One can
also use `3 a_B / |E|` with BCHKS25 Theorem 4.6 as the scalar input. These are
statements about the same RS code in every row. The four conjugate domains of
`E tensor_F F4` and the protocol's running claim still need the J3/P4 argument.
The level-1 scalar charge is also still conditional: J1 proves ordinary CA,
not affine MCA. Jo Theorem 4.7 alone would charge one line numerator per
column variable. A dimension-free same-set reduction is needed in J3.

**Newer source check.** An arbitrary-dimension affine MCA theorem with the pairs
numerator is **not stated** in the four supplied newer papers. The checked
statements are:

- Jo 2026/1432, **Theorem 2.3, p. 7**, imports BCHKS25 Theorem 4.6 for lines.
  **Theorem 3.4, p. 9**, extends a line bound a fixed number of integer steps
  beyond Johnson; it does not state the requested affine-space form.
- Chojecki 2026/1463, **Theorem 1.3, pp. 5-6**, imports the same line numerator.
  **Theorem 1.1, p. 5**, gives a shortening exponent beyond Johnson; it does not
  state the requested affine-space or tensor form.
- Chojecki 2026/1479, **Proposition 2.1, p. 5**, concerns line thresholds.
  Section 2 on that page explicitly says no theorem improves the companion
  paper's safe threshold. The requested form is **not stated**.
- Arnon, Boneh, and Fenzi 2026/680 define affine spaces among the possible
  samplers (pp. 3-4) but focus on lines. Theorem 4.12, p. 19, restates a line bound, with a half-size `m` and a
  different rate convention from BCHKS25 Theorem 4.6. We do not substitute
  that restatement for the source formula. A dimension-free affine bound is
  **not stated** there. The exact later transfer
  is Jo 2026/891, **Corollary 4.6, p. 9**, as above.

**Verdict.** The missing-public-source issue and the scalar/interleaved/tensor
MCA transfer are closed by citations. This does not close P4's adaptive protocol
composition or J3's packed-alphabet and opening argument.

**J3. List-regime rewrite of the level-1 close case (spec 12.2).** `miss_probability` returns
`sqrt(rate) + eta` per query in `REGIME_JOHNSON`, and the ledger raises it to each level's query
count. Spec 12.1 derives the unique regime's `(1 + rate)/2` from a case split that uses uniqueness: at
radius `t = floor((d-1)/2)` there is at most one `X̃` with block distance `Δ(X, G X̃) <= t`, the close
case compares `G y` against that one `X̃ β`, and `1 - (t+1)/L <= (1 + rate)/2` bounds both branches. At
`gamma = 1 - sqrt(rho) - eta` the close case yields a list, of size at most about `1 / (2 eta sqrt(rho))` by
the Johnson list bound (16 at level 1; this is not the `m` of Theorem 1.5, which is 4 there), and the split does not hold as written.

Spec 12.2 sketches what replaces it, and it is a sketch. The extractor picks one list element instead
of the unique `X̃`, and mutual correlated agreement (J2) is what decomposes a list element of the
folded word column by column into list elements of the columns. The choice is then pinned by
out-of-domain binding: the openings `alpha_{c,z}` at the `z` sampled after the `Q` commitment separate
two distinct columns of bidegree below `(h1, h2)` except with probability `(h1 + h2)/|E|`, for a Bind
allowance of about `columns * (list size)^2 * (h1 + h2) / (2 |E|)`. That allowance is quadratic in the
list size and **is not a ledger term today** - the ledger carries no list-size-dependent numerator at
all. The checks that must bind the chosen element, rather than "the" decoded word, are
`verifier._residual`, `_small_grid`, `_boundaries` and `_restrictions` (the P2 row) together with the
tail's running claim at `r̄_l`.

The query error must then be re-derived, not assumed. If the rewrite works, the close case becomes a
field-size event and the only surviving per-query event is the far case at miss `1 - gamma =
sqrt(rho) + eta`, which is what `miss_probability` already returns; the ledger therefore charges the
rewrite's conclusion in advance of the rewrite. The list size enters the field terms and not the
exponent: the query term stays `(sqrt(rho) + eta)^{|S|}` with no list factor, and a union bound over
the list in the exponent would be the wrong shape. Note also that `rho = k/n` is off by `1/n` from the
rate the code reports, which does not move these bit counts at these lengths.

Finally, the packed four-coordinate descent survives only with a new proof. Step (b) of 12.1 concludes
that the nearby interleaved codeword is `F4`-rational because Galois conjugation fixes it *by
uniqueness*, which is the step a list breaks; 12.2 argues instead that the agreement set has more than
`k` positions, so the close codeword is still `F4`-rational. That is a different argument and has not
had the adversarial review 12.1 had. Step (c)'s `n_cw > 1` split, which uses "any codeword within `t`
of it equals `W_q β` since `2t < d`", has no list-regime version at all. Closing J3 means a rewritten
12.1 over the `E ⊗ F4` alphabet including that split, a stated Bind lemma with its numerator added to
the ledger as a named field term, and a derivation that the far case is the only query event left.

**What section 4 as written would compile (2026-09-14).** The `johnson_bchks25_4.2_4.6` projection
charges Theorems 4.2 and 4.6 exactly as stated, in place of the pairs form: `M * a` at the doubled `m`,
with `M = columns - 1` at level 1 (555 for ECDSA; `_level1_symbol` batches the columns only, the
`gamma` weights are opening claims, not RS words) and `M = 1` for each pair-folding tail challenge.
This is a number, not a proof: it does not turn the uniform `beta` into a curve, and it supplies none
of the transfer, Bind, or query arguments of J1 to J3. ECDSA compiles 95.99 bits against 107.65; the
CSP hash cases 97.9 to 103.7 against 107.65 to 110.19. The level-1 term dominates so completely that
the tail rate no longer matters. Of the 11.7 ECDSA bits, 4.28 come from the doubled `m` (107.65 to
103.37) and 7.4 from the factor `M`. The levers if this were the final charge: a larger `eta` (smaller
`m`, more queries), a larger `e`, or the unique regime at 401 queries. A smaller `eta` only grows `m`.

**J1, the reduction: confirmed. Own proof, needs review.** The independent read
used BCIKS20 **Theorem 1.4, printed p. 3 (PDF p. 4)**, **Theorem 1.7,
printed p. 4 (PDF p. 5)**, and **section 6.3, printed pp. 28-29 (PDF pp. 29-30)**.
The replacement line theorem is BCHKS25 **Theorem 1.5, p. 9**. This substitution
is our proof. Neither paper states the resulting combined theorem.

Let `C = RS_E(D, K-1)`, `rho = (K-1)/n > 0`, and
`gamma < 1-sqrt(rho)`. Write `a(gamma)` for the numerator in BCHKS25
Theorem 1.5, with `m=max(ceil(sqrt(rho)/(2(1-sqrt(rho)-gamma))),3)`.
Assume `a(gamma) < Q` and that the list bound at `gamma` is less than `Q`.
If a uniform point of an affine space `U` is gamma-close to `C` with
probability greater than `a(gamma)/Q`, then every generator of `U` agrees
with a codeword on one common set of at least `(1-gamma)n` positions.
The error has no factor for the dimension of `U`.

The checked steps are:

1. Put `V=U-U`. For each nonzero `v in V`, partition `U` into parallel
   lines with direction `v`. Their mean close fraction is the close fraction
   of `U`. One line exceeds the threshold. The line theorem gives codewords
   for both its offset and `v` on a common set. Thus every `v in V` is close.
   This is exactly the argument of BCIKS20 **Lemma 6.3**, whose statement is
   on printed p. 27 and proof on printed p. 28 (PDF pp. 28-29).
2. This alone bounds the directions, not all affine points. This missing step
   in the former summary is needed. If `0 notin U`, `span(U)` is the disjoint
   union of `V` and the sets `zU` for nonzero `z`. Multiplication by nonzero
   `z` preserves distance to a linear code. All of `V` is close, and each
   `zU` has the original close fraction. Apply step 1 again to `span(U)`.
   It follows that every point of `U` is close. If `0 in U`, the first
   application already gives this result.
3. Choose `u* in U` at the maximum distance `gamma* <= gamma` from `C`.
   If `gamma*=0`, then `U` is contained in `C` and the claim is immediate.
   Otherwise list all codewords nearest to `u*`. Each has distance exactly
   `gamma*`. Let their exact agreement sets be `A_i`. No shorter-distance
   codeword exists. This exact equality is used in step 5.
4. Define `U_i={u in U: u|A_i in C|A_i}`. It is an affine subspace or empty,
   since restriction is linear and `C|A_i` is linear. It is nonempty because
   it contains `u*`. The codeword for `u` can differ from the codeword for
   `u*`; equality of those two codewords is not required.
5. For each `u != u*`, every point on their line lies in `U` and is
   gamma*-close. Theorem 1.5 applies at `gamma*`: its numerator is at most
   `a(gamma) < Q`. To check monotonicity, fix `rho`; both the ceiling that
   defines `m` and the positive expression in `m` and `gamma` are
   nondecreasing in `gamma`. The line theorem gives an agreement set `A`
   of size at least `(1-gamma*)n` for codewords of `u*` and `u`. The first
   codeword is on the complete nearest list, so `A ⊆ A_i`. Both sets have
   the same required size, by step 3; hence `A=A_i`. Therefore `u in U_i`.
6. Thus fewer than `Q` affine subspaces `U_i` cover `U`. A proper affine
   subspace has at most `|U|/Q` points. At least one `U_i` must be all of
   `U`. Its set `A_i` is the required joint agreement set. Its size exceeds
   `K-1`, so restriction determines each degree-`<K` codeword uniquely.

For the compiled Johnson radii, the list bound is at most
`1/(2 eta sqrt(K/n))`, far less than `Q=127^20`; the counting proof is in
spec 12.3. The argument uses a uniform element of `U`, which independent
uniform column coefficients provide even when columns are dependent.
There is no coefficient-count factor and no doubled `m` for this CA claim.

**Verdict:** the dimension-free ordinary CA reduction is valid with the stated
hypotheses. J1 is discharged as a mathematical derivation, subject to review.
It does not prove MCA on every agreement set, descent for `E tensor F4`,
or soundness of openings. Those are separate steps in J2/J3 and P1/P4.

**Grinding.** Before every level's positions are sampled the prover absorbs an 8-byte nonce whose
grind word (the first u32 of squeeze block 0) has `grind_bits` leading zeros; the positions come from
block 1 on (`transcript.grind`, `HostTranscript.grind`). Finding it costs `2^grind_bits` hashes per
level per attempt, so a prover that makes `W` transcript attempts pays `W 2^grind_bits` hashes; the
ledger charges the query term per attempt divided by `2^grind_bits`, the usual grinding accounting
(ethSTARK, section 6.2 of the Ligerito note), and sizes `s_i` from `lambda' - grind_bits`. This is a
computational bound in the random-oracle model, not a statistical one: it is part of P5's claim, and
the field terms are untouched by it. The search runs on device (smallest passing nonce, 3 ms at 20
bits on the M1 Pro) and ends with probability 1.

`A_pcs_batch` is conservatively the number of scalar claims batched before each three-round
sumcheck: `4*s_previous+1` at the first tail level, `s_previous+1` subsequently. The factor four
comes from `_Tail.level`'s four coordinate equations per level-1 query. These use independent
uniform coefficients, so a sharper batching analysis may reduce this allowance to one per batch;
the ledger deliberately keeps the larger count. Final clear-vector checks are direct. The initial
column/point batching gets another `2/Q`.

## Relation numerators

These allowances assume that the decoded columns are fixed at their respective commit barriers,
the packed encoding is an isomorphism to bounded-degree trace polynomials, the workload lowering
is faithful, and the checks compose as described below. They are not a proof of those assumptions.

| Ledger name | Numerator over Q | Reason / code |
|---|---|---|
| `alpha_batch` | largest grid family index + largest small-grid family index | Powers of alpha, two polynomial batches; indices read from entries/ends/wires, not the number of entries |
| `grid_identity` | `2 h1 + 2 h2` | Bidegree bound on R minus `(A + X2^h2 B)(X1^h1-1) + Q2(X2^h2-1)`; `verifier._residual` |
| `small_grid` | `3 h2` when accumulators exist | Degree of R2 minus `Q3(X2^h2-1)`; `verifier._small_grid` |
| `starts_and_zero_rows` | `(accumulators + zero_rows)(h2-1)` | Union allowance for the separate random-z2 line checks in `_boundaries` |
| `restrictions` | sum of `max(h1, public_line_length)-1` | Separate random-z1 line equalities in `_restrictions` |
| `horner_identities` | maximum challenge degree of a chain-end term | Coefficient identities after substituting the Horner recurrence; see below |
| `wire_fingerprints` | maximum challenge degree of an endpoint used as a slot/public factor | Selected unequal underlying values must have unequal fingerprints except on a root event; with a row-group selector on the ingest (2026-09-17) the map is injective on the selected coordinates only, so the verifier refuses a public factor whose selector is zero on its chain |
| `wire_product` | `F = slots*h2 + public_factors` | Joint permutation product in the independent `(beta_w, gamma_w)` challenges |
| `wire_zero_factors` | `2 F` | Union allowance for a zero numerator/denominator factor, conditional on the earlier fingerprint challenges |

The calculator propagates total degrees through `Shape.chals`: sampled challenges have degree 1,
one has degree 0, addition takes the maximum, multiplication adds. For a Horner recurrence
`R_next = scale*R + ingest`, there are `h1-1` transitions. With scale degree d and maximum ingest
degree a, the endpoint degree is bounded by

```
max((h1-2)*d + a, (h1-1)*d if start != 0 else 0).
```

A chain-end product adds its operands' degrees and its explicit challenge weight. This bound uses
the full compiled row range, so it can exceed a hand-derived bound exploiting zero selectors.
It avoids assuming that every workload has the old spec's approximately 514-degree mulmod shape.

The maximum endpoint degree is a **single bad-identity allowance**, conditional on proving that a
false semantic relation determines a nonzero challenge polynomial before those challenges are
sampled. All identities have to hold: select a fixed false one, rather than union-bounding every
honest identity. This argument needs the product-piece selectors, integer range bounds, and distinct
rho powers to prevent symbolic cancellation. For wiring, first select a fixed unequal edge, then
use its fingerprint bound and the joint permutation argument in fresh independent challenges.
Nonzero factors let the recurrence telescope; the `2F` allowance covers their possible zeros.

Herder's table and grand-product terms are intentionally **not implemented** in this executable.
The four CSP workloads compile with zero KIND_LOOKUP/KIND_PERM accumulators. Adding one fails the
ledger instead of silently using a formula that omits table or zero-denominator events.

## Proof-to-code map and status

| ID | Obligation | Evidence / remaining work |
|---|---|---|
| P1 | Packed level-1 columns admit the scalar RS guarantee in the block metric | Vault spec 12.1; scalar citation checked above. Independently review descent, reconstruction of F-valued coordinates, and the joint opening claim |
| P2 | Two-grid identities imply all intended row and chain-end constraints | `relations/statement.compile`, `proof.Shape`, `verifier._residual`, `_small_grid`, `_boundaries`, `_restrictions`. Review degree bounds and adaptive transcript composition, including the shared alpha and z |
| P3 | Integer/curve/hash constraints faithfully express the workloads | `workloads/sha256`, `keccak`, `poseidon`, `mulmod`, `ecdsa`. Check carries, selectors, canonical values, idle rows, hints, and the nonzero-polynomial argument for Horner/wiring; degree counting alone cannot establish it |
| P4 | The tail is the analyzed scalar tensor-fold protocol | `_Tail.level`, `_Tail.clear`, `pcs/tensor`, `pcs/tail`. Review row basis, mixed digits, four-coordinate first batch, last clear check, and per-round adaptivity |
| P5 | Cryptographic compilation preserves the required concrete security | `core/hash`, `core/transcript`, `proof.prefix_bytes`. Establish the exact Fiat-Shamir/hash model and losses, including a separate quantum claim if desired |
| J1 | The batched-column fold admits a dimension-free ordinary CA numerator | **Reduction checked: own proof, needs review.** BCIKS20 section 6.3, printed pp. 28-29, accepts BCHKS25 Theorem 1.5, p. 9, as its line input. The six checked steps below include the linear-span step, monotonicity at the farthest radius, and list size `< Q`. MCA and the packed alphabet remain separate |
| J2 | Mutual (list) correlated agreement at the Johnson radius | **Source/transfer closed:** Hab25 Theorem 2, p. 4, is public (ePrint 2025/2110). Main ledger charges its quadratic numerator. Jo 2026/891 Corollary 4.6, p. 9, and Theorem 4.7, pp. 10-11, supply the interleaved/tensor transfer. Packed level 1 and adaptive running claims remain J3/P4 |
| J3 | The level-1 close case binds one list element, so `sqrt(rate) + eta` is the right per-query event | Vault spec 12.2 is a sketch. The out-of-domain Bind allowance (quadratic in the list size) is not a ledger term; 12.1's Galois descent and `n_cw > 1` split need list-regime proofs; `verifier._residual`, `_small_grid`, `_boundaries` and `_restrictions` must bind the chosen element |
| I1 | Every adversarial field coordinate obeys the arithmetic contract | **Implemented:** `bytes.check_field_bytes` rejects coordinates >=127. `ProofReader.field_bytes` covers Z2, Q3, openings, every sumcheck, and the clear vector; `pcs.merkle.check_multiproof` checks authenticated W/Z/Q/tail rows; `_check_statement` checks derived public field data |
| I2 | Optimized arithmetic agrees with the field for all admitted inputs | `core/field` assumes canonical bytes and explicit integer/fp32 bounds; existing scalar/GPU tests cover samples. Audit the bounds and keep differential checks on adversarial canonical values |
| I3 | Report remains tied to the submitted configuration | Compare the case whitelist with both CLI routes; run against the pinned submission commit. Recompute after changing field, grid, rates, arity, or workloads |

I1 was an observed missing precondition, not a demonstrated end-to-end forgery. For example,
`f_add` uses byte addition under the assumption that the sum is below 254. Allowing arbitrary
proof bytes violates that premise. Merkle authentication establishes that bytes match a commitment;
it does not make those bytes canonical field elements. The verifier now rejects noncanonical
coordinates before arithmetic. Roots, sibling digests, length words, and raw public-input bytes
remain opaque; only values interpreted as field coordinates get this check. This closes the
identified parsing gap without establishing I2 or the protocol-level obligations.

The regression tests cover every byte value 127 through 255 in the field reader, both ends of
each direct proof-field region (including all recursive levels), and derived public field data.
Merkle tests recompute matching roots for malformed F4 and E rows, ensuring the failure is field
validation rather than failed authentication; canonical 126 and opaque 255-valued sibling hashes
are accepted. Existing valid proofs still round-trip.

The transcript order in `Prover.prove` and `verify` currently matches the intended barriers:
W -> stage-1 and wiring challenges; Z/Z2 -> alpha; Q/Q3 -> z; openings -> independent beta/gamma;
each tail root -> previous-level queries/batching; each sumcheck message -> its round challenge;
clear vector -> final queries. `prefix_bytes` binds the parameters, compiled descriptors, tables,
and public inputs. `sample` uses rejection sampling for both F127 coordinates and bounded positions.
These are code observations, not substitutes for P5.

## Validation and next decision

The executable self-check covers a hand-computable query probability, degree propagation, nonzero
Horner starts, summation of error probabilities, invalid inputs, and monotonicity. Two tiny compiled
statements also check hand-derived full ledgers with and without a committed tail, including its
final query error and the four-coordinate first batch. Build it with
`--Werror`; the existing `run_tests.sh` also compiles every bench file.

For this snapshot, all 16 reports completed and their query errors and displayed bit totals were
independently reproduced with 80-digit Decimal arithmetic. After the canonical-byte fix,
`sh run_tests.sh` passed all 106 tests across 23 test files and built all 11 benchmarks with
`--Werror`. Running the ledger again produced byte-identical output. Native and Claude reviews
found no implementation defects in the validation change; those reviews do not discharge P1-P5.

Existing adversarial tests already do more than corrupt serialized proofs:

- `test_prover`: invalid permutation and lookup traces, broken Horner chains and wiring, public
  restrictions, transcript-bound point/challenge descriptors, and recursive tails.
- `test_mulmod`: wrong product coefficients, idle-row carries, and circuit wiring.
- `test_sha256`, `test_keccak`, `test_poseidon`: newly generated proofs against false digests;
  Poseidon also mutates coefficient witnesses.
- `test_tensor`, `test_open`, `test_field`, `test_transcript`: basis/functionals, interpolation,
  field arithmetic, and host/device transcript agreement.

ECDSA's changed-message round trip also exercises its native public-data walk; it is not by itself
evidence that malicious slope witnesses are constrained. Review that path separately. No practical
number of successful random tests measures a failure probability near 2^-96.

Before submission: close P1-P5 and the lowering/code
obligations with independent cryptographic review, then evaluate the complete claim at the chosen
target. A better theorem (the list-decoding regime, spec section 12) is the lever on the query term
and the proof size; adding queries cannot reduce the `A/Q` terms, and the extension is now sized so
they sit below the query term. Only then treat the `security_bits: 112` metadata as more than a
query target.
