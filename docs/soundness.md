# Soundness ledger for covered configurations

Status: **conditional analysis, not certification**. The executable ledger
uses the current source with `E=F_(127^20)` and query target 112. This note
records checked citations, our proof steps, and the remaining composition
conditions. A successful calculation or test does not establish a security
level.

**The numbers, and which is which.** There are several conditional bounds for the same
protocol and parameters (the executable prints all but the linear MCA row, which is a separate
research calculation); they differ in the theorem used for the level-1 field term and
in the list model, not in the prover. All are for E = F_(127^20), and none is a measured attack
cost or a certified level.

| Bound | ECDSA | The 16 CSP cases | Status |
|---|---:|---:|---|
| Main ledger: Hab25 Theorem 2 with the J3 list unions, work bits | 87.28 | 87.28-99.52 | conservative baseline, public scalar source; the four RSA and passport statements give 81.01-83.88 |
| Linear MCA: BCHKS25 Theorem 4.6 as the scalar input, work bits | 102.36 | 102.36-105.91 | research projection, the theorem's proof is a sketch |
| DKT26 Theorem 5.12, old list bounds, work bits | 106.41 | 105.67-106.41 | conditional comparison |
| DKT26 with the joint block-list cap, interactive / work bits | 88.21 / 108.03 | 87.50-88.21 / 107.41-108.18 | conditional comparison; the [checkpoint](security-assurance.md) covers 20 cases including RSA, SOD, DSC and the passport (87.50-88.37 / 107.12-108.18) |

"Work bits" discount the query error by the 20 grinding bits; "interactive bits" do not (the
formulas are in the checkpoint; the main ledger's interactive bits for the 16 CSP cases are 85.36-86.31). The tables below print work bits unless a column says otherwise.
No complete implementation guarantee is established: the compact E-column fold needs an explicit
connection to the checked MCA and descent model (P1/P4), and the cryptographic work reduction
remains open (P5). The next step is the review of the shared proof-to-code and cryptographic
composition arguments (P1-P5 below), including constraint correctness; repeating the calculation
for more workloads does not close them.

The benchmark metadata says `security_bits: 112`. In `core/params.mojo`, 112 is the per-level
**fixed-message query target**, sized as 92 bits of queries at the Johnson radius `1 - sqrt(rate) - 1/16` (BCHKS25
Theorem 1.5 below) plus 20 bits of grinding on every query seed, with
`E = F_(127^20)` (20 coordinates; `E16` in `core/field.mojo` rebuilds the old
16-coordinate tower for measurement). It is not the total soundness budget. Do not use the metadata
as a verified claim. The recursive list union factors reduce the query-only
bound below that target; the ledger includes them. The upstream
[eligibility rule](https://github.com/ethereum/csp-benchmarks/blob/main/CONTRIBUTING.md#benchmark-eligibility)
requires at least 96 bits; the project's own goal is more than 100 bits.

## Claim and composition

The desired claim is that an adversarial prover cannot make the verifier accept a false workload
statement, except with bounded probability. The specification cited by section number below
("spec 12.1", "spec 12.3") is not published; the arguments the ledger charges from it are restated in
J1-J3. The adversary may write every witness and proof byte;
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

**Unique-regime comparison.** The Johnson charges are stated in the output
guide and J3 below. Let `Q = |E|`, `L_i` be the code length, `k_i` the message dimension, `s_i` the query count, and
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

**Scalar citation.** Ben-Sasson, Carmon, Ishai, Kopparty, and Saraf (BCIKS20,
[version of July 7, 2020](https://sites.math.rutgers.edu/~sk1233/GS-FRI.pdf#page=3)), Theorem 1.2,
gives error `L/Q` in the unique-decoding regime. Theorem 1.7 in that version supplies
correlated agreement for affine spaces with the same error, independent of the dimension of the
space (Theorem 1.6 there is about parameterized curves; theorem numbers differ between versions).

Uniform coefficients in `E^columns` induce a uniform element of their span even when columns are
dependent. The scalar theorem therefore needs no column-count multiplier. The separate transfer
to Caracal's four-coordinate `E tensor_F F4` alphabet is the lemma of spec section 12.1:
split into four conjugate RS codes over E, descend a unique nearby word to F4, and use the block
metric to check all coordinates at the same position. **This transfer and its composition with
opening claims remain review obligations**, not consequences of finding the citation.

For the tail, three tensor challenges give `3 L_i/Q`, conditional on the interleaved/tensor
proximity result; see Diamond and Gruen,
[Proximity Gaps in Interleaved Codes](https://eprint.iacr.org/2024/1351), Theorems 3.1 and 3.6,
Corollary 3.7, and the Ligerito construction. The last committed code is included even though its
folded message is sent in the clear. There is no extra code or sumcheck after that clear message.

**The configuration.** `CLIENT` compiles `REGIME_JOHNSON` with `eta = 1/16` and the tail rate rule
`tail_rate_inv = 8` in place of spec 9.5's 1/32. The ledger's ECDSA sweep over the tail rate (all at
74 level-1 queries) is why: at 1/32 the tail levels sit near rate 1/60, where the `rho^-1.5` constant
of the scalar theorem is large and the field term caps the case; 1/8 costs 288 queries in all, and its tail domains are seven times smaller (level 2:
92,160 against 645,120), which the prover's tail encode and the verifier's openings both see. The
executable prints the four tail rates 1/4, 1/8, 1/16 and 1/32 as comparisons. A per-level `eta` (the
tail at rate 1/8 could take `eta = 1/24` with `m` still 3) is a small further lever not taken.

**Grinding.** Each level absorbs an 8-byte nonce. Squeeze block zero must
pass the leading-zero test; positions use block one onward. The executable
prints undiscounted interactive error and a separate work diagnostic that
divides only the query error by `2^grind_bits`. A fresh ideal probe passes
with probability `2^-grind_bits`; this does not prove a deterministic cost
per accepted attempt or a concrete adversarial success bound. Prefix reuse,
adaptive attempts, Fiat-Shamir, and hash binding remain in P5. The device
search probes only 32 nonce bits and can repeat after wraparound. It does
not have a probability-one termination guarantee for a fixed state. See
[the checkpoint](security-assurance.md) for the exact scope.

`A_pcs_batch` is conservatively the number of scalar claims batched before each three-round
sumcheck: `4*s_previous+1` at the first tail level, `s_previous+1` subsequently. The factor four
comes from `_Tail.level`'s four coordinate equations per level-1 query. These use independent
uniform coefficients, so a sharper batching analysis may reduce this allowance to one per batch;
the ledger deliberately keeps the larger count. Final clear-vector checks are direct. The initial
column/point batching costs `2B/Q` in the Johnson regime and `2/Q` in the
unique regime, as derived in J3.

### Open obligations of the Johnson regime (J1-J3)

J1 has a checked ordinary CA derivation. J2 has public scalar and transfer
citations. J3 has local and recursive list arguments with explicit union charges.
These own proofs need review; P1-P5 remain open.

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

**J1, the reduction in detail. Own proof, needs review.** The sources are
BCIKS20 **Theorem 1.4, printed p. 3 (PDF p. 4)**, **Theorem 1.7,
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
   This is exactly the argument of BCIKS20 **Lemma 6.3** (section 6.3).
2. This alone bounds the directions, not all affine points; the next step is needed. If `0 notin U`, `span(U)` is the disjoint
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
hypotheses. J1 is checked as a mathematical derivation, pending review.
It does not prove MCA on every agreement set, descent for `E tensor F4`,
or soundness of openings. Those are separate steps in J2/J3 and P1/P4.

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
still includes `M = opened-1` at level 1, as a power-curve comparison only.
The protocol still samples independent column coefficients. Neither projection
is a proof for that sampler. Historical projections retain their old rate
normalization. The main Hab25 calculation pads Float64 upward by `1e-12` before
rounding; this is diagnostic sizing, not interval certification.

**Transfer source.** Diamond and Gruen **Theorem 3.1, p. 6**, and
**Corollary 3.7, p. 9**, restrict the radius to unique decoding. The proof uses
uniqueness in **Lemma 3.2, p. 7** to identify every nearby word with one fixed
word. That step fails for lists. **Theorem 3.6, p. 9**, itself has no unique-radius
restriction: it assumes the required line gap for every interleaving.

Jo, [ePrint 2026/1432](https://eprint.iacr.org/2026/1432), p. 5,
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
J1 proves ordinary CA, not affine MCA. Jo Theorem 4.7 alone would charge
one line numerator per column variable. J3 supplies a separate dimension-free
same-set reduction as an own proof and charges all four conjugate codes.

**Other recent sources.** An arbitrary-dimension affine MCA theorem with the pairs
numerator is **not stated** in four newer papers. The checked statements are:

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

**J3. List argument and conservative composition: own proof, needs review.** Spec
**12.3** states the list-regime lemma. All its new reductions are
**own proof, needs review**. The close list has at most
`1/(2 eta sqrt(K/n))` members in the joint block metric. The proof counts
pairs of agreement sets; it does not take a product of column lists.

The proof extends line MCA to a uniform affine space with error `a/(Q-1)`.
It averages over a random direction through each bad point. At least
`1-1/Q` of the directions retain a fixed bad agreement set, and the line
bound caps the average at `a/Q`. For integer `A>=a`, `A<Q`, charge `(A+1)/Q`.
Use this once for each of the four conjugate RS codes. This gives the new
level-1 charge `4(A_H+1)`, rather than one scalar numerator. Jo 2026/891,
Corollary 4.6, p. 9, handles all codeword splits within each conjugate code.
For a block-close fold, MCA keeps one common agreement set across every
column and split. Interpolation on more than `K-1` points makes each decoded
polynomial F4-rational and makes the four conjugate decodings agree. This
proves descent without uniqueness and covers `n_cw > 1`. The executable
still rejects `n_cw != 1`; its workload scope has not been enlarged.

The extractor selects the candidate that matches the out-of-domain answer.
The event **Bind** is a collision of two candidates from a list fixed before
`z=(z1,z2)`. A difference of trace polynomials has total degree at most
`h1+h2-2`. Thus, with `B=ceil(1/(2 eta sqrt(K/n)))`, the new term is

```
A_list_bind = opened * B(B-1)/2 * (h1+h2-2).
```

Off Bind, an answer identifies at most one candidate per column. It need
not identify any candidate. For each fixed joint candidate with a wrong
opening, the beta/gamma batch is a nonzero degree-two polynomial. Union over
the joint list costs `2B/Q`. The ledger therefore changes `opening_batch`
from 2 to `2B` as well. Here `opened` is the number of opened columns, one per E-valued
column (`Shape.opened()`). ECDSA has `B=23`, `opened=252`, `h1=144`, `h2=576`,
so `list_bind=45,776,808` and `opening_batch=46`.

For a message fixed before the row queries, the proof now has the required
case split. A far message costs at most `(sqrt(K/n)+eta)^s`; a close wrong
message is one of the charged field events. There is no list factor for this
**fixed-message** query statement.

**Conservative composition.** A root fixes a list before the previous-level
queries, not one member. Spec 12.3 now pays for that choice. If level i has a
next committed root, its query term is `B_(i+1) miss_i^s_i`. The last clear
message is fixed before its queries and uses factor one. The factor is
outside the exponent; the old unmultiplied claim is not proved.

Extract backward from the final clear message. A close scalar fold lifts by
tensor MCA to one member of its root's joint list, fixed before the three
sumcheck rounds. For each fixed member, false-claim sumcheck error is `6/Q`
and the old batching allowance is `A_batch_i/Q`. Union over the list costs
`B_i(6+A_batch_i)/Q`. Off these events, the chosen member satisfies the
previous running claim and all previous queried equations. It is one of
at most B_i messages fixed before those previous queries, so the preceding
query union applies. Repeat until level 1, then use its four-component MCA,
`2B_1/Q` opening batch, and Bind. This argument allows selection after the
queries; it does not assume an earlier unique decoding.

For the relation checks, use the W list before stage-1 challenges, W/Z before
alpha, and W/Z/Q before z. Each joint list has at most B_1 members. Apply the
existing fixed-candidate bound at the correct barrier to every candidate,
then union over the list. The extra term is
`list_relations=(B_1-1) * sum(the nine relation numerators)`. This handles a
choice that depends on z; Bind alone did not. The nonzero-polynomial and
code-mapping contracts of P1-P4 remain required. The extra list-selection
assumption is removed. The augmented-RS-coordinate sketch in 12.2 is not used.

**Verdict:** J3's list accounting is supplied with conservative union bounds,
subject to review of this own proof and the existing P1-P5 contracts. The
ledger charges the unions. A sharper result with no next-list query factor
is **not proved** here.

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
| `wire_fingerprints` | maximum challenge degree of an endpoint used as a slot/public factor | Selected unequal underlying values must have unequal fingerprints except on a root event; with a row-group selector on the ingest the map is injective on the selected coordinates only, so the verifier refuses a public factor whose selector is zero on its chain |
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

## Run and interpret

```sh
uv run mojo run --Werror -I src bench/bench_soundness.mojo
```

The program runs its arithmetic self-checks, compiles 20 main cases and four tail-rate comparisons, and prints their
actual `Params`, `Shape`, tail levels, query error, and named field-error numerators. It neither
constructs a GPU context nor generates a witness or proof. SHA-256 and Keccak input bytes,
Poseidon elements, and ECDSA signature values do not affect the compiled statement at a fixed grid.
The ECDSA report calls the workload's `statement()` and its fixed `Ecdsa.circuit()` without a live walk.

The first 16 cases cover the grid geometries in `cli/ffi.mojo` and `cli/main.mojo`;
four more use the primary RSA, SOD, DSC, and passport benchmark statements. Those routes must be compared again
when adding sizes or changing routing. All dimensions, descriptors, and counts after routing come
from production compilation; there is no copied table of shape counts or reimplementation of the
domain/tail planner. The ledger rejects Herder lookup/permutation accumulators (the covered workloads have none), split tail codewords, other tail arities, and challenge-dependent ordinary families that would need
another argument. This is scoped to these workloads, not a generic IR security checker.

Output:

- `field_numerator NAME A` means the conditional contribution `A / 127^e`.
- `query_error_per_attempt` sums every committed level, with the next
  committed root's list-size factor (one for the final clear message). `query_error` divides it by `2^grind_bits` under the work
  model in the Grinding paragraph.
- `REGIME_JOHNSON` uses radius `gamma=1-sqrt(K/n)-eta`. Its main charge is
  Hab25 Theorem 2, p. 4: `a_H=(m+1/2)^7 n^2/(3 rho_0^1.5)`, where
  `rho_0=(K-1)/n`, `eta_H=1-sqrt(rho_0)-gamma`, and
  `m=max(ceil(sqrt(rho_0)/(2 eta_H)),3)`. Level 1 charges `4(ceil(a_H)+1)`;
  each scalar tail charges `3 ceil(a_H)`. See J2 and J3 for the scope.
- `list_bind` charges `opened * B(B-1)/2 * (h1+h2-2)`, `opened` the number of opened
  columns (`Shape.opened()`, one per E-valued column since the single-value openings), where
  `B=ceil(1/(2 eta sqrt(K/n)))`. `opening_batch` charges `2B` in the
  Johnson regime. `list_relations` adds `(B-1)` times the relation allowances.
  Batching and sumcheck at a tail root are multiplied by that root's list
  bound. In the unique regime all list factors are one and Bind is zero.
- `johnson_bchks25_1.5_projection` and `johnson_bchks25_4.2_4.6_projection`
  retain the earlier scalar gap charges, rate convention, and curve factor.
  The new non-gap list terms are included. These comparisons do not supply
  the packed-alphabet proof. `capacity_conjecture` has no proven gap or list
  bound; its field terms are placeholders, not a security claim.
- `conditional_interactive_bits` uses `query_error_per_attempt + sum(A)/127^e`.
- `conditional_work_bits` uses `query_error_per_attempt / 2^grind_bits + sum(A)/127^e`.
  It replaces the misleading `conditional_iop_bits` label. This discount has
  not been proved for the implemented transcript. Both displays round down
  to two decimals. The e16/e20 projections change only the denominator.
- `johnson_dkt26_joint_list_conditional` uses the same DKT26 scalar charge
  and the checked integer block-list allowance described in the checkpoint.
  The main Hab25 and DKT26 old-list comparisons remain available.
  `numeric_code` prints each actual compiled level for certificate comparison.

Calculations use Float64 for diagnostic sizing, not interval arithmetic or a machine-checked proof.
Exit zero means the calculation succeeded. The program always reports unresolved obligations and
never emits `security_bits` or a certification pass. Fiat-Shamir, hash binding, quantum security,
and the gaps below are not assigned zero error; they are **outside this conditional total**.

### Current results

These are conditional diagnostic bounds. The local and recursive list arguments are own proofs that need review.
P1-P5 remain proof-to-code and cryptographic obligations. They are not certified
security levels or measured attack costs.

| Workload | Input | Grid | Main ledger: Hab25 and list allowances, e = 20, work bits |
|---|---:|---:|---:|
| SHA-256 | 128 B | 32 x 224 | 94.83 |
| SHA-256 | 256 B | 32 x 336 | 93.92 |
| SHA-256 | 512 B | 32 x 672 | 91.66 |
| SHA-256 | 1024 B | 32 x 1152 | 89.92 |
| SHA-256 | 2048 B | 32 x 2688 | 87.36 |
| Keccak | 128 B | 64 x 24 | 99.52 |
| Keccak | 256 B | 64 x 48 | 97.19 |
| Keccak | 512 B | 64 x 96 | 95.19 |
| Keccak | 1024 B | 64 x 192 | 93.09 |
| Keccak | 2048 B | 64 x 384 | 91.09 |
| Poseidon | 2, 4, 8 elements | 64 x 384 | 91.09 |
| Poseidon | 12, 16 elements | 64 x 896 | 87.60 |
| ECDSA | 1 signature | 144 x 576 | 87.28 |
| RSA-2048 | benchmark fixture | 144 x 2016 | 83.88 |
| SOD | benchmark fixture | 144 x 2688 | 81.01 |
| DSC | benchmark fixture | 144 x 2688 | 81.01 |
| passport | benchmark fixture | 144 x 4032 | 81.88 |

ECDSA is **87.28 conditional bits**. The quadratic J2 numerator (Hab25) is the dominant term;
the four-component level-1 charge, Bind, the larger opening batch and the list unions are small
beside it, but all are charged. More queries do not reduce the dominant
field term. The executable also prints the pairs form of BCHKS25 Theorem 1.5 (the
`johnson_bchks25_1.5_projection`, 106.00 bits for ECDSA with the list terms included) as a labelled
projection, not a bound.

### DKT26 Johnson MCA comparison

**Status: conditional comparison.** The executable now prints
`johnson_dkt26_5.12_conditional` beside the main Hab25 result. With the same
E20 parameters, ECDSA gives **106.41 conditional bits**. The 16 recorded CSP
cases give **105.67-106.41**. The prover, verifier, field, proof format, query
counts, and all non-gap allowances are unchanged. P1-P5 remain open.

DKT26 is Dao, Kominers, and Thaler, *Reed-Solomon Codes Beyond Johnson:
Efficient Decoding and Smaller Cryptographic Proofs*; the authors' source, with the Johnson
constant checks, is [quangvdao/rs-beyond-johnson](https://github.com/quangvdao/rs-beyond-johnson).

**Theorem 5.12, p. 53**, applies to degree-at-most-D RS codes on any n
prescribed distinct field points, in every characteristic. It requires
`1 <= D <= n-2` and agreement `a > sqrt(D/n)`, with `a <= 1`.
It supplies full agreement-set MCA, with error at most `E_line/Q`.
The numerator has order `n/eta_0^3` at fixed rate, where
`eta_0=a-sqrt(D/n)`. This result stays at the Johnson radius. The improved
count reconstructs only the actual message coefficients and separates the
message and challenge degrees (**section 5.7, pp. 53-54**).

For each compiled level, set `D=K-1` and retain `eta=1/eta_inv`. Set
`A=ceil(sqrt(n*K)+n/eta_inv)` and apply the theorem with `a=A/n`.
This is exactly the configured integer Hamming ball, not a change to queries.
The scalar calculation is:

```
m = max(ceil(sqrt(D/n)/(2*(A/n-sqrt(D/n)))), 3)
t = m + 1/2
B = ceil(t/sqrt(D/n)) - 1
H = ceil(t^2/(3*D/n)) - 1
Psi = 1 + (2*D-1)*(2*B-1) + 2*max(0, B-2*D-1)
E_line = (2*B-1)*H + (n-D)/(A-D)*(B+H*Psi) + (n-D-1)*B
C_i = ceil(E_line).
```

This is Theorem 5.12's `B_cf`, using the ordinary numerator of **Lemma 5.3,
pp. 40-43**, at curve degree one and witness threshold `L=D+1`. No free
threshold optimization is used. The executable computes these ceilings by
integer comparisons and rejects intermediate Int overflow. Only the final
probabilities and bit display use Float64.

**Corollary 7.2, p. 66**, supplies affine MCA with error `E_line/(Q-1)`,
independent of the number of coefficients. **Corollary 7.7, pp. 68-69**,
supplies the interleaved form without a width factor. **Corollary 7.8,
pp. 69-70**, supplies the shared-level multilinear fold bound `h*E_line/Q`.
All retain the scalar hypotheses. These statements supply public references
for the affine, interleaved, and three-level transfers used in this comparison.
The Caracal packed descent, joint-list argument, and adaptive extraction are
still **own proof, needs review**. These scalar transfers do not prove them.

The comparison retains the four conjugate-code allowance at level 1:

```
A_gap = 4*(C_1+1) + 3*sum(C_i for committed levels i >= 2)
conditional_bits = -log2(P_query + (A_gap+A_other)/127^20).
```

`C_1<Q` justifies `E_line/(Q-1) <= (C_1+1)/Q`; the executable checks this
condition. `A_other` and `P_query` come directly from the main ledger.
In particular, this comparison keeps the old list bounds, Bind, opening,
relation, sumcheck, next-list query factors, and grinding model. It does not
substitute the smaller scalar list bound in Theorem 5.12 for the packed joint
list. Fiat-Shamir, hash binding, and quantum-security losses remain outside
the total.

For ECDSA, the level arrays are `K=[20736,10368,1296,162]`,
`n=[161280,92160,10752,1344]`, and `s=[74,70,72,72]`, with `eta_inv=16`.
They give `A=[67910,36672,4405,551]`, `m=[3,3,3,3]`, `B=[9,10,10,10]`,
`H=[31,36,33,34]`, and `C=[66374040,44914431,5031516,641610]`.
Thus `A_gap=417258835`, `A_other=46706513`, and the total field numerator
is `463965348`. The query-only bound is 106.48 bits. The field-only bound is
110.98 bits; adding probabilities gives 106.41 bits, not their sum.

| Workload | Input | Hab25 conditional bits | DKT26 conditional bits |
|---|---:|---:|---:|
| SHA-256 | 128 B | 94.83 | 106.31 |
| SHA-256 | 256 B | 93.92 | 106.14 |
| SHA-256 | 512 B | 91.66 | 106.31 |
| SHA-256 | 1024 B | 89.92 | 105.77 |
| SHA-256 | 2048 B | 87.36 | 106.03 |
| Keccak | 128 B | 99.52 | 106.15 |
| Keccak | 256 B | 97.19 | 106.20 |
| Keccak | 512 B | 95.19 | 106.20 |
| Keccak | 1024 B | 93.09 | 105.78 |
| Keccak | 2048 B | 91.09 | 105.78 |
| Poseidon | 2, 4, 8 elements | 91.09 | 105.78 |
| Poseidon | 12, 16 elements | 87.60 | 105.67 |
| ECDSA | 1 signature | 87.28 | 106.41 |

All bits are rounded down. The main Hab25 result is kept beside the new
comparison. The separate ECDSA tail-rate checks at inverse rates 4, 8, 16,
and 32 give DKT26 totals 106.46, 106.41, 106.06, and 105.30, respectively.
Recompute the full ledger after any field, shape, rate, query, or workload
change. These results do not cover unlisted configurations.

The beyond-Johnson result is separate. **Theorem 1.1, p. 6**, requires
characteristic `p > max(K-1, B_partial)` for its stated quantitative bounds.
All four current ECDSA dimensions fail even `p>K-1` at `p=127`.
E24 also has characteristic 127. A larger extension cannot fix that hypothesis.
This is not a claim that every beyond-Johnson result must fail for our codes.

The next security work is to review the packed and adaptive composition,
then consider query parameters. ECDSA is close to its query-only bound under this
comparison, so increasing the extension degree alone cannot raise it with these queries.

### E20 linear MCA research comparison

**Status: research projection.** The recorded ECDSA case gives **102.36
conditional bits with the same protocol and parameters**, if the scalar input
is BCHKS25 **Theorem 4.6, pp. 28-29** and the J3 transfers hold. The source is
[ePrint 2025/2055](https://eprint.iacr.org/2025/2055). Its proof is a sketch.
The main ledger still uses Hab25 **Theorem 2, p. 4** and reports **87.28**.
This comparison changes the bound on the protocol's error. It has no effect
on the prover, verifier, proof bytes, or an attacker's actual success probability.

The scalar input is Theorem 4.6 with `M=1`, `rho=(K-1)/n`, and its doubled
`m_B`, as written in J2 above. For each level, set
`t=floor(n*(1-sqrt(K/n)-eta))` and use `gamma=t/n` in that theorem.
This is the same Hamming ball as the configured radius. Round the scalar
numerator up to `A_i=ceil(a_B)`. The research calculation then charges

```
A_gap = 4*(A_1+1) + 3*sum(A_i for committed levels i >= 2)
P_query = sum(B_(i+1)*(sqrt(K_i/n_i)+eta)^s_i) / 2^20
conditional_bits = -log2(P_query + (A_gap+A_other)/127^20).
```

Here `B_i=ceil(1/(2*eta*sqrt(K_i/n_i)))`, and `B_(i+1)=1` after the last
committed level. `A_other` retains every relation, list, Bind, opening,
batching, and sumcheck allowance in J3. Probabilities are added before
conversion to bits. The query union and grinding model are unchanged.

The level-1 affine MCA and packed descent are **own proof, needs review**
(spec 12.3 and J3 above). J1's ordinary CA reduction alone does not suffice.
The tail uses Jo, [ePrint 2026/891](https://eprint.iacr.org/2026/891),
**Corollary 4.6, p. 9**, and **Theorem 4.7, pp. 10-11**, with the new scalar
input. P1-P5 remain open. Fiat-Shamir, hash binding, and quantum-security
losses are outside these totals.

Do not identify this calculation with the executable's historical
`johnson_bchks25_4.2_4.6_projection` or `section4=True` option. That option
uses `M=opened-1` at level 1 and the old rate convention. The new comparison
uses `M=1` only as the input to the own affine MCA proof, then pays for all
four conjugate codes. It does not change independent column sampling into
a power-curve sampler. It is not yet a named mode in the executable ledger.

The recorded input is one ECDSA signature, grid `144 x 576`, 556 committed
columns (252 opened), `e=20`, `lambda_bits=112`, `grind_bits=20`, and `tail_rate_inv=8`.
The arrays below run from level 1 through the last committed tail level:

```
K = [20736, 10368, 1296, 162]
n = [161280, 92160, 10752, 1344]
eta=1/16: s = [74, 70, 72, 72], B = [23, 24, 24, 24]
eta=1/8:  s = [88, 83, 85, 85], B = [12, 12, 12, 12]
```

| Recorded ECDSA analysis | Total queries | A_gap | A_other | Conditional bits |
|---|---:|---:|---:|---:|
| Main Hab25, eta=1/16 | 288 | 6307003635292025 | 46706513 | 87.28 |
| Linear MCA, eta=1/16 | 288 | 171668204677 | 46706513 | 102.36 |
| Linear MCA, eta=1/8 | 341 | 7775288102 | 12427584 | 106.01 |

All displayed conditional bits are rounded down. For linear MCA at
`eta=1/16`, the field terms alone give about 102.45 bits, and the query
terms alone give about 106.48 bits. Adding their error probabilities gives
102.36 bits. More queries
alone cannot raise this fixed E20 profile above its field-only bound. At `eta=1/8`, the field
terms alone give about 106.91 bits. Neither value is a limit on every possible
E20 parameter choice or on the protocol's actual security.

Across the recorded 16 CSP workload cases, the linear MCA calculation gives
102.36-105.91 bits at `eta=1/16` and 105.89-107.28 bits at `eta=1/8`.
These ranges do not cover other workloads. The second choice changes the
parameters and adds 53 queries for ECDSA; it is not part of the analysis-only improvement.

The linear ranges came from a separate research calculation. The executable
ledger does not print them. The recorded Hab25 row retains the executable's
continuous radius and upward Float64 pad; the linear rows use the integer
radius and upward integer rounding stated above.

Before using these results again, run `bench/bench_soundness.mojo` and compare
the compiled dimensions, columns, and all non-gap terms with this input.
Recompute after any change. The run prints the main Hab25 bound; the formula
above is a separate research calculation. Keep the conservative result beside
the linear projection until its theorem input and own transfers pass review.

### Section-4 curve projection

The `johnson_bchks25_4.2_4.6_projection` output applies
BCHKS25 Theorem 4.2/4.6 (pp. 27-29) as a power-curve bound with its own rate normalization: `M * a` at
the doubled `m`, with `M = opened - 1` at level 1 (251 for ECDSA; `_level1_symbol` batches the
columns only, the `gamma` weights are opening claims, not RS words) and `M = 1` for each pair-folding
tail challenge. It is a number, not a proof: it does not turn the uniform `beta` into a curve, and it
supplies none of the transfer, Bind, or query arguments of J1 to J3.

## Proof-to-code map and status

A [scoped packed and recursive list review](list-composition.md) checked the
abstract descent, Bind, opening batch, and backward extraction argument
against the observed challenge order. No missing PCS charge was found.
The basis identities remain a P1/P4 contract. Each relation charge needs a
bad event at its own challenge barrier and a separate coverage proof that
allows later adaptive Z/Z2/Q/Q3 choices. These remain P2/P3 contracts.
The review does not justify the grinding work reduction or cryptographic
compilation in P5.

| ID | Obligation | Evidence / remaining work |
|---|---|---|
| P1 | Packed level-1 columns admit the scalar RS guarantee in the block metric | Spec 12.1/12.3 and the machine-checked transfers are conditional. The current `_expand_beta` uses one independent weight per logical E column and physical weights `beta*b_t`; the checked fixed-raw-coordinate model needs an explicit grouped tensor/descent bridge. See [checkpoint](security-assurance.md) |
| P2 | Two-grid identities imply all intended row and chain-end constraints | `relations/statement.compile`, `proof.Shape`, `verifier._residual`, `_small_grid`, `_boundaries`, `_restrictions`. Review degree bounds and adaptive transcript composition, including the shared alpha and z |
| P3 | Integer/curve/hash constraints faithfully express the workloads | `workloads/sha256`, `keccak`, `poseidon`, `mulmod`, `ecdsa`. Check carries, selectors, canonical values, idle rows, hints, and the nonzero-polynomial argument for Horner/wiring; degree counting alone cannot establish it |
| P4 | The tail is the analyzed scalar tensor-fold protocol | `_Tail.level`, `_Tail.clear`, `pcs/tensor`, `pcs/tail`. Review row basis, mixed digits, four-coordinate first batch, last clear check, and per-round adaptivity |
| P5 | Cryptographic compilation preserves the required concrete security | `core/hash`, `core/transcript`, `proof.prefix_bytes`. Establish the exact Fiat-Shamir/hash model and losses, including a separate quantum claim if desired |
| J1 | The batched-column fold admits a dimension-free ordinary CA numerator | **Reduction checked: own proof, needs review.** BCIKS20 section 6.3, printed pp. 28-29, accepts BCHKS25 Theorem 1.5, p. 9, as its line input. The six checked steps of J1 include the linear-span step, monotonicity at the farthest radius, and list size `< Q`. MCA and the packed alphabet remain separate |
| J2 | Mutual (list) correlated agreement at the Johnson radius | **Source/transfer closed:** Hab25 Theorem 2, p. 4, is public (ePrint 2025/2110). Main ledger charges its quadratic numerator. Jo 2026/891 Corollary 4.6, p. 9, and Theorem 4.7, pp. 10-11, supply the interleaved/tensor transfer. Packed level 1 and adaptive running claims remain J3/P4 |
| J3 | The level-1 close case and adaptive list choice have explicit allowances | **Own proof, needs review.** Spec 12.3 gives packed descent, splits, Bind and backward list extraction. The ledger pays next-list query factors, list-scaled batching/sumcheck, `2B` openings, and `list_relations`. The original no-list-factor claim is not proved; P1-P5 contracts remain |
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
final query error and the four-coordinate first batch. The checks also cover
the Hab25 constant, integer overflow, list size, and Bind. Build with `--Werror`.
`run_tests.sh` executes this CPU-only benchmark and compiles the other benches.

The displayed totals and the upward gap charges were checked independently with 80-digit
arithmetic. These checks do not discharge the remaining protocol proof obligations.

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

Before any security claim, close P1-P5 and obtain independent cryptographic review.
The [scoped list review](list-composition.md) records the next algebraic and
relation contracts. The Hab25 baseline is field-limited and gives 87.28
conditional bits for ECDSA. The DKT26 comparison gives 106.41 with the same
parameters and is close to its query-only bound. Both use the ledger's
unproved grinding work model. Neither is a security certification.
The `security_bits: 112` metadata is a query target only.
