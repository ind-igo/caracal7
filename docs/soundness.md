# Soundness ledger for the CSP configuration

Status: **conditional analysis, not certification**. Reviewed against prover commit `8f94cc7` on
2026-09-12; parameters updated 2026-09-13 (`E = F_(127^20)`, query target 112, decisions.md). The
executable ledger compiles the current source; the conclusions below require the proof obligations
and implementation gaps in this note to be closed. A successful calculation or test run does not
establish a cryptographic security level.

The benchmark metadata says `security_bits: 112`. In `core/params.mojo`, 112 is the per-level
**query target**, met as 92 bits of queries plus 20 bits of grinding on every query seed (below), with
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
- `capacity_conjecture ...` and `johnson_bchks25_1.5 ...` are projections of the other two regimes at the
  same geometry: the queries each level would need at the same per-level target, the query bits at those
  queries, the `pcs_gap` numerator of that regime, and the `conditional_iop_bits` the switch would compile.
  `Profile.regime` selects the regime; `CLIENT` stays at `REGIME_UNIQUE`.
  - `REGIME_CAPACITY` (radius `1 - rate - eta`, miss `rate + eta`) is the unproven up-to-capacity
    conjecture. It has no proven field term; the projection keeps the unique `pcs_gap` as a placeholder,
    so its `conditional_iop_bits` is the query term only. Nothing in this note certifies it.
  - `REGIME_JOHNSON` (radius `1 - sqrt(rate) - eta`, miss `sqrt(rate) + eta`) charges the correlated
    agreement error of BCHKS25 Theorem 1.5 ([[raw/papers/proximity-gaps-rs-codes]], any domain, any field):
    `a / |E|` per code with `m = max(ceil(sqrt(rho) / (2 eta)), 3)` and
    `a = (2 (m + 1/2)^5 + 3 (m + 1/2) gamma rho) / (3 rho^1.5) * n + (m + 1/2) / sqrt(rho)`, linear in `n`
    where BCIKS20 1.2 had `n^2`. The theorem is stated for pairs (lines); the ledger applies it in the
    same places as the unique term (level 1 uniform fold, three tensor elements per tail level). The
    affine-space (batched columns) form, the mutual form (Haböck 2025, cited there), and the list-regime
    rewrite of the level 1 close case (spec 12.2) are open obligations before this regime is proven for
    caracal7. The constant grows as `rho^-1.5`, so the low-rate tail levels dominate the term.
- `conditional_iop_bits` is `-log2(query_error + sum(A)/127^e)` at the compiled `e`, rounded down to
  two decimals.
- `projected_e16_...` and `projected_e20_...` change only the denominator to `127^16` or `127^20` at
  the same geometry and queries. They are not performance estimates or security certifications.

Calculations use Float64 for diagnostic sizing, not interval arithmetic or a machine-checked proof.
Exit zero means the calculation succeeded. The program always reports unresolved obligations and
never emits `security_bits` or a certification pass. Fiat-Shamir, hash binding, quantum security,
and the gaps below are not assigned zero error; they are **outside this conditional total**.

### Current results

These are diagnostic bounds under the assumptions below, not measured attack costs or certified
security levels. Poseidon sizes sharing a grid have identical ledgers.

| Workload | Input | Grid | Conditional IOP bits, e = 20, lambda' 112 | e = 16, lambda' 103 (before 2026-09-13) |
|---|---:|---:|---:|---:|
| SHA-256 | 128 B | 32 x 224 | 110.46 | 95.00 |
| SHA-256 | 256 B | 32 x 336 | 110.70 | 94.51 |
| SHA-256 | 512 B | 32 x 672 | 110.46 | 93.44 |
| SHA-256 | 1024 B | 32 x 1152 | 110.26 | 92.64 |
| SHA-256 | 2048 B | 32 x 2688 | 110.21 | 90.71 |
| Keccak | 128 B | 64 x 24 | 110.70 | 97.08 |
| Keccak | 256 B | 64 x 48 | 110.54 | 96.11 |
| Keccak | 512 B | 64 x 96 | 110.54 | 95.17 |
| Keccak | 1024 B | 64 x 192 | 110.26 | 94.20 |
| Keccak | 2048 B | 64 x 384 | 110.15 | 93.20 |
| Poseidon | 2, 4, 8 elements | 64 x 384 | 110.15 | 93.20 |
| Poseidon | 12, 16 elements | 64 x 896 | 110.11 | 91.66 |
| ECDSA | 1 signature | 144 x 576 | 110.58 | 90.69 |

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

**Johnson projection (e = 20, eta = 1/16, 2026-09-14).** ECDSA: 74/39/45/45 queries (203 against 401) at
101.45 conditional bits; SHA-256 2048: 76/39/45/45/45 at 101.52; Poseidon 12: 88/42/44/45/45 at 103.07;
the small cases 105 to 107. The field term caps the regime near 101 bits because the tail levels sit at
rates near 1/60, where the `rho^-1.5` constant is large; holding 112 would take e = 22, a higher tail
rate rule for this regime, or a per-level `eta`. Not adopted: the three obligations above are open.

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
| `wire_fingerprints` | maximum challenge degree of an endpoint used as a slot/public factor | Selected unequal underlying values must have unequal fingerprints except on a root event |
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
