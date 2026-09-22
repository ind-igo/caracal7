# Security assurance checkpoint

Status: **conditional analysis, not a security certification**. This checkpoint
connects the numerical proof work to the current verifier. No protocol parameter,
proof format, query count, or runtime algorithm changes here. Own derivations
and the machine-checked certificates below (written in Bend, a proof language, in the separate
`caracal7-fv` repository) are **own proof, needs review**.

The checked range is 20 main cases at E=F_(127^20), plus four
ECDSA tail-rate comparisons. The 20 cases give 87.50-88.37 conditional
interactive bits, or 107.12-108.18 under the separate grinding work calculation.
ECDSA gives 88.21 and 108.03. A 100-bit guarantee for the implementation is not established.
Unproved events and cryptographic losses are not assigned zero error.

## What the two numbers mean

Let Q=127^20, A be the sum of the field numerators, P be the query error per
attempt with every required list factor, and g=20.

```
interactive diagnostic = -log2(P + A/Q)
work diagnostic        = -log2(P/2^g + A/Q)
```

The first uses fresh ideal independent challenges and binding commitments,
conditional on P1-P4. The second also needs an adversarial grinding and
Fiat-Shamir model under P5. The old `conditional_iop_bits` output used the
second formula. Its new name is `conditional_work_bits`. This correction
does not show a new attack or change the verifier.

The work discount is not a proved per-hash success bound. A fresh ideal
nonce probe passes with probability 2^-g, but expected waiting time is not a
deterministic cost for each successful attempt. A concrete bound must count
all oracle work, failed probes, prefix reuse, adaptive retries, state collisions,
and commitment binding. Quantum security needs a separate statement.

`core/transcript.k_grind` casts probes to UInt32 and writes a zero high nonce
word. A fixed state has only 2^32 distinct trials. It can exhaust them and
repeat. Under independent ideal probes, the probability that none passes is
(1-2^-g)^(2^32), which is positive. The earlier probability-one termination
claim was incorrect. This is a termination issue, not a demonstrated forgery.
The runtime algorithm is unchanged; a bounded-search/reseed design remains open.

## Smaller list allowance

The scalar input stays DKT26 **Theorem 5.12, p. 53**, with the numerator from
**Lemma 5.3, pp. 40-43**; see [the source calculation](soundness.md#dkt26-johnson-mca-comparison).
For each actual compiled code, K is dimension, D=K-1 is degree, and N is length.
Use exact integers:

```
T = ceil(sqrt(N*K) + N/16)
delta = T*T - N*D > 0
L = ceil(N*(T-D)/delta)
```

The Bend incidence and RS block-list proofs imply that L is a sufficient
list cap for agreement at least T. This applies to full blocks and their
joint ambient lists, for arbitrary block width and supplied distinct domains.
It is not the scalar list bound substituted into a packed protocol. The
certificates check the upper inequality and failure of the preceding integer
for this sufficient ratio. They do not assert the true list has size L, or
that L is an optimal combinatorial bound.

`cspcerts.bend` in the `caracal7-fv` repository checks 53 distinct code tuples for the 20 cases.
It supplies the existing tail/packed numeric interfaces; actual domains still
need exact lengths and distinctness. The main caps are 4-7. ECDSA changes
from [23,24,24,24] to [7,7,7,7]. All next-list query factors, current-list
batch/sumcheck factors, Bind pairs, opening batches, and relation unions use
the corresponding cap. The four-component root MCA charge stays 4(C+1),
and each scalar tail stays 3C. Older Hab25 and DKT26 old-list outputs remain.

The new `johnson_dkt26_joint_list_conditional` output is a conditional
comparison. Its use for the implemented prover still requires the grouped
column connection below. Exact integer certificates do not certify Float64
probabilities or an end-to-end soundness bound.

## Covered cases

The benchmark compiles real statements and reads their actual Params, Shape,
descriptors, and tail schedule. No copied shape-count table supplies the
ledger. Run `uv run mojo run --Werror -I src bench/bench_soundness.mojo`.
The numbers below round down to two decimals.

| Workload | Input | Grid | DKT old-list work | Joint-list interactive | Joint-list work |
|---|---|---|---:|---:|---:|
| sha256 | 128 B | 32 x 224 | 106.31 | 88.13 | 108.12 |
| keccak | 128 B | 64 x 24 | 106.15 | 88.18 | 108.18 |
| sha256 | 256 B | 32 x 336 | 106.14 | 87.95 | 107.93 |
| keccak | 256 B | 64 x 48 | 106.20 | 88.16 | 108.15 |
| sha256 | 512 B | 32 x 672 | 106.31 | 88.13 | 108.10 |
| keccak | 512 B | 64 x 96 | 106.20 | 88.01 | 108.00 |
| sha256 | 1024 B | 32 x 1152 | 105.77 | 87.64 | 107.59 |
| keccak | 1024 B | 64 x 192 | 105.78 | 87.76 | 107.74 |
| sha256 | 2048 B | 32 x 2688 | 106.03 | 87.88 | 107.74 |
| keccak | 2048 B | 64 x 384 | 105.78 | 87.70 | 107.66 |
| poseidon | 2 elements | 64 x 384 | 105.78 | 87.70 | 107.66 |
| poseidon | 4 elements | 64 x 384 | 105.78 | 87.70 | 107.66 |
| poseidon | 8 elements | 64 x 384 | 105.78 | 87.70 | 107.66 |
| poseidon | 12 elements | 64 x 896 | 105.67 | 87.50 | 107.41 |
| poseidon | 16 elements | 64 x 896 | 105.67 | 87.50 | 107.41 |
| ecdsa | 1 signature | 144 x 576 | 106.41 | 88.21 | 108.03 |
| rsa2048 | benchmark fixture | 144 x 2016 | 106.45 | 88.37 | 107.83 |
| sod | benchmark fixture | 144 x 2688 | 105.91 | 88.18 | 107.12 |
| dsc | benchmark fixture | 144 x 2688 | 105.92 | 88.18 | 107.12 |
| passport | benchmark fixture | 144 x 4032 | 106.28 | 88.37 | 107.44 |

The first 16 cases cover the 12 grid geometries of the current CLI routes.
Intermediate admitted hash sizes and Poseidon sizes use those same grids;
their compiled statements depend only on the parameters. The four added
cases cover the primary RSA, SOD, DSC, and passport benchmark metadata.
They share three additional grids. These are statements for the named
fixtures, not all possible document lengths or all workload parameters.

This is not coverage of all possible prover configurations. Extended SHA
sizes, SHA chains, standalone mulmod scaling, and lookup benchmarks need
separate profile checks. The ledger rejects split codewords, other tail
arities, Herder accumulators, and unhandled challenge-dependent ordinary
families. The four rate-sweep reports are comparisons outside the main
20-case minimum. Changing parameters or routes requires a fresh run and
certificate comparison.

## What still blocks the implemented claim

| Obligation | Evidence and next action |
|---|---|
| P1/P4: compact E columns | `Shape.opened` and `verifier._expand_beta` sample one weight per logical column. For an E column, physical weights are beta*b_t. The existing Bend packed model counts independent weights on its input words and requires fixed raw coordinates. Prove a grouped tensor/basis map, its descent, and joint-list extraction for this actual sampling space. Do this before claiming the current packed proof applies. |
| P1/P4: layout and folds | Connect `pcs/encode.pack_slot`, `join_index`, the ordered domains in `core/tables`, `verifier.encode_at`, and the tensor row/evaluation maps to the checked abstract identities. Preserve all four coordinates and the split rule. |
| P2/P3: relations | Prove that false semantic claims give nonzero polynomials fixed at each challenge barrier, then that the checked equations imply the intended workload. Cover adaptive W, Z/Z2, and Q/Q3 choices. Degree arithmetic alone is insufficient. |
| P5: cryptographic compilation | Specify adversary resources and the exact transcript/hash model. Prove grinding, Fiat-Shamir, commitment-binding, and state-collision losses. Observed challenge order and unbiased rejection sampling do not supply this reduction. |
| I2/I3: implementation and coverage | Keep adversarial parser/arithmetic checks and compare each supported profile to the proof premises. Code review and tests support correspondence; Bend does not prove Mojo or its compiler. |

The compact-column issue is a missing proof connection, not a demonstrated
attack. One cannot apply an independent-physical-weight count to the restricted
weights beta*b_t. Grouping into logical columns instead makes the raw values
E-valued, so the current raw-fixation premises no longer follow automatically.
Retain sum_t b_t tensor X_t in E tensor_F F4: independence over F does not
imply independence over F4. `_level1_symbol` does retain all four coordinates;
no scalar collapse was observed in that code.

The next useful proof is this grouped tensor and descent connection. Reuse
the existing scalar, interleaved, affine, and tensor results. Then connect
individual openings and the earlier relation barriers. Continue the concrete
field/domain construction and sampler work where those consumers require it.
