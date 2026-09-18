# zkML: plan and background

Goal: prove the inference of a quantized language model, sequence by sequence, on the same prover.
First target: Ternary Bonsai 2 27B (Qwen3.8 27B backbone, 64 blocks, ternary weights with FP16 scales per
group of 128, about 75% linear attention). It ships as a 6 to 7 GB local model, so the fixtures come from
the real model on the M1. Nothing here is implemented; this file is the pick-up point.

## Why the prover fits

- Quantized inference is integer arithmetic. The prover has bit-level carry chains and accumulators
  along chains. A cell is below 127 without a range check, but radix 127 is zero in the field, so
  integer sums must be carried in bits.
- The prover commits cells and evaluates relations at GEMM throughput. The plan keeps the multiply-accumulate
  work native and commits only vectors.
- The weakness is the field size: an int8 dot product of length 768 reaches 24 bits, and the field sees
  each cell modulo 127, so every partial sum must be committed in small digits. The next section is the
  record of why no relation removes that cost.

## The matmul relation: why there is none

A matrix product is a global contraction over the inner index. Two ideas looked like they would keep
the multiply-accumulate work out of the proof: a Freivalds check on accumulator chains, and a sumcheck
over the inner slot digits ending in openings at multilinear points (the tail's own round kernels on
prover-held vectors). The design of the second is recorded in `docs/matmul.md`. Both reviews found the
same hole, and it is the field, not the argument.

Every check in the prover, quotient or sumcheck, is an identity in characteristic 127, and such an
identity sees an integer only modulo 127. The relation proves `C = A B mod 127`. One int8 by int8
product is already above 127, and a carry is a multiple of 127, which is zero in the field. So base-127
limbs have a free range and a useless radix: no field-linear family can recombine them. The existing
workloads are not affected, because mulmod, SHA-256 and Keccak accumulate in bits, where the radix is 2.

Integer accumulation in this field therefore needs every partial sum held in digits of a radix below 127
with an explicit range check per digit: bit cells, or lookups. That is O(T k n) committed cells with a
constant of about one to three cells per MAC for ternary weights and int8 activations, the naive cost the
plan set out to remove. A prover in a large field spends one field operation per MAC and never commits a
partial sum; dense matmul is that prover's ground, not this one's.

What still holds: the non-matmul parts of a block (norms, activations, requantize) are lookups and
bit-level relations where the byte field is at home; small models and matvec workloads can be proven at
the bit-level cost; and the lookup chain itself could become a fraction sum proven by the tail's round
kernels, which would remove the 16 chain columns per row at the price of committed inverses. None of
these makes a 27B model competitive. The plan below is kept as written for the record; the build order
is not started.

## Quantization formats

Products are never committed, so the weight format only sets the limb count and the carry width.

| Format | Limbs per weight | Note |
|---|---|---|
| binary, ternary | 1 | folded group scale adds 1 to 2 |
| int4 with group scales | 1 | folded scale adds 1 |
| int8 | 2 | |
| int16 accumulators | 3 | carry chain only |

Ternary saves about one limb per weight in the model commitment and nothing per proof. The line that
matters is integer versus float: bf16 or fp16 kernels round at every step in an order the vendor picks.
The proof is about our integer reference model (below), and the drift from the float run is measured,
not proven. A model with integer weights and int8 activations is the comfortable case: kernel and
reference can agree bit for bit.

Bonsai specifics: fix the FP16 group scales to an integer grid; fold the blockwise Hadamard rotation
(block 1024, fixed signs) into the committed weights, which makes them dense small integers; the 26M
high-precision parameters need their own integer grid.

What moves the per-token cost: activation width (committed limbs per token), accumulation width (carry
chains) and the number of lookups in the nonlinearities. Not the weights.

## Build order

Each step unlocks the next. Step 2 does not need Herder, so it goes first; steps 3 and the fixtures run
beside it.

1. **Herder.** Specified (`docs/milestone-3-lookup.md`, vault spec section 6), not built. Every
   nonlinearity and every requantize step is a lookup: softmax, GELU or SiLU, RMSNorm's rsqrt, the
   rescale-round-clip step, all 256-entry tables on int8 inputs. Shared dependency with the passport work.
2. **Matmul relation.** A design doc first (`docs/matmul.md`): matrix layout over columns and slot
   digits, the limb dimension, the round messages in the transcript, the prover kernel for the round
   polynomials, the verifier loop, the ledger term. Reviewed before code. Then the rounds in the prover,
   tail and verifier, and the builder hook. Build the slot-digit version directly; the column version is
   throwaway.
   First milestone: an int8 matvec workload, 256 x 256, committed weights, measured in committed cells per
   MAC and seconds per MAC on the M1 against the published zkML provers. Prediction: well under one
   committed cell per MAC.
3. **Integer reference model, host side.** Python or NumPy model of one Bonsai block with exact integer
   semantics: scales on an integer grid, Hadamard folded, int8 activations, every activation function as
   a fixed table. It defines what the circuit proves and measures accuracy retention before any relation
   is written.
4. **One transformer block as a workload.** Operator set: matmul, residual add, RMSNorm (sum of squares
   plus one rsqrt lookup), RoPE (public coefficients, linear), SwiGLU (SiLU lookup times product), softmax
   (exp lookup, sum, reciprocal lookup), the linear-attention recurrence (elementwise plus small per-token
   matmuls). Each is a family table. Measure one block before scaling.
5. **External model commitment.** Commit the weights once, root in the public inputs, openings in the
   proof. Today every column set is committed inside one proof; this is Ligerito plumbing, a second
   column source whose root the verifier binds instead of recomputing. Decide before the first zkML
   workload; the 27B model must not be re-encoded per inference.
6. **Segments.** Sixty-four blocks and a long sequence do not fit one grid. Vault spec section 11 has
   segments. Activations and the attention state cross segment boundaries as committed columns; a
   sequence proof is a chain of proofs with shared roots.
7. **Ledger and review.** Matmul rounds and Herder numerators in `docs/soundness.md`, then the same Opus and
   Codex review loop the RSA work got.

## Fixtures

A workload is one block, so the test vector is one block's weights and the hidden state entering and
leaving it for a short prompt. Bonsai fits the M1 (16 GB), so capture from the real model:

```
pip install mlx-lm
git clone https://github.com/PrismML-Eng/Bonsai-demo
```

Use their loader, not plain `mlx_lm` (the model card says the stock loader skips the activation
transform and the inverse embedding lookup). A short script loads the model, captures block 0 input and
output for a 32-token prompt, and saves the block weights (trits plus FP16 group scales, and the integer
grid version) and both hidden states as `.npy` under `tests/`. Capture first and free the model before
proving; model and prover do not fit in RAM together. Also capture one full-attention block.

The int8 matvec milestone needs only one projection matrix from block 0 and one token's activation.

Fallback if the loader is trouble: Qwen3 0.6B through `transformers`, same operator set except the
linear-attention recurrence.

## Use cases

A user proving a local run to themself is pointless. The value is proof to someone who was not there:

- Local data, remote verification: a public model on private input on the device, only the proof leaves
  (age from a face, a diagnosis from a scan). Same shape as the passport work; the strongest fit.
- Attested local agents: an on-device agent signs, bids or posts, and the counterparty gets proof that the
  committed policy model on the committed inputs produced that action.
- Edge inference markets: proof replaces a TEE as the honesty guarantee for a node that runs the model
  for others.
- Weaker: content provenance, benchmark integrity, proof that a safety filter ran.

## References

- Bonsai 2 27B announcement: https://prismml.com/news/bonsai-2-27b
- Model card: https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
- Whitepaper: https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf
