# zkML: plan and background

Goal: prove the inference of a quantized language model, sequence by sequence, on the same prover.
First target: Ternary Bonsai 2 27B (Qwen3.8 27B backbone, 64 blocks, ternary weights with FP16 scales per
group of 128, about 75% linear attention). It ships as a 6 to 7 GB local model, so the fixtures come from
the real model on the M1. Nothing here is implemented; this file is the pick-up point.

## Why the prover fits

- Quantized inference is integer arithmetic. The prover already has base-127 limbs, carry chains and
  accumulators along chains. Every field element is below 127, so a base-127 limb needs no range check.
- The prover commits cells and evaluates relations at GEMM throughput. The plan keeps the multiply-accumulate
  work native and commits only vectors.
- The weakness is the field size: an int8 dot product of length 768 reaches 24 bits, so a naive
  multiply-accumulate is several committed cells. The matmul relation below removes that cost.

## The matmul relation (Freivalds over E)

For C = A B with A (T x k), B (k x n), all integer matrices:

1. The prover commits A, B and C as base-127 limb columns and computes C natively on the GPU.
2. After the commit barrier the verifier samples r in E. The prover commits u = r^T A (k cells over E),
   an accumulator chain over the rows of A with the challenge powers as multiplier, the same shape as the
   Herder fingerprint. The relation checks u B = r^T C, a second chain. Both are degree 2.
3. Integer faithfulness: with A = sum_a 127^a A_a and B = sum_b 127^b B_b, the check runs per limb pair,
   C_(a,b) = A_a B_b, and one carry chain per output entry folds the pairs into the canonical limbs of C.
   This is the mulmod polynomial identity applied to matrices.
4. Soundness: one Schwartz-Zippel term, about n / 127^20, added to the ledger.

What it buys:

- Committed cells O(T k + k n + T n) instead of O(T k n). No product is ever committed.
- The prover's native work is one k x n pass over the weights per matmul, per proof, not per token. A
  1000-token prefill costs the same matmul checks as one token. The right unit of proof is a whole sequence.
- Private weights cost the same as public ones. Attention (Q K^T, the value product) uses the same relation.
- Public weights alone also work without this relation: constants are coefficients in a degree-1 relation
  row, so a block of 16 products is one row. This is the fallback, not the plan.

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

Each step unlocks the next. Steps 3 and 4 can run beside step 1.

1. **Herder.** Specified (`docs/milestone-3-lookup.md`, vault spec section 6), not built. Every
   nonlinearity and every requantize step is a lookup: softmax, GELU or SiLU, RMSNorm's rsqrt, the
   rescale-round-clip step, all 256-entry tables on int8 inputs. It also builds the post-commit challenge
   as a relation coefficient, which the matmul relation reuses. Shared dependency with the passport work.
2. **Matmul relation.** New relation kind in the statement builder, one ledger term. First milestone: an
   int8 matvec workload, 256 x 256, committed weights, measured in committed cells per MAC and seconds per
   MAC on the M1 against the published zkML provers. Prediction: well under one committed cell per MAC.
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
7. **Ledger and review.** Freivalds and Herder numerators in `docs/soundness.md`, then the same Opus and
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
