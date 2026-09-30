# Sketched level-1 code (experiment)

Bolt (Gurkan, Novakovic, Rothblum 2026) encodes a message `x` as `(x, RS(H x))`: `H` is a random sparse
matrix that shortens `x` by `alpha`, and only the sketch gets a Reed-Solomon code. This branch tries it as
the level-1 code. `src/pcs/sketch.mojo` holds all of it; the prover and the verifier do not call it.

## What is built

- `SketchMatrix`: the biregular ensemble on the host, column weight `c`, row weight `d`, coefficients
  uniform in `F*`, from a 64-bit seed.
- `k_sketch`: `u = H x` on the packed `F4` symbols, one thread per (sketch row, column), a CSR gather.
- `sketch_code`: the sketch, then `rs_encode_on` at rate 1/2 on its own domain. The first `K` rows of the
  codeword are the packed message in place (`e.packed` points at them), so the systematic part costs nothing.
- `tests/test_sketch.mojo`: the sampler's shape and seed, and the codeword against `H x` and direct
  evaluation of its RS code on every coset.
- `bench/bench_sketch.mojo`: A/B against `rs_encode` and its Merkle tree on the same packed data.

## Measured

M1 Pro, the W tree of each shape, ms per commit (`bench_sketch`, 5 repetitions, the two paths alternate).
`to_packed` (`idft2`, `to_stored`, `pack`) is common to both and does not change: 6, 61 and 195 ms.

| shape | columns | RS encode + Merkle | sketched, H 16/128 | sketched, H 8/64 | sketched, H 16/64 |
|---|---:|---:|---:|---:|---:|
| sha256, 32 x 2688 | 43 | 7.6 | 2.2 | 1.8 | 2.4 |
| rsa-2048, 144 x 2016 | 172 | 40.6 | 10.6 | 9.4 | 16.5 |
| passport, 144 x 4032 | 285 | 141.9 | 36.4 | 24.9 | 43.7 |

On the passport the sketch kernel is 23 ms at 16/128 and 12.5 ms at 8/64, the RS code of the sketch 5 ms,
the tree 7 ms. The Z and Q trees scale the same way. A prove saves about 8 percent: most of `encode W` is
`to_packed`, and the LDE and the residual do not change.

The kernel (`k_sketch`) reads `c` times the message, 2.6 GB at 16/128 on the passport, as rows in random
order. Probes on that shape: with `x` read from 64 hot rows it took 20 ms, with no `x` reads 11 ms, so
half of the first version (39 ms) was instructions. Four columns per thread (one entry decode per four
columns, 512 contiguous bytes per SIMD group and row) and the entries as aligned 16-byte groups brought it
to 23 ms; eight columns per thread crash the Metal shader compiler. Column tiles for cache reuse (16 to
96 columns) and rows sorted by column did not help.

## Distance and queries

Minimum distance of the `F127` ensemble by Bolt's `distance_analysis/q_ldpc.py` (the [YHC+11] Theorem 6.2
bound, q = 127), and the level-1 queries at 92 bits when the message part is tested at `gamma / 3`
(Bolt Lemma 5.1, unique decoding). The RS code now needs 74 to 104.

| H (c / d) | alpha | gamma | queries |
|---|---:|---:|---:|
| 16 / 128 | 1/8 | 0.0717 | 2638 |
| 8 / 64 | 1/8 | 0.0622 | 3046 |
| 16 / 64 | 1/4 | 0.1596 | 1167 |
| 16 / 32 | 1/2 | 0.3651 | 491 |

## Not built

The opening side: two query pools, the level-1 functional of a sketch position (`H^T` of a generator row,
dense, no tensor form), the verifier's dense part of `w~` through the tail, `k_frontier` past
`MAX_QUERIES = 1024`, the proof regions, and a ledger case. The soundness argument is own work and needs
review; the distance of a sampled `H` holds with high probability over the seed and cannot be checked.
