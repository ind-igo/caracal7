# caracal7-prover

Mojo implementation of **Caracal**, a uniform prover over F127 with Ligerito as the commitment layer. Target: the ethproofs client-side suite on an M1, proof under 500 KB, prover under 400 ms, more than 100 bits.

The spec lives in the notes vault: `wiki/projects/caracal7/specs/` (`sorted-copy-prover.md` is the main page, `statement-layer.md` the frontend, `polynomial-mulmod.md` the multiplication relation). This repo does not restate it. Implementation decisions go in `docs/decisions.md`.

## Layers

| layer | name | where |
|---|---|---|
| 4 | frontend | statement to IR program |
| 3 | relations | Herder (lookup, permutation, memory), bit certificates, mulmod, copy, public columns |
| 2 | Caracal IR and its check | families, accumulators, chains, small grid, quotients |
| 1 | Ligerito | Reed–Solomon columns in Blake3 Merkle trees, opened at P points |

## Build order

1. Ligerito on synthetic columns: F127 and tower E, two-pass encoder (L = 80,640), Blake3 tree with 1,024-byte leaves, commit, open, fold, tail, verifier, transcript.
2. Caracal IR: families, fused residual pass, Q1/Q2, Z tree, chain ends, small grid, Q3.
3. Herder memory instance.
4. Keccak-128 through the frontend.

## Layout

```
src/caracal7/   package, one module per layer piece
tests/          test_*.mojo, each a TestSuite runner
docs/           decisions.md
```

## Use

```
uv sync
./run_tests.sh
./run_tests.sh -D CARACAL_NVIDIA_MMA   # NVIDIA sm_80+: the integer tensor-core backend
./run_tests.sh -D CARACAL_APPLE_MMA    # Apple M5 (GPU family 10): the simdgroup MMA backend
```

`-D CARACAL_DFT_PROFILE` on a bench build synchronizes and prints every radix stage (radix, inputs, positions
per thread, threads, microseconds); the encoders, the LDE, the quotient and the tail encode all run on it.

On a rented NVIDIA box (Linux x86, driver installed, any sm_80 or later GPU), in this order:

```
uv sync
./run_tests.sh -D CARACAL_NVIDIA_MMA                       # the backend test first: a failure there is the fragment mapping
uv run mojo build --Werror -D CARACAL_NVIDIA_MMA -I src bench/bench_gemm.mojo -o bench_gemm && ./bench_gemm
uv run mojo build --Werror -D CARACAL_NVIDIA_MMA -I src bench/bench_sha256_chain.mojo -o bench_chain && ./bench_chain
```

Without a GPU the NVIDIA build still cross-compiles: `--target-accelerator sm_90` on any host.

Mojo only. No Python scaffolding. CPU first for correctness; GPU kernels later behind fixed buffer interfaces.

## csp-benchmarks results

The [soundness ledger](docs/soundness.md) and `bench/bench_soundness.mojo` track the conditional
security budget. Since 2026-09-13 the extension is `E = F_(127^20)` and the query target 112 per
level, and since 2026-09-14 the queries are sized at the Johnson radius (Ben-Sasson, Carmon, Haböck,
Kopparty, Saraf, STOC 2026, Theorem 1.5) with 20 bits of grinding on the query seeds: 107.7 to 110.2
conditional bits on every case (90.7 to 97.1 before 2026-09-13). The reported `security_bits: 112` is
that query target, not a verified total; the note records the bounds and the outstanding proof and
verifier obligations.

Recorded for reference; the harness and its caveats are in `docs/csp.md`. Method: the csp-benchmarks
Rust harness (`csp-rust/`, a thin crate over the Mojo prover's C ABI): Criterion times `prove` and `verify`
in-process on a prepared session (prover built, tables uploaded, arena filled), the same way the published
Rust systems are timed, so these are the like-for-like numbers and the ones to submit. Preprocessing is the
per-grid tables; peak memory is RSS, which excludes the Metal arena. Cells are committed witness cells. The
shell track (`caracal7/`, whole cold process per run through `benchmark.sh`) is kept for the harness's
non-Rust path; its numbers are about 150 ms higher on the small cases and 260 ms on ECDSA (process start,
Metal setup, kernel compile, arena fill) and are not tabulated here. Warm in-process GPU-only times are the
`warm prove` line of `bench/bench_<target>.mojo`.

**2026-09-14, Apple M1 Pro 16 GB, commit 9c09a5a (Criterion mean of 10 samples; `collect_benchmarks` fills the durations and the memory report)**

| target | input | prove ms | verify ms | proof bytes | preprocessing bytes | peak RSS MB | cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sha256 | 128 B | 51 | 20 | 177,400 | 77,424 | 46 | 308,224 |
| sha256 | 256 B | 59 | 29 | 187,860 | 72,956 | 48 | 462,336 |
| sha256 | 512 B | 70 | 35 | 198,536 | 149,980 | 49 | 924,672 |
| sha256 | 1024 B | 90 | 33 | 215,228 | 355,956 | 52 | 1,585,152 |
| sha256 | 2048 B | 160 | 49 | 215,900 | 651,796 | 56 | 3,698,688 |
| keccak | 128 B | 48 | 28 | 258,692 | 56,652 | 46 | 218,112 |
| keccak | 256 B | 47 | 29 | 267,288 | 61,972 | 48 | 436,224 |
| keccak | 512 B | 54 | 32 | 282,040 | 80,724 | 49 | 872,448 |
| keccak | 1024 B | 85 | 35 | 301,716 | 86,980 | 52 | 1,744,896 |
| keccak | 2048 B | 111 | 43 | 320,112 | 143,588 | 57 | 3,489,792 |
| poseidon | 2 | 133 | 40 | 372,141 | 143,588 | 53 | 1,916,928 |
| poseidon | 4 | 145 | 40 | 373,845 | 143,588 | 52 | 1,916,928 |
| poseidon | 8 | 138 | 43 | 373,653 | 143,588 | 52 | 1,916,928 |
| poseidon | 12 | 225 | 49 | 396,789 | 346,120 | 58 | 4,472,832 |
| poseidon | 16 | 223 | 49 | 394,069 | 346,120 | 59 | 4,472,832 |
| ecdsa | 1 sig | 425 | 119 | 543,620 | 553,684 | 92 | 19,574,784 |

Criterion's 10-sample runs flag high outliers on most rows (an iteration lands on the previous
session's arena being released); the 1024-byte SHA-256 row is one such run, its warm prove is 75 ms.

