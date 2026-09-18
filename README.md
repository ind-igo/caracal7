# Caracal7

Mojo implementation of **Caracal7**, a uniform, GPU native prover over F127 with Ligerito as the commitment layer. Target: the ethproofs client-side suite on an M1, proof under 500 KB, prover in the hundreds of milliseconds, more than 100 bits of security. Everything in the prover runs on the GPU (Apple Metal or NVIDIA); the verifier runs on the host.

The idea: a 7-bit field does not make streaming work faster, but it makes matrix units reachable. Witness values are single bytes, so the encoder and the residual pass become integer or fp16 GEMMs on tensor cores and simdgroup units, a path that is closed to 31-bit fields. The design restructures the prover until GEMMs dominate. The full spec lives in a notes vault outside this repo; implementation decisions are dated in `docs/decisions.md`, and each workload has a design doc in `docs/`.

## The prover, component by component

**Fields.** Data lives in `F = F127`. The polynomial domain is `F2 = F127^2` (`i^2 = -1`), the code alphabet is `F4 = F127^4`, and every Fiat-Shamir challenge, accumulator and quotient value lives in `E = F_127^20`, a tower `F4[u] / (u^5 - g)`. Elements are stored as 1, 2, 4 or 20 bytes, one coordinate per byte. Since 2026-09-13 `E` has 20 coordinates for about 116 bits of field soundness.

**Grid and chains.** A trace is a bivariate product grid `H1 x H2` of multiplicative subgroups of `F2*` (order 16128). A chain is one row of `H2`: `h1` grid points in linear order, and a statement is a list of chains. The residual grid `G = G1 x G2` doubles each axis, so every degree-2 relation is checked exactly. Grids are legal when each axis order is `2^a * m` with `a` in 2..7 and `m` dividing 63. All benchmarks below use one grid per statement, chosen at compile time.

**The IR (layer 2).** A column is a function on the grid, one byte per point. A family is a degree-2 relation between columns (with cyclic reads at fixed offsets) that must vanish on the grid; its residual is what is left after moving every term to one side. An accumulator is an `E`-valued column with a recurrence along a chain: grand products for wiring and Horner sums for fingerprints, whose chain-end values land on a small grid. All residuals are batched with one challenge and divided by the vanishing polynomials into three quotients, `Q1` and `Q2` on the grid and `Q3` on the small grid, all checked at the opening points. A wrap residual on the small grid ties chain ends to chain starts.

**Relations (layer 3).** Written in the IR, no new protocol: bit certificates (every column that must be a bit, and the fixed bitwise functions of SHA-256 and Keccak); polynomial-identity multiplication mod a 256-bit prime, where `a b = r` is checked as a polynomial identity at a challenge point and no product is ever committed (`docs/mulmod.md`); the copy constraint as a wiring grand product with public factors; public columns of closed form that the verifier evaluates at a point instead of receiving; row groups with masks and selectors, so several statements share one grid (`docs/statement-builder.md`). The lookup argument (Herder) is planned and not built; measured on SHA-256, a lookup channel costs more than the bit columns it would replace.

**Commitment layer (layer 1).** Ligerito as published, with a subfield first round. Every committed column is a Reed-Solomon codeword over `F4` on a domain of one to four cosets of a subgroup of `F4*` (order dividing 161280), rate at most 1/16, and the codeword rows sit in Blake3 Merkle trees with 1024-byte leaves. Three trees: the witness, the accumulators, the quotients. After the opening challenge the folds are not sent; each fold is committed as the next level, and the tail is Ligerito's batched partial sumcheck, three binary digits per level, with the last level in the clear. The verifier opens query rows per level and checks them against the fold. The query count is sized at the Johnson radius (BCHKS25, Theorem 1.5) with 20 bits of grinding on the query seeds, for a per-level target of 112 bits.

**Transcript and verifier.** Blake3 over the proof bytes, computed on the device in the prover, so the host never reads a challenge. The verifier is a separate host program (`verifier.mojo`): it rebuilds the transcript, evaluates the public columns and the statement's public data, checks the quotient identity at the opening points, and walks the Ligerito levels. A proof verifies in tens of milliseconds for the hash workloads; the RSA and passport verifies are dominated by deriving their public data, which the next step moves to a product form.

**Soundness status.** Conditional analysis, not certification. `docs/soundness.md` is the ledger: the commitment-layer bound, the relation numerators, the proof-to-code map, and the open obligations of the Johnson regime. `bench/bench_soundness.mojo` prints the budget per case: 107.7 to 110.2 conditional bits on every csp-benchmarks case. The reported `security_bits: 112` is the query target, not a verified total.

## Statements

Every statement implements the `Workload` trait in `workload.mojo` (public inputs, trace, public data) and is proved by `prove_workload`. Each has a test under `tests/` and a bench under `bench/`.

| workload | what it proves | doc |
|---|---|---|
| `sha256` | SHA-256 of 128 to 2048 bytes, one round per chain, every column a bit | `docs/sha256.md` |
| `keccak` | Keccak-256 of 128 to 2048 bytes, 142 bit columns | `docs/keccak.md` |
| `poseidon` | Poseidon over Mersenne-31, width 16, hashing 2 to 16 field elements | `docs/poseidon.md` |
| `ecdsa` | one secp256k1 signature verification on the mulmod chains | `docs/ecdsa.md` |
| `rsa` | one RSA-2048 signature verification (e = 65537, 17 modmuls), squaring symmetry | `docs/decisions.md` |
| `sha256g` | SHA-256 as a row group on the 144-row chain, digests wired between groups | `docs/decisions.md` |
| `sod` | a passport SOD: SHA(DG1), SHA(security object), SHA(signed attributes), RSA verify, one proof | `docs/passport.md` |

## Run

Requirements: Mojo 1.0 through MAX 26.5 or later, installed by `uv sync`; a Mac with Metal (any M-series) or a Linux box with an NVIDIA GPU of sm_80 or later. There is no CPU prover: each kernel's test compares against a few lines of scalar Mojo.

```
uv sync
./run_tests.sh                          # every tests/test_*.mojo, then every bench builds; JOBS=4 by default
uv run mojo run --Werror -I src tests/test_sha256.mojo
uv run mojo run --Werror -I src bench/bench_sha256.mojo   # prints setup, warm prove, verify, proof bytes
```

Backends are chosen at compile time. Without a flag the prover uses the lanes backend, which runs on any GPU and is the M1 path of every number below.

```
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

The csp-benchmarks harness is `csp-rust/`, a Criterion crate over the prover's C ABI; `docs/csp.md` says how to run it and what its numbers mean.

## Benchmarks

### RSA-2048 and the passport SOD

M1 Pro, warm prove, proof verified, 2026-09-18 (`bench/bench_rsa.mojo`, `bench/bench_sod.mojo`, grid 144 x 2016):

| statement | chains | prove ms | verify ms | proof bytes |
| --- | ---: | ---: | ---: | ---: |
| RSA-2048 verify | 1888 | 586 | 1100 | 627,000 |
| passport SOD (3 x SHA-256 + RSA-2048) | 1987 | 1724 | 2100 | 1,030,000 |

About half of each verify is public-data derivation. The next steps are in `docs/passport.md`.

### SHA-256 hash chain on NVIDIA

RTX 3090 (Vast.ai, 2026-09-16, `bench_sha256_chain`, warm prove median, proof verified),
against Jolt's 0.33 ms per hash on an M5 Max:

| grid | hashes | prove ms | ms per hash |
| --- | ---: | ---: | ---: |
| 32 x 8064 | 125 | 35.3 | 0.28 |
| 96 x 8064 | 375 | 88.0 | 0.23 |
| 288 x 2688 | 369 | 89.4 | 0.24 |
| 224 x 8064 | 875 | 195.3 | 0.22 |
| 672 x 2688 | 861 | 212.7 | 0.25 |

The M1 Pro proves the 125-hash grid in about 215 ms (1.7 ms per hash).

### csp-benchmarks results

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

**2026-09-16, Apple M1 Pro 16 GB, commit 335ceca plus the 8x8 simdgroup open kernel (Criterion mean of 10 samples; `collect_benchmarks` fills the durations and the memory report; the memory pass is the 2026-09-14 one; load average 10 to 12 during the run)**

| target | input | prove ms | verify ms | proof bytes | preprocessing bytes | peak RSS MB | cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sha256 | 128 B | 47 | 22 | 174,460 | 77,424 | 47 | 308,224 |
| sha256 | 256 B | 51 | 29 | 187,600 | 72,956 | 48 | 462,336 |
| sha256 | 512 B | 60 | 35 | 199,084 | 149,980 | 50 | 924,672 |
| sha256 | 1024 B | 87 | 34 | 216,924 | 355,956 | 52 | 1,585,152 |
| sha256 | 2048 B | 145 | 53 | 213,628 | 651,796 | 57 | 3,698,688 |
| keccak | 128 B | 51 | 28 | 257,028 | 56,652 | 47 | 218,112 |
| keccak | 256 B | 58 | 29 | 269,024 | 61,972 | 48 | 436,224 |
| keccak | 512 B | 60 | 33 | 283,656 | 80,724 | 49 | 872,448 |
| keccak | 1024 B | 91 | 35 | 301,492 | 86,980 | 51 | 1,744,896 |
| keccak | 2048 B | 100 | 43 | 319,120 | 143,588 | 56 | 3,489,792 |
| poseidon | 2 | 124 | 40 | 372,373 | 143,588 | 52 | 1,916,928 |
| poseidon | 4 | 134 | 41 | 373,365 | 143,588 | 52 | 1,916,928 |
| poseidon | 8 | 123 | 40 | 371,061 | 143,588 | 52 | 1,916,928 |
| poseidon | 12 | 197 | 49 | 396,245 | 346,120 | 59 | 4,472,832 |
| poseidon | 16 | 202 | 49 | 394,005 | 346,120 | 59 | 4,472,832 |
| ecdsa | 1 sig | 407 | 122 | 543,972 | 553,684 | 93 | 19,574,784 |

Criterion's 10-sample runs flag high outliers on most rows (an iteration lands on the previous
session's arena being released); the small rows move a few ms between runs on a loaded machine.
