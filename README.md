# Caracal7

Mojo implementation of **Caracal7**, a uniform, GPU native prover over F127 with Ligerito as the commitment layer. Target: the ethproofs client-side suite on an M1, proof under 500 KB, prover in the hundreds of milliseconds, more than 100 bits of security. Everything in the prover runs on the GPU (Apple Metal or NVIDIA); the verifier runs on the host.

The idea: a 7-bit field does not make streaming work faster, but it makes matrix units reachable. Witness values are single bytes, so the encoder and the residual pass become integer or fp16 GEMMs on tensor cores and simdgroup units, a path that is closed to 31-bit fields. The design restructures the prover until GEMMs dominate. The full spec lives in a notes vault outside this repo; implementation decisions are dated in `docs/decisions.md`, and each workload has a design doc in `docs/`.

## Constructions

Every relation is compiled to degree-2 constraints on a two-dimensional grid and checked by one quotient argument; there is no sumcheck over the trace and no custom gates. Relations are defined as tables: a family is a row of a table that names the columns it reads, the offsets it reads them at and the coefficients, and a workload is a list of such rows plus a column-major trace. Copy constraints and fingerprints are grand-product and Horner accumulators along chains. The lookup argument, Herder, is a new grand-product construction in the plookup family, specified and not implemented (see below); the hash workloads use bit certificates instead. Wide multiplication mod a prime is a polynomial identity checked at a challenge point, with no product column. The commitment layer is Ligerito: Reed-Solomon columns in Blake3 Merkle trees, a batched partial sumcheck as the tail, queries sized at the Johnson radius with grinding. Blake3 is also the Fiat-Shamir transcript, run on the device. The challenge field is a degree-20 extension of F127.

### Herder lookups (TODO)

Herder is the lookup, permutation and read-write memory argument. The prover commits the lookup records and a copy of them ordered by table position. One grand-product identity over adjacent pairs of the copy proves that the copy walks the public table in order and that it is a permutation of the records, so membership needs no separate permutation argument. The table enters as one constant the verifier computes, not as witness rows. Records are fingerprinted by a fixed injective linear map into the challenge field, with no compression challenge, and repetitions are exponents of factors, not field counters, so the argument holds in characteristic 127. Ordering the copy by address and timestamp gives memory.

## The prover, component by component

**Fields.** Data lives in `F = F127`. The polynomial domain is `F2 = F127^2` (`i^2 = -1`), the code alphabet is `F4 = F127^4`, and every Fiat-Shamir challenge, accumulator and quotient value lives in `E = F_127^20`, a tower `F4[u] / (u^5 - g)`. Elements are stored as 1, 2, 4 or 20 bytes, one coordinate per byte. The field contribution depends on the code lengths and the theorem used; see the soundness ledger.

**Grid and chains.** A trace is a bivariate product grid `H1 x H2` of multiplicative subgroups of `F2*` (order 16128). A chain is one row of `H2`: `h1` grid points in linear order, and a statement is a list of chains. The residual grid `G = G1 x G2` doubles each axis, so every degree-2 relation is checked exactly. Grids are legal when each axis order is `2^a * m` with `a` in 2..7 and `m` dividing 63. All benchmarks below use one grid per statement, chosen at compile time.

**The IR (layer 2).** A column is a function on the grid, one byte per point. A family is a degree-2 relation between columns (with cyclic reads at fixed offsets) that must vanish on the grid; its residual is what is left after moving every term to one side. An accumulator is an `E`-valued column with a recurrence along a chain: grand products for wiring and Horner sums for fingerprints, whose chain-end values land on a small grid. All residuals are batched with one challenge and divided by the vanishing polynomials into three quotients, `Q1` and `Q2` on the grid and `Q3` on the small grid, all checked at the opening points. A wrap residual on the small grid ties chain ends to chain starts.

**Relations (layer 3).** Written in the IR, no new protocol: bit certificates (every column that must be a bit, and the fixed bitwise functions of SHA-256 and Keccak); polynomial-identity multiplication mod a 256-bit prime, where `a b = r` is checked as a polynomial identity at a challenge point and no product is ever committed (`docs/mulmod.md`); the copy constraint as a wiring grand product with public factors; public columns of closed form that the verifier evaluates at a point instead of receiving; row groups with masks and selectors, so several statements share one grid (`docs/statement-builder.md`). Herder, the lookup, permutation and memory argument described above, is planned and not built; measured on SHA-256, a lookup channel costs more than the bit columns it would replace.

**Commitment layer (layer 1).** Ligerito as published, with a subfield first round. Every committed column is a Reed-Solomon codeword over `F4` on a domain of one to four cosets of a subgroup of `F4*` (order dividing 161280), rate at most 1/16, and the codeword rows sit in Blake3 Merkle trees with 1024-byte leaves. Three trees: the witness, the accumulators, the quotients. After the opening challenge the folds are not sent; each fold is committed as the next level, and the tail is Ligerito's batched partial sumcheck, three binary digits per level, with the last level in the clear. The verifier opens query rows per level and checks them against the fold. The query count is sized at the Johnson radius (BCHKS25, Theorem 1.5, p. 9) with 20 bits of grinding on the query seeds, for a per-level target of 112 bits.

**Transcript and verifier.** Blake3 over the proof bytes, computed on the device in the prover, so the host never reads a challenge. The verifier is a separate host program (`verifier.mojo`): it rebuilds the transcript, evaluates the public columns and the statement's public data, checks the quotient identity at the opening points, and walks the Ligerito levels. A proof verifies in tens of milliseconds for the hash workloads and in a few hundred milliseconds for the RSA and passport statements, most of it the clear vector and the public data.

**Soundness status.** Conditional analysis, not certification. `docs/soundness.md` is the ledger: the commitment-layer bound, the relation numerators, the proof-to-code map, and the open obligations of the Johnson regime. `bench/bench_soundness.mojo` prints the budget per case: 87.28 to 99.52 conditional bits on every csp-benchmarks case. The reported `security_bits: 112` is the query target, not a verified total.

The [E20 linear MCA research comparison](docs/soundness.md#e20-linear-mca-research-comparison)
gives 102.36 conditional bits for the same ECDSA protocol and parameters, subject
to the stated proof conditions. The executable ledger still charges Hab25.

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
| `sod` | a passport SOD: SHA(DG1), SHA(security object), SHA(signed attributes), RSA verify with the DSC key committed, a disclosed MRZ window and a nullifier, one proof | `docs/passport.md` |
| `mrz` | the verifier's predicates on the disclosed window: nationality, birth, expiry, age | `docs/passport.md` |
| `dsc` | the DSC certificate check: SHA(certificate body), RSA verify by the CSCA key, the committed DSC key inside the body | `docs/passport.md` |
| `csca` | the verifier's side of a passport: the CSCA registry (a list of key ids and exponents), the PKCS#1 padding limbs, one commitment in both proofs | `docs/passport.md` |
| `passport` | the whole passport in one proof: the SOD groups, the certificate body and two RSA verifies on one grid, the DSC key a wire, no commitment | `docs/passport.md` |

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

### RSA-2048 and the passport

M1 Pro, warm prove, proof verified (`bench/bench_rsa.mojo` on 144 x 2016, `bench/bench_sod.mojo` and
`bench/bench_dsc.mojo` on 144 x 2688, `bench/bench_passport.mojo` on 144 x 4032):

| statement | chains | prove ms | verify ms | proof bytes |
| --- | ---: | ---: | ---: | ---: |
| RSA-2048 verify | 1745 | 593 | 300 | 642,000 |
| passport SOD (3 x SHA-256 + commitment + nullifier + RSA-2048, s and n witness) | 1958 | 1750 | 250 | 1,123,000 |
| DSC certificate check (SHA-256 + commitment + RSA-2048, s witness) | 2003 | 1390 | 410 | 1,064,000 |
| passport in one proof (4 x SHA-256 + nullifier + 2 x RSA-2048, the DSC key a wire) | 3799 | 2520 | 650 | 1,355,000 |

A passport is one proof: the SOD's hashes, the certificate body's hash and both RSA verifies on one grid, the
DSC key wired from the body to the SOD's verify, no commitment; it discloses the MRZ fields and a nullifier
per scope. The two-proof form (the SOD and the DSC check sharing a commitment digest, 3.1 s prove, 2.2 MB,
0.66 s verify) stays for grids that hold no more than 2688 chains. The verify is host-side field arithmetic
over the proof; its public data is under 1 MB (`docs/public-columns.md`). zkPassport's Noir circuits for the same passport,
proved with Barretenberg on the same machine, take 45 s (six Honk subproofs of about 0.9 s each and a 40 s
recursive outer proof) for a 14.7 KB proof that verifies in 0.09 s and is zero-knowledge; the recipe and the
per-circuit numbers are in `bench/zkpassport/README.md`.
The next steps are in `docs/passport.md`.

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
security budget. The extension is `E = F_(127^20)` and the query target 112 per
level; the queries are sized at the Johnson radius (Ben-Sasson, Carmon, Haböck,
Kopparty, Saraf, Theorem 1.5, p. 9), with the public Haböck MCA allowance
(Theorem 2, p. 4) and 20 bits of grinding on the query seeds: 87.28 to 99.52
conditional bits on every case. The reported `security_bits: 112` is
that query target, not a verified total; the note records the bounds and the outstanding proof and
verifier obligations.

Recorded for reference; the harness and its caveats are in `docs/csp.md`. Method: the csp-benchmarks
Rust harness (`csp-rust/`, a thin crate over the Mojo prover's C ABI): Criterion times `prove` and `verify`
in-process on a prepared session (prover built, tables uploaded, arena filled), the same way the published
Rust systems are timed, so these are the like-for-like numbers and the ones to submit. Preprocessing is the
per-grid tables; peak memory is RSS, which excludes the Metal arena. Cells are committed witness cells. The
shell track (`csp/`, whole cold process per run through `benchmark.sh`) is kept for the harness's
non-Rust path; its numbers are about 150 ms higher on the small cases and 260 ms on ECDSA (process start,
Metal setup, kernel compile, arena fill) and are not tabulated here. Warm in-process GPU-only times are the
`warm prove` line of `bench/bench_<target>.mojo`.

**2026-09-18, Apple M1 Pro 16 GB (Criterion mean of 10 samples; `collect_benchmarks` fills the durations and the memory report; load average about 7 during the run)**

| target | input | prove ms | verify ms | proof bytes | preprocessing bytes | peak RSS MB | cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sha256 | 128 B | 34 | 20 | 174,424 | 77,552 | 52 | 308,224 |
| sha256 | 256 B | 40 | 30 | 186,416 | 72,860 | 54 | 462,336 |
| sha256 | 512 B | 51 | 35 | 197,864 | 146,076 | 54 | 924,672 |
| sha256 | 1024 B | 67 | 30 | 216,380 | 336,308 | 55 | 1,585,152 |
| sha256 | 2048 B | 94 | 48 | 216,380 | 624,212 | 61 | 3,698,688 |
| keccak | 128 B | 49 | 27 | 254,940 | 46,972 | 52 | 218,112 |
| keccak | 256 B | 48 | 28 | 265,152 | 52,052 | 53 | 436,224 |
| keccak | 512 B | 51 | 33 | 283,552 | 70,292 | 54 | 872,448 |
| keccak | 1024 B | 72 | 36 | 301,908 | 75,204 | 57 | 1,744,896 |
| keccak | 2048 B | 77 | 44 | 321,296 | 129,252 | 62 | 3,489,792 |
| poseidon | 2 | 106 | 41 | 371,285 | 129,252 | 57 | 1,916,928 |
| poseidon | 4 | 104 | 40 | 373,109 | 129,252 | 57 | 1,916,928 |
| poseidon | 8 | 105 | 40 | 373,845 | 129,252 | 57 | 1,916,928 |
| poseidon | 12 | 169 | 50 | 397,621 | 321,544 | 64 | 4,472,832 |
| poseidon | 16 | 164 | 49 | 395,861 | 321,544 | 63 | 4,472,832 |
| ecdsa | 1 sig | 323 | 121 | 544,516 | 497,940 | 98 | 19,574,784 |

Criterion's 10-sample runs flag high outliers on most rows (an iteration lands on the previous
session's arena being released); the small rows move a few ms between runs on a loaded machine.
