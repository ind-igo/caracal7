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
```

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
orchestrator (`BENCH_INPUT_PROFILE=full`, hyperfine 10 runs) against `caracal7/`, so every time is a
whole cold process through the shell wrapper (the warm in-process prover is 2 to 4 times faster; see
`bench/`), and peak memory is RSS, which excludes the Metal arena. Cells are committed witness cells.

**2026-09-14, Apple M1 Pro 16 GB, commit 9c09a5a** (`E = F_(127^20)`, Johnson regime, tail rate 1/8, 20 bits of
grinding). Harness columns from `benchmark.sh` with `BENCH_INPUT_PROFILE=full` (hyperfine 10 runs); the warm
prove column is the in-process `Prover.prove` on a constructed prover (the `warm prove` line of
`bench/bench_<target>.mojo`; the keccak and poseidon benches run three sizes). The gap between the two is the cold floor:
process start, the shell wrapper, Metal setup and first-launch kernel compile, and the arena fill (about
0.3 ms per MB; 1.57 GB for ECDSA). The 2026-09-11 table at `e = 16`, lambda' 103 is in git history (commit 127e53c).

| target | input | prove ms | warm prove ms | verify ms | proof bytes | peak RSS MB | cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sha256 | 128 B | 161 | 40 | 54 | 177,400 | 50 | 308,224 |
| sha256 | 256 B | 174 | 51 | 65 | 187,860 | 51 | 462,336 |
| sha256 | 512 B | 195 | 60 | 82 | 198,536 | 52 | 924,672 |
| sha256 | 1024 B | 269 | 83 | 79 | 215,228 | 55 | 1,585,152 |
| sha256 | 2048 B | 345 | 157 | 83 | 215,900 | 60 | 3,698,688 |
| keccak | 128 B | 164 | 43 | 63 | 258,692 | 51 | 218,112 |
| keccak | 256 B | 168 | – | 66 | 267,288 | 51 | 436,224 |
| keccak | 512 B | 183 | – | 71 | 282,040 | 53 | 872,448 |
| keccak | 1024 B | 210 | 73 | 73 | 301,716 | 55 | 1,744,896 |
| keccak | 2048 B | 246 | 116 | 79 | 320,112 | 60 | 3,489,792 |
| poseidon | 2 | 289 | 133 | 87 | 371,381 | 57 | 1,916,928 |
| poseidon | 4 | 300 | – | 84 | 370,261 | 57 | 1,916,928 |
| poseidon | 8 | 293 | 121 | 92 | 372,917 | 56 | 1,916,928 |
| poseidon | 12 | 486 | – | 97 | 397,077 | 62 | 4,472,832 |
| poseidon | 16 | 407 | 229 | 93 | 395,381 | 62 | 4,472,832 |
| ecdsa | 1 sig | 684 | 369 | 158 | 545,892 | 97 | 19,574,784 |

The verify column and the 128-byte SHA-256 row carry run-to-run noise of tens of ms (process start,
the first size of a sweep paying cold caches); the in-process verifier is 15 to 90 ms across these.

### Rust track

The same harness's Rust track (`csp-rust/`, see `docs/csp.md`): Criterion times `prove` in-process on a
prepared session (prover built, tables uploaded, arena filled), so the number is trace, public data,
loads and the prove, without process start or kernel compile. Preprocessing is the per-grid tables.

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

The ECDSA prove is 404 ms in the 2026-09-13 Rust-track A/B (decisions.md "ECDSA: the serial spots run wide"),
425 here; this machine's GPU clock drifts by 10 percent over a long session, so rows within that band are equal.

Criterion's 10-sample runs flag high outliers on most rows (an iteration lands on the previous
session's arena being released); the 1024-byte SHA-256 row is one such run, its warm prove is 75 ms.

