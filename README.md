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

Recorded for reference; the harness and its caveats are in `docs/csp.md`. Method: the csp-benchmarks
orchestrator (`BENCH_INPUT_PROFILE=full`, hyperfine 10 runs) against `caracal7/`, so every time is a
whole cold process through the shell wrapper (the warm in-process prover is 2 to 4 times faster; see
`bench/`), and peak memory is RSS, which excludes the Metal arena. Cells are committed witness cells.

**2026-09-11, Apple M1 Pro 16 GB, commit 127e53c**

| target | input | prove ms | verify ms | proof bytes | peak RSS MB | cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sha256 | 128 B | 280 | 52 | 214,556 | 47 | 308,224 |
| sha256 | 256 B | 183 | 75 | 224,900 | 49 | 462,336 |
| sha256 | 512 B | 223 | 56 | 246,224 | 50 | 924,672 |
| sha256 | 1024 B | 292 | 90 | 280,468 | 52 | 1,585,152 |
| sha256 | 2048 B | 425 | 76 | 297,592 | 57 | 3,698,688 |
| keccak | 128 B | 187 | 59 | 283,404 | 48 | 218,112 |
| keccak | 256 B | 226 | 60 | 302,884 | 49 | 436,224 |
| keccak | 512 B | 194 | 106 | 323,548 | 50 | 872,448 |
| keccak | 1024 B | 250 | 100 | 351,000 | 53 | 1,744,896 |
| keccak | 2048 B | 307 | 68 | 378,704 | 57 | 3,489,792 |
| poseidon | 2 | 319 | 90 | 418,957 | 53 | 1,916,928 |
| poseidon | 4 | 318 | 74 | 422,029 | 53 | 1,916,928 |
| poseidon | 8 | 315 | 96 | 417,597 | 53 | 1,916,928 |
| poseidon | 12 | 479 | 87 | 458,765 | 59 | 4,472,832 |
| poseidon | 16 | 435 | 76 | 460,461 | 59 | 4,472,832 |
| ecdsa | 1 sig | 979 | 130 | 654,464 | 91 | 19,574,784 |

The verify column and the 128-byte SHA-256 row carry run-to-run noise of tens of ms (process start,
the first size of a sweep paying cold caches); the in-process verifier is 15 to 90 ms across these.
