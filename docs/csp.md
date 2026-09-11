# csp-benchmarks harness

`csp/` is the system folder for [privacy-ethereum/csp-benchmarks](https://github.com/privacy-ethereum/csp-benchmarks)
(the non-Rust path: `benchmark.sh --system-dir`). `cli/main.mojo` is the binary behind it, one grid per
target and input size, built by `csp/osx_local_setup.sh` into `csp/target/caracal7`:

    caracal7 prove|verify <target> <size> <proof-path> <input...>

Targets and grids: sha256 (32 rows, 193 to 2113 chains), keccak (64 rows, 24 chains per 136-byte
block), poseidon (64 x 368 up to 8 elements, 64 x 720 up to 16), ecdsa (144 x 576, one secp256k1
signature). The prepare scripts turn the utils generator's lines into a state JSON; prove and verify
pass its fields to the binary with jq; measure runs one proof and reads the sizes.

Local run, from a csp-benchmarks checkout with `cargo build --release -p utils` done:

    BENCH_INPUT_PROFILE=reduced bash ./benchmark.sh --system-dir <this repo>/csp --logging --quick

## What the numbers mean

- **Prove and verify time** are whole-process wall clock, cold: hyperfine starts the binary per run.
  That includes the Metal device setup and the first-launch kernel compile, so a 128-byte SHA-256 proof
  reports about 180 ms where the warm prover takes 57 ms. `bench/` has the warm and per-stage numbers.
- **Peak memory** is maximum RSS under `/usr/bin/time`. The prover's arena lives in Metal buffers, which
  macOS does not count in RSS; the SHA-256 sweep reports 70 MB. The arena size is the honest figure and
  `Prover` prints it under `profile`.
- **Preprocessing size** is 0: the compiled statement (a few KB of family entries) is derived at run
  time in under a millisecond, nothing is persisted between runs.
- **Circuit size** (`circuit_sizes.json`) is committed witness cells, `columns_w x N`, the quantity the
  prover's cost model scales with.

## Generator gaps

The utils CLI prints secp256r1 signatures and 254-bit Poseidon elements; both a k256 and a Mersenne-31
generator exist in the utils library but are not exposed. Until they are, `ecdsa_prepare.sh` uses a
fixed secp256k1 vector and `poseidon_prepare.sh` reduces the elements mod 2^31 - 1.
