# csp-benchmarks harness

`caracal7/` is the system folder (its basename is the system name in the metrics) for [privacy-ethereum/csp-benchmarks](https://github.com/privacy-ethereum/csp-benchmarks)
(the non-Rust path: `benchmark.sh --system-dir`). `cli/main.mojo` is the binary behind it, one grid per
target and input size, built by `caracal7/osx_local_setup.sh` into `caracal7/target/caracal7`:

    caracal7 prove|verify <target> <size> <proof-path> <input...>

Targets and grids (the smallest legal grid that holds the input; `Profile.grid` rounds each axis up to
2^a m): sha256 32 rows x 224 / 336 / 672 / 1152 / 2688 chains for 128 to 2048 bytes, keccak 64 x
24 / 48 / 96 / 192 / 384, poseidon 64 x 384 up to 8 elements and 64 x 896 up to 16, ecdsa 144 x 576
(one secp256k1 signature; the harness's one input size, 32). The sizes are a compile-time whitelist:
any other `INPUT_SIZE` aborts. The scripts need jq >= 1.6 and bash (3.2 is enough). The prepare scripts turn the utils generator's lines into a state JSON; prove and verify
pass its fields to the binary with jq; measure runs one proof and reads the sizes. The verifier is given
the message and the generator's digest, so the timed verify never hashes on the host.

Local run, from a csp-benchmarks checkout with `cargo build --release -p utils` done:

    BENCH_INPUT_PROFILE=reduced bash ./benchmark.sh --system-dir <this repo>/caracal7 --logging --quick

## Rust track (`csp-rust/`)

csp-benchmarks times its Rust crates differently: Criterion runs `prepare` per iteration outside the
timer and times `prove(&prepared)` in-process, the way plonky2's `prove(circuit_data, pw)` is measured.
`csp-rust/` enters caracal7 on that track through the C API in `cli/ffi.mojo`, built as a shared
library by the crate's `build.rs` with the repo's Mojo toolchain.

- **Prepared context** (`Session` in `workload.mojo`): the compiled statement, the constructed prover,
  its tables uploaded, the arena filled, synchronized. Sessions share one `DeviceContext`
  (`c7_runtime`): kernels compile once per context, so a fresh context per session would make every
  prove cold.
- **Timed prove**: everything that depends on the inputs. Trace, advice, public data, the loads, then
  `Prover.prove`. Verify is `verify_workload` with `W.public_data` inside, as on the shell track.
- **Preprocessing size** is the per-grid tables the constructor builds (`Session.preprocessing_bytes`),
  the analogue of a prover key. Circuit size is committed witness cells as before.
- **Peak memory** comes from `src/bin/<target>_mem_caracal7`, prepare and prove once, sampled by
  `measure_mem_avg.sh`; RSS still excludes the Metal arena.
- Inputs come from the `utils` generators in Rust, including the k256 signature and the Mersenne-31
  Poseidon elements the CLI does not expose, so the shell track's generator gaps do not apply here.

To run it, from a csp-benchmarks checkout: copy `csp-rust/` to `<checkout>/caracal7-rs/` (a symlink
does not work: cargo resolves it outside the workspace), add `"caracal7-rs"` to the workspace members,
add `Caracal7` to `ProvingSystem` in `utils/src/harness.rs` with `as_str` `"caracal7"`, then

    cd caracal7-rs && CARACAL7_REPO=<this repo> BENCH_INPUT_PROFILE=full cargo bench --bench sha256

`build.rs` runs `uv run mojo build --emit shared-lib` in `CARACAL7_REPO` (default: the crate's parent)
and links the library with rpaths into the repo's venv, so `cargo bench` needs no further setup.
Results land in `caracal7-rs/*_metrics.json`; `collect_benchmarks` fills the durations from Criterion's
`target/criterion/*/new/estimates.json` (mean point estimate). Only one entry per system goes on the
site; the Rust track is the one to submit.

## What the numbers mean (shell track)

- **Prove and verify time** are whole-process wall clock, cold: hyperfine starts the binary per run.
  That includes the Metal device setup and the first-launch kernel compile, so a 128-byte SHA-256 proof
  reports about 180 ms where the warm prover takes 57 ms. `bench/` has the warm and per-stage numbers.
- **Verify time** includes `W.public_data`: the verifier rebuilds the public columns over the grid
  (O(N) host work for the hashes) and, for ecdsa, runs a native secp256k1 check with the hint
  generation before the proof check. It is the cost of verifying the statement as given, not the proof
  check alone; the profile marks in `verify` separate the two.
- **Peak memory** is maximum RSS under `/usr/bin/time`. The prover's arena lives in Metal buffers, which
  macOS does not count in RSS; the SHA-256 sweep reports 70 MB. The arena size, `ProverLayout.bytes`, is
  the honest figure.
- **Process overhead.** A 128-byte SHA-256 verify is 15 ms in-process, 20 ms as a bare binary run and
  40 ms through `verify.sh` (bash and one jq call). Both are inside every timed number; a loaded
  machine (the RAM pass runs ten proofs back to back) stretches them further.
- **Preprocessing size** is 0: the compiled statement (a few KB of family entries) is derived at run
  time in under a millisecond, nothing is persisted between runs.
- **Circuit size** (`circuit_sizes.json`) is committed witness cells, `columns_w x N`, the quantity the
  prover's cost model scales with.

## Generator gaps

The utils CLI prints secp256r1 signatures and 254-bit Poseidon elements; both a k256 and a Mersenne-31
generator exist in the utils library but are not exposed. Until they are, `ecdsa_prepare.sh` uses a
fixed secp256k1 vector and `poseidon_prepare.sh` reduces the elements mod 2^31 - 1.
