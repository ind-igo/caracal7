# OpenVM baseline on the same machine

The comparison for `bench/bench_sha256_iter.mojo` (`docs/bench-plan.md`, Track 3): OpenVM's `sha256_iter`
guest proved with OpenVM's CUDA prover on the same rented box. The guest starts from SHA-256 of the empty
message and hashes the 32-byte digest 150,000 times; it has a prebuilt ELF in their repository, so only a
host binary is added.

## Recipe

On a Linux box with an NVIDIA GPU and the CUDA toolkit (`nvcc` on the PATH):

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
git clone --depth 1 --branch v2.0.2 https://github.com/openvm-org/openvm
cp sha256_iter.rs openvm/benchmarks/prove/src/bin/
printf '\n[[bin]]\nname = "sha256_iter"\npath = "src/bin/sha256_iter.rs"\n' >> openvm/benchmarks/prove/Cargo.toml
cd openvm/benchmarks/prove && cargo build --release --bin sha256_iter --features cuda
export VPMM_PAGES=2048 RUST_LOG=info
export JEMALLOC_SYS_WITH_MALLOC_CONF="retain:true,background_thread:true,metadata_thp:always,thp:always,dirty_decay_ms:-1,muzzy_decay_ms:-1,abort_conf:true"
OUTPUT_PATH=app.json ../../target/release/sha256_iter --app-only   # segment proofs, no aggregation
OUTPUT_PATH=agg.json ../../target/release/sha256_iter              # plus leaf and internal aggregation
```

`sha256_iter.rs` is their `sha2_bench.rs` with the `sha256_iter` guest and no input. The flags and the
environment are those of their CI (`.github/workflows/benchmark-call.yml`). The metrics file has
`app_prove_time_ms` (the app proof, keygen excluded) and `total_proof_time_ms` per group (`app`, `leaf`,
`internal`); the log prints the proof size. OpenVM's benchmark parameters target 100 bits
(`app_params_with_100_bits_security`); that is their stated target, not a figure this repository checks.
