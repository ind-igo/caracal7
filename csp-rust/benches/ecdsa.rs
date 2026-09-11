use caracal7_bench::{cells, prepare, preprocessing_size, proof_size, prove, verify, Target, CARACAL7_BENCH_PROPERTIES};
use utils::harness::ProvingSystem;

utils::define_benchmark_harness!(
    BenchTarget::Ecdsa,
    ProvingSystem::Caracal7,
    Some("secp256k1"),
    "ecdsa_mem_caracal7",
    CARACAL7_BENCH_PROPERTIES,
    |_| None,
    |size| prepare(Target::Ecdsa, size),
    cells,
    prove,
    verify,
    preprocessing_size,
    proof_size
);
