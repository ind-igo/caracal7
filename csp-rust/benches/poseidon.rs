use caracal7_bench::{cells, prepare, preprocessing_size, proof_size, prove, verify, Target, CARACAL7_BENCH_PROPERTIES};
use utils::harness::ProvingSystem;

utils::define_benchmark_harness!(
    BenchTarget::Poseidon,
    ProvingSystem::Caracal7,
    None,
    "poseidon_mem_caracal7",
    CARACAL7_BENCH_PROPERTIES,
    |_| None,
    |size| prepare(Target::Poseidon, size),
    cells,
    prove,
    verify,
    preprocessing_size,
    proof_size
);
