//! The csp-benchmarks Rust track for caracal7. A `Session` is a workload with its prover built (the
//! harness's prepared context); `prove` runs every input-dependent step (trace, public data, loads,
//! prove) on it, which is what the harness times. The inputs come from the shared `utils` generators.

use std::borrow::Cow;
use std::sync::OnceLock;
use utils::harness::{AuditStatus, BenchProperties};

#[derive(Clone, Copy, Debug)]
pub enum Target {
    Sha256 = 0,
    Keccak = 1,
    Poseidon = 2,
    Ecdsa = 3,
}

unsafe extern "C" {
    fn c7_runtime() -> isize;
    fn c7_open(runtime: isize, target: isize, size: isize, data: *const u8) -> isize;
    fn c7_prove(handle: isize, target: isize, size: isize, dst: *mut u8, cap: isize) -> isize;
    fn c7_verify(handle: isize, target: isize, size: isize, proof: *const u8, n: isize) -> isize;
    fn c7_cells(handle: isize, target: isize, size: isize) -> isize;
    fn c7_preprocessing_bytes(handle: isize, target: isize, size: isize) -> isize;
    fn c7_close(handle: isize, target: isize, size: isize) -> isize;
}

pub struct Session {
    handle: isize,
    target: Target,
    size: usize,
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { c7_close(self.handle, self.target as isize, self.size as isize) };
    }
}

/// The input bytes as `cli/ffi.mojo` lays them out: message then digest; little-endian u32 elements;
/// or e, x_Q, y_Q, r, s as 32-byte big-endian words.
fn input_bytes(target: Target, size: usize) -> Vec<u8> {
    match target {
        Target::Sha256 => {
            let (m, d) = utils::generate_sha256_input(size);
            [m, d].concat()
        }
        Target::Keccak => {
            let (m, d) = utils::generate_keccak_input(size);
            [m, d].concat()
        }
        Target::Poseidon => utils::generate_poseidon_input_m31(size)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        Target::Ecdsa => {
            let (digest, (x, y), sig) = utils::generate_ecdsa_k256_input();
            [digest, x, y, sig].concat()
        }
    }
}

/// One device context for the process: kernels compile once per context, so sessions share it.
fn runtime() -> isize {
    static RUNTIME: OnceLock<isize> = OnceLock::new();
    *RUNTIME.get_or_init(|| {
        let rt = unsafe { c7_runtime() };
        assert!(rt > 0, "caracal7: no device context");
        rt
    })
}

pub fn prepare(target: Target, size: usize) -> Session {
    let data = input_bytes(target, size);
    let handle = unsafe { c7_open(runtime(), target as isize, size as isize, data.as_ptr()) };
    assert!(handle > 0, "caracal7: c7_open failed for {target:?} {size}");
    Session { handle, target, size }
}

pub fn prove(s: &Session) -> Vec<u8> {
    let mut buf = vec![0u8; 4 << 20];
    let n = unsafe { c7_prove(s.handle, s.target as isize, s.size as isize, buf.as_mut_ptr(), buf.len() as isize) };
    assert!(n >= 0, "caracal7: c7_prove failed ({n})");
    buf.truncate(n as usize);
    buf
}

pub fn verify(s: &Session, proof: &Vec<u8>) {
    let ok = unsafe { c7_verify(s.handle, s.target as isize, s.size as isize, proof.as_ptr(), proof.len() as isize) };
    assert_eq!(ok, 1, "caracal7: proof rejected");
}

/// Committed witness cells: the circuit size the benchmark reports.
pub fn cells(s: &Session) -> usize {
    unsafe { c7_cells(s.handle, s.target as isize, s.size as isize) as usize }
}

/// The per-grid tables built before any input is seen.
pub fn preprocessing_size(s: &Session) -> usize {
    unsafe { c7_preprocessing_bytes(s.handle, s.target as isize, s.size as isize) as usize }
}

pub fn proof_size(proof: &Vec<u8>) -> usize {
    proof.len()
}

/// The RAM-measurement binaries (`src/bin/<target>_mem_caracal7.rs`): prepare and prove once for
/// `--input-size`, which `measure_mem_avg.sh` samples with `/usr/bin/time`.
pub fn mem_main(target: Target) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let size: usize = match args.as_slice() {
        [flag, n] if flag == "--input-size" => n.parse().expect("input size"),
        [one] if one.starts_with("--input-size=") => one["--input-size=".len()..].parse().expect("input size"),
        _ => panic!("usage: --input-size <n>"),
    };
    let session = prepare(target, size);
    let _proof = prove(&session);
}

pub const CARACAL7_BENCH_PROPERTIES: BenchProperties = BenchProperties {
    proving_system: Cow::Borrowed("caracal7"),
    field_curve: Cow::Borrowed("F_127 (bit-layout, 16-coordinate extension)"),
    iop: Cow::Borrowed("column families over a chain grid, batched quotient, tensor-code tail"),
    pcs: Some(Cow::Borrowed("Reed-Solomon tensor code with Merkle commitments")),
    arithm: Cow::Borrowed("column families over a chain grid"),
    is_zk: false,
    is_zkvm: false,
    security_bits: 103,
    is_pq: true,
    is_maintained: true,
    is_audited: AuditStatus::NotAudited,
    isa: None,
};
