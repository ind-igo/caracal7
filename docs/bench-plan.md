# Comparison bench plan

This page is the plan for one bench that compares caracal7 with the public prover benchmarks. It is not built.
The plan has four tracks, one for each group of public benchmarks. One set of rules applies to every row. A
list of work items at the end gives the order in which to build.

## Rules for every row

A number is comparable only when the timer, the columns and the machine match. Every track uses these rules.

- **One timer from input to proof bytes.** It includes the host trace, the upload, `load_trace` / `load_advice`
  / `load_public` and `prove`. The one-time setup (tables, arena, kernel compile) is a separate line, as the
  other systems report keygen. The warm-prove median stays as an extra column. The `csp-rust` track already
  times this way (`docs/csp.md`); `bench/bench_sha256_chain.mojo` does not, because it times the warm `prove`
  only.
- **The same columns.** Workload and size, total ms, ms per unit, units per second, proof bytes, verify ms, peak
  host memory, peak device memory (the arena), hardware, and the security numbers: 87.50 to 88.37 interactive
  bits and 107.12 to 108.18 work bits (`docs/soundness.md`), printed as they are, never as a certified level.
  For every other system, record the security target that it states.
- **The same machine.** A comparison row runs on the same device as the other system, or it says clearly that
  the devices differ. For NVIDIA, rent one box and run every system on it (`bench/<system>/README.md` recipes,
  as `bench/zkpassport/README.md` does).
- **The same computation.** Where the other system hashes a fixed input, use the same input and check that the
  output digest is equal.
- **Output.** One JSON line per row, so that one script can make every table.

## Track 1: the client-side suite

[csp-benchmarks](https://github.com/privacy-ethereum/csp-benchmarks) publishes to
[ethproofs.org/csp-benchmarks](https://ethproofs.org/csp-benchmarks). It runs SHA-256, Keccak, Poseidon,
Poseidon2 and ECDSA at 128 to 2048 bytes (2 to 16 field elements for the Poseidons) on an AWS `mac2.metal`
(M1, 8 cores, 16 GB). The page lists 16 systems; caracal7 is not one of them.

| target, largest input | leaders on the page | caracal7 (`README.md`) |
|---|---|---|
| sha256, 2048 B | Flock 34 ms, Binius64 67 ms, Spartan2 542 ms | 94 ms |

The harness exists: `csp/` (shell track) and `csp-rust/` (Rust track, the like-for-like one). What is missing:

- The submission itself: the system folder named `caracal7`, the metadata that the harness asks for
  (`BenchProperties`: classification, security level, audit status), and a run on the reference host.
- A `poseidon2` target. The current Poseidon workload is the Expander Poseidon (`docs/poseidon.md`), not
  Poseidon2. Poseidon2 over M31 needs its own round constants and its internal matrix; the bit layout stays.
- Fill in the Keccak, Poseidon, Poseidon2 and ECDSA leaders from the page before the submission. This plan has
  only the SHA-256 row.

## Track 2: hash throughput

The hash-proving systems publish throughput on a large batch, not the time for one small message. The
csp numbers are mostly a fixed cost: Keccak at 2048 B is 16 permutations in 77 ms, which says little about the
cost of one permutation.

| source | workload | hardware | published number |
|---|---|---|---|
| [Binius](https://www.binius.xyz/benchmarks/) | 1365 Keccak-f permutations | c8g.16xlarge (64-core Graviton) | 111.82 ms, 304 KiB proof |
| [Binius](https://www.binius.xyz/benchmarks/) | 2048 Blake2s compressions | c8g.16xlarge | 166.15 ms, 360 KiB proof |
| [Expander](https://blog.polyhedra.network/introducing-expander-the-fastest-gkr-proof-system-to-date/) | Keccak-f | M3 Max | 4500 permutations per second |
| [Stwo](https://starkware.co/blog/starkware-new-proving-record/) | Poseidon2 | 4-core CPU | 500K hashes per second |
| [Plonky3](https://polygon.technology/blog/open-source-polygon-plonky3-is-once-again-the-fastest-zk-proving-system) | Poseidon2 | M3 Max | 2M hashes per second |
| [OpenVM CI](https://github.com/openvm-org/openvm/blob/benchmark-results/index.md) | SHA-256 of 10 MiB (163,840 compressions) | AWS g7.4xlarge (RTX PRO 4500 Blackwell) | app proof 5.40 s, 6.42 s with aggregation |

The bench: for each primitive (SHA-256 compression, Keccak-f, Poseidon and Poseidon2 permutation), fill the
largest grid that fits the device, and report the time per unit and the units per second. Run it on the M1 Pro
and on the NVIDIA box. Also run the exact Binius instance, 1365 Keccak-f permutations, as one proof.

What the rows mean:

- Stwo and Plonky3 prove Poseidon2 in its native field, where a permutation is about one row. Caracal7 splits
  the values into bits, so it will lose this row. Print the row; do not lead with it.
- Blake2s has no workload. Leave it out until one exists.
- The largest grid is bounded by the arena. `docs/design.md` names arena region reuse as the next item for
  large grids; this track shows how much that costs.

## Track 3: zkVM suites

Caracal7 is a circuit prover, not a VM, so it can run only the rows that have the shape of a precompile.
Fibonacci, regex, btreemap, ETH transfers and Ethereum blocks are out of reach. Every table in this track
says that the other rows are general zkVMs and that their numbers include the cost of that generality.

| suite | programs | rows caracal7 can run |
|---|---|---|
| [OpenVM CI](https://github.com/openvm-org/openvm/blob/benchmark-results/index.md) | fibonacci, keccak, sha2 10 MiB, regex, ecrecover, pairing, kitchen_sink; guest `sha256_iter` | keccak, sha2, sha256_iter, ecrecover |
| [a16z zkvm-benchmarks](https://github.com/a16z/zkvm-benchmarks) | fib, sha2, sha2-chain, sha3, sha3-chain, btreemap | sha2, sha2-chain, sha3, sha3-chain |
| [grandchildrice/zkvm-benchmarks](https://github.com/grandchildrice/zkvm-benchmarks) ([Fenbushi write-up](https://fenbushi.vc/2025/08/29/benchmarking-zkvms-current-state-and-prospects/)) | fib, sha2 of 2048 B, ecdsa, ethtransfer | sha2, ecdsa |
| [zkbenchmarks.com](https://blog.alignedlayer.com/zkbenchmarks/) | fib, keccak, RSP | keccak |

Published GPU numbers for SHA-256 on the RTX 5090 (grandchildrice, one 2048-byte message, 16 cores):
RISC Zero GPU 0.51 s, SP1 GPU 3.75 s; on the CPU, OpenVM 1.03 s and Jolt 3.46 s. These are one-message times
with a fixed cost, closer to the csp row than to a chain.

### The hash chain

The hash chain is the main row of this track. OpenVM's `sha256_iter` starts from the SHA-256 of the empty
message and hashes the 32-byte digest 150,000 times. The a16z `sha2-chain` has the same shape. One hash of a
32-byte value is one compression, so OpenVM's 10 MiB result also compares per compression: about 0.033 ms,
against caracal7's 0.22 ms per hash on the RTX 3090 (`README.md`). The GPUs differ and the sizes differ by
about 190x, so this is not yet a fair comparison. The chain bench makes it fair:

1. **Same input.** Start from SHA-256 of the empty message, as `sha256_iter` does, and check the final digest
   against OpenVM's output at the same n.
2. **Same timer.** The input-to-proof timer of the rules above; the host trace is inside it.
3. **Same scale.** n = 1K, 10K and 150K. Split the chain into segments of the largest grid that fits the card
   (875 hashes on `CLIENT.grid(224, 8064)` today). The last digest of each segment is the public input of the
   next segment. Report the sum of the segment times. This compares with OpenVM's `app_proof` line, which is
   also segments without aggregation.
4. **No aggregation.** Caracal7 has no recursion, so 150K hashes give about 170 proofs of about 190 KB, about
   33 MB in total, where OpenVM gives one aggregated proof. Show the total proof bytes, and show the other
   systems both with and without aggregation (OpenVM's leaf and internal proofs add about 1 s).

### Other systems on the same box

Run OpenVM (CUDA), RISC Zero (CUDA) and SP1 (CUDA) on the rented NVIDIA box at the same n for `sha2-chain`,
`sha3-chain` and `ecrecover`. One recipe folder per system under `bench/`. The EF
[ere](https://github.com/eth-act/ere) crate gives one interface to several zkVMs; check that it drives each
prover's CUDA path before you use it in place of the per-system recipes.

## Track 4: applications

- **Passport.** Done: zkPassport's Noir circuits with Barretenberg on the same M1 Pro, 45 s against 2.5 s
  (`bench/zkpassport/README.md`).
- **zk-email.** RSA-2048 and SHA-256 over an email header have the same shape as the `rsa` and `sha256`
  workloads. Find the published zk-email prover numbers and their circuit sizes first.
- **P-256.** Passkeys and zkID verify P-256 signatures. The P-256 workload (`workloads/ecdsa_p256.mojo`) is
  in progress. Find a public P-256 prover benchmark to compare against.

## Work items

In this order. Each item is one change with its own check.

1. **Common row output.** One helper that times input to proof bytes and prints one JSON row with the
   columns of the rules. Use it in the new benches; keep the current benches' text output.
2. **Track 1 submission.** The `caracal7` system folder, the metadata, the Poseidon2 target, a run of
   `BENCH_INPUT_PROFILE=full` on the M1, and the pull request to csp-benchmarks.
3. **Track 2 throughput bench.** `bench/bench_throughput.mojo`: SHA-256, Keccak-f, Poseidon, Poseidon2 on the
   largest grid that fits; the 1365-permutation Keccak instance.
4. **Track 3 chain bench.** Change `bench/bench_sha256_chain.mojo`: the `sha256_iter` start value, the digest
   check, the segment sweep, the new timer. Then the Keccak chain in the same form.
5. **Track 3 recipes.** `bench/openvm/README.md`, then RISC Zero and SP1, each run on the same box as item 4.
6. **Tables.** One script reads the JSON rows of every track and writes the tables for `README.md`.

## Open questions

- What security level does each other system target, and with which assumptions? A table without this column
  compares different things.
- Does the M1 GPU on `mac2.metal` behave like the M1 Pro's GPU? The csp harness has not run caracal7 there.
- What is the largest grid that fits in 24 GB on the RTX 3090, and in 16 GB on the M1? This sets the segment
  size of the chain bench and the batch size of the throughput bench.
