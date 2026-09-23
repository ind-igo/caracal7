# sha256_iter on one RTX 5090: the data

The measurements behind the README table "SHA-256 chain against OpenVM on one RTX 5090". Both systems ran on
the same rented box, one after the other, never at the same time.

## Machine

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, 32 GB, driver 580.159.03 |
| CPU | AMD EPYC 7B12 (2.25 GHz), 256 threads visible, 503 GB RAM |
| Image | `nvidia/cuda:12.8.0-devel-ubuntu22.04`, the compat `libcuda` moved out of the loader path |
| caracal7 | MAX 26.5 (`uv sync`), `-D CARACAL_NVIDIA_MMA` |
| OpenVM | v2.0.2, `--features cuda`, the CI environment of `README.md` in this folder |

## OpenVM

From the metrics file (`OUTPUT_PATH`) of each run; times are sums over the segments.

| metric | segment proofs (`--app-only`) | with aggregation |
| --- | ---: | ---: |
| `app_prove_time_ms` | 47,521 | 47,336 |
| `total_proof_time_ms`, app (21 segments) | 46,651 | 46,476 |
| `total_proof_time_ms`, leaf | | 9,916 |
| `total_proof_time_ms`, internal | | 5,194 |
| proof bytes (compressed) | | 315,319 (278,228) |

The aggregated total is 47,336 + 9,916 + 5,194 = 62,446 ms. The split of the segment proofs (app run):

| part | ms |
| --- | ---: |
| `execute_metered_time_ms` (one pass, 96,750,720 RISC-V instructions) | 827 |
| `execute_preflight_time_ms` | 6,165 |
| `set_initial_memory_time_ms` | 7,362 |
| `trace_gen_time_ms` | 5,857 |
| `stark_prove_excluding_trace_time_ms` | 27,231 |
| of which `prover.rap_constraints` (zerocheck and LogUp GKR) | 20,365 |
| of which `prover.openings` (stacked reduction and WHIR) | 4,845 |
| of which `prover.main_trace_commit` | 1,997 |

Total cells: 5,916,863,740. GPU memory pool: 4.7 GiB. The benchmark parameters are
`app_params_with_100_bits_security`, OpenVM's stated target.

## caracal7

`caracal7_rtx5090.jsonl` holds the output of `bench_sha256_iter` for four runs, one after each host change;
the `step` field names the change. The last step is the one in the README. Per hash, at n = 150,000 with
875-hash segments:

| run, after | total s | host ms | upload ms | GPU ms | verify s |
| --- | ---: | ---: | ---: | ---: | ---: |
| first run (rows not kept) | 98.3 | 0.246 | 0.189 | 0.221 | 98.2 |
| trace lanes in parallel, power table, public columns copied by row | 74.1 | 0.235 | 0.040 | 0.219 | 98.8 |
| scatter, selectors and public columns in parallel | 83.9 | 0.248 | 0.090 | 0.221 | 141.3 |
| carries in one SIMD sum | 53.7 | 0.092 | 0.046 | 0.220 | 91.1 |
| pipelined (host under the device) | 39.6 | 0.096 | 0.044 | | 91.0 |

The third run was on a busier box: its upload and verify times rose with no code change in those paths. In
the pipelined run the host time is mostly under the device time, so the columns do not add up to the total.

The same rows with 125-hash segments (the grid the soundness ledger covers), total seconds: 196.1, 173.2,
181.8, 122.3, 87.2. The last digests: n = 1,000 `0dc9b0e0…70f0c8`, n = 10,000 `20abb5cf…76bc4b`,
n = 150,000 `8f7ce107…818166`; all three agree with Python `hashlib`.
