# Design: memory, kernels, and boundaries

Spec: `wiki/projects/caracal7/specs/caracal-prover.md` (main), `statement-layer.md` (frontend), `polynomial-mulmod.md`. This page fixes what the spec leaves open: how bytes sit in memory, which kernels exist, and where CPU and GPU meet. Section numbers below refer to the main spec.

## 1. Rules

1. **Device resident.** Every prover stage is a GPU kernel from its first commit, including the Blake3 transcript and challenge derivation (section 10: tree and transcript on device). The host only enqueues launches in stage order and reads the finished proof bytes. The verifier is a separate program and runs on the host.
2. **One buffer type.** All committed data is bytes: one `UInt8` per `F` coordinate. `F2`, `F4`, and `E` values are 2, 4, and `e` consecutive bytes. No struct-of-arrays and no per-field buffer types. Kernels are specialized on the coordinate count, not on a field type.
3. **Parameters are comptime.** One `Params` struct carries every knob. Kernels take it as a parameter, so every loop bound and stride is a constant.
4. **No copies between stages.** Unified memory on M1 makes host and device views of one buffer cheap. A stage writes its output where the next stage reads it. There are no host reads between barriers: challenges are derived on device from the device-resident transcript, and kernels read them from device memory.
5. **Scalar reference in the test, never a CPU prover.** Each kernel's test runs a few lines of scalar Mojo on a small size and compares. The reference does not grow into a second implementation.

## 2. Parameters

```mojo
struct Params:
    comptime e: Int          # extension degree, 16 (section 1)
    comptime a1: Int         # h1 = 2^a1 * m1, 2 <= a1 <= 7 (9.1)
    comptime m1: Int         # odd, m1 | 63
    comptime a2: Int
    comptime m2: Int
    comptime L0: Int         # level-1 subgroup order, L0 | 161280; default 80,640
    comptime m_cosets: Int   # 1, 2, or 4 (9.1)
    comptime leaf_bytes: Int # 1,024, one Blake3 chunk
    comptime tail_digits: Int    # 3 per level (9.3)
    comptime tail_clear_max: Int # ~2,500 E elements (9.3)
```

Derived at comptime: `h1`, `h2`, `N = h1 * h2`, `L = m_cosets * L0`, the rate, the query count `|S|` from the formula of section 9, `n_cw`. `P` (opening points) and the column count are runtime values of the IR program, not parameters.

Milestone 1 profile: `e = 16`, `a1 = 3, m1 = 9` (h1 = 72), `a2 = 5, m2 = 1` (h2 = 32), `L0 = 80,640`, `m_cosets = 1`. This is the reference profile for synthetic columns, not a Keccak grid: the main spec lists Keccak-128 at 72 × 32 and statement-layer section 9 at 64 × 24, and that is reconciled when the frontend lands.

The tail schedule is derived, not chosen: from `N` and `e`, level by level, the message length, `L_l`, the exact rate, and `|S_l|` by the query formula, until the clear-vector bound of 9.3 stops the recursion. The whole schedule is computed before the transcript prefix is hashed, because the prefix commits to it.

## 3. Buffers

Every buffer is a `DeviceBuffer[UInt8]` with a documented shape. Shapes are (slowest ... fastest).

| buffer | shape | notes |
|---|---|---|
| `trace` | (column, x2, x1) | witness in row order of section 2, one byte per value; a chain is contiguous |
| `coeff` | (column, x2, x1, 2) | F2 monomial coefficients after the inverse 2D DFT, the two coordinates adjacent |
| `stored` | (column, slot) | mixed basis, N bytes per column. F-valued columns use the Frobenius-real slots of 9.1. E-valued columns (A, B, Q2) use the plain `(x1, x2, r)` slots of 9.2, one stored column per E coordinate, written directly by the quotient path |
| `packed` | (column, i, 4) | F4 symbols, N/4 per column |
| `code` | (s, column, n_cw, 4) | codeword rows, leaf-major: leaf `s` is contiguous, `4 * n_cw * columns` bytes |
| `tree` | (level, node, 32) | Blake3 digests, leaves first |
| `lde` | (column, G2, G1, coord) | evaluations on the residual grid G; 2 coordinates for witness columns, e for accumulators |
| `residual` | (G2, G1, e) | batched residual, E-valued |
| `v_l` | (query, n_cw, 4, e) at level 2; (query, e) later | expected symbols sent per level (9.3) |
| `w_tilde` | (slot, e) | the batched query of the current level, reused per level |
| `round_msgs` | (level, 3, 3, e) | sumcheck messages |
| `transcript` | Blake3 state plus a challenge buffer | device resident (section 6) |
| `fold_y` | (slot, e) | level-2 message y in E, slot order |
| `tail_code_l` | (s, 8, e) | level-l codeword rows, 8 E per row (9.3) |

Leaf-major `code` and `tail_code_l` cost a transpose: both encoders produce column-major codewords and the Merkle leaf wants all columns at one position. Both encoders write their output transposed in their last pass. If Apple GPU shared memory makes that pass slow, we keep a column-major copy and transpose once; measure first.

Row width of a leaf is `4 * n_cw * columns` bytes. With 1,024-byte leaves that caps a tree at 256 columns at `n_cw = 1`, which is the 16-accumulator ceiling of section 13. Wider trees split into chunked leaves later.

## 4. Kernels, milestone 1 (Ligerito on synthetic columns)

Each kernel is one `def` taking device buffers and `Params`. Grid and block shape are chosen inside a thin launcher next to the kernel, not by the caller.

| kernel | in → out | threads | notes |
|---|---|---|---|
| `idft2` | trace → coeff | one per (column, line) | inverse 2D DFT over F2, axis by axis, mixed radix 2/3/7 stages; each stage a batched small GEMM |
| `to_stored` | coeff → stored | one per (column, slot) | length-m DFT over F per axis on the odd digit, then the Frobenius-real slot bijection of 9.1; the quotient specialization writes E coordinates to plain slots (9.2) |
| `pack` | stored → packed | one per (column, i) | gather 4 slots on the packing digit into one F4 symbol |
| `rs_encode` | packed → code | one per (column, butterfly) | coset twist by `g_k^i`, then the order-L0 DFT over F4 in two passes: the power-of-two part, then the 315-point Good-Thomas part; each stage a block GEMM with 4×4 F-matrices as twiddles |
| `merkle` | code → tree | one per node per level | Blake3, 1,024-byte leaves, 32-byte nodes, one launch per level |
| `open` | stored, w_z → alpha | one per (column, point) | contraction `<w_z, stored(c)>` in E; `w_z` is built on device from the twelve tensor factors of 9.1 for F-valued columns and the single `Mon(x) L(r)` product for quotient coordinate columns |
| `fold` | stored, beta → fold_y | one per slot | GEMV over all columns of all three trees |
| `query_gather` | code, tree, S → proof bytes | one per query, then one per frontier node | opened leaf rows and a Merkle multiproof: the unique sibling frontier of S is computed first, each sibling emitted once |
| `expected_symbols` | y_next, G rows at S → v_l | one per (query, coordinate) | the values the previous level must match at the opened positions (9.3) |
| `tail_materialize` | tensor terms → w_tilde | one per slot | sum of the active claim batch: up to `12 P + 4 n_cw |S_1|` tensor terms at level 2, `|S_{l-1}|` plus the running claim later |
| `tail_round` | w~, y → s_i | one per row | Hadamard and reduce over all but one digit, three evaluations |
| `tail_fold` | y, r̄ → y_next | one per row | GEMV with the 8-column matrix |
| `tail_encode` | y → tail_code | one per (coord, butterfly) | e/4 independent F4 DFTs on coefficient data, no inverse; last pass writes the `(s, 8, e)` leaf-major layout |
| `lde` | coeff → lde | one per (column, line) | coset LDE of 10.2: twist by `g^i`, forward DFT; shares the `idft2` stages |
| `residual` | lde, tables → residual | one per row of G | the fused pass of statement-layer 5 over the linear and quadratic tables; milestone 1 runs it on synthetic families |
| `quotient` | residual → coeff of A, B, Q2 | one per row, then `idft2` | multiply by `−1/2`, one axis-1 LDE of `S1`, inverse 2D DFT to E coefficients, then `to_stored` in plain slots |

Milestone 1 ends with the W and Q trees, the residual and quotient on synthetic families, openings at P points, the tail, and the verifier, as the spec's build order says. Milestone 2 adds the Z tree: `factor`, `batch_invert`, `chain_scan`, the small grid and Q3. Milestone 3 adds `radix_sort`.

Stage order the host enqueues for the tail, per level: commit `Mat(y_l)` (`tail_encode`, `merkle`, transcript absorb) → squeeze `S_{l-1}` → `query_gather` on the previous level → `expected_symbols` → absorb `v` → squeeze batching scalars → `tail_materialize` → three times (`tail_round`, absorb, squeeze `r_i`) → `tail_fold`. The last level sends `y_ell` in the clear.

The first kernel written is `rs_encode`, because it decides prover time and tells us what Apple GPU support in Mojo can do.

## 5. Field arithmetic

`field.mojo` holds F127 on `UInt8` and SIMD lanes: add, sub, mul with one reduction, no branches. `F2` and `F4` are fixed-width arrays of coordinates with the tower constants `i^2 = −1`, `j^2 = c_2` chosen and checked at comptime. `E` is `e` coordinates, schoolbook multiply, inversion by exponentiation; zero inversion raises. The tower constants `c_k` are recorded in `docs/decisions.md` once chosen.

Twiddles are precomputed tables in device memory: the order-`L0` subgroup generator powers as 4×4 F-matrices for the F4 DFT, and the F2 roots for the grid DFTs. One table per `Params`, built once.

## 6. Transcript, host, verifier

- `transcript.mojo`: device-resident. A small buffer holds the running Blake3 state; one kernel absorbs a message (tree root, clear values, sumcheck messages) with the domain separator of 9.4, one kernel squeezes challenges into a device buffer: E elements as e bytes, positions as uniform integers below L. Every kernel that needs a challenge reads it from that buffer.
- `verifier.mojo`: host program, separate from the prover. The seven steps of statement-layer section 6 for milestone 1 reduced to the Ligerito checks; builds `w_z` from the twelve tensor factors; checks consistency at opened positions with the E ⊗ F4 alphabet rule of 9.1; runs the sumcheck checks per level.
- `proof.mojo`: the byte layout of statement-layer section 7. Exact encoding is fixed when the first proof is serialized.
- `prover.mojo`: host orchestration only. It enqueues every kernel of the pipeline in stage order on one stream and synchronizes once at the end to read the proof bytes. Barriers are ordering on the stream, not host synchronization points.

## 7. Layout

```
src/caracal7/
  params.mojo       Params, derived constants, twiddle tables
  field.mojo        F, F2, F4, E
  encode.mojo       idft2, to_stored, pack, rs_encode
  merkle.mojo       Blake3 tree
  open.mojo         open, fold
  tail.mojo         materialize, round, fold, encode
  transcript.mojo
  proof.mojo
  prover.mojo
  verifier.mojo
tests/              one file per module, TestSuite runner, scalar references inline
```

## 8. Kernel discipline

Every kernel follows the ladder of Boehm's matmul article (siboehm.com/articles/22/CUDA-MMM), which took a naive kernel from 1% to 94% of cuBLAS in ten steps. The steps, in the order they pay:

1. **Naive, correct, measured.** One thread per output. The scalar-reference test passes. Record GB/s and GMAC/s against the M1 Pro roofline (about 200 GB/s, fp32 peak in the low TFLOPs) before touching anything.
2. **Coalescing.** Consecutive threads of a SIMD group read consecutive bytes. This is a layout rule, not a kernel trick: every buffer shape in section 3 puts the index that threads walk fastest last. Gave 6× in the article.
3. **Threadgroup tiling.** Load a tile of inputs into threadgroup memory once and let every thread of the group reuse it. Apple threadgroup memory is 32 KB; tile sizes are comptime parameters of the kernel.
4. **Register tiling.** Each thread computes a `TM × TN` tile of outputs, not one. This is where arithmetic intensity comes from and where the article got its largest single step (2.8×, then another 1.9× for 2D tiles). For our GEMM-shaped stages (the F4 DFT butterflies, the fold, the residual pass) this is the step that decides whether we are compute or bandwidth bound.
5. **Vectorized loads.** `SIMD[UInt8, 16]` per thread: 16 F values, or one E element, per load. Store tiles transposed if that makes the inner loop vector-friendly.
6. **Bank conflicts, double buffering, warp tiling.** Only for the kernel that is the budget, and only after step 4 is measured. The article spent four weekends on the last 14%.
7. **Autotune.** Tile parameters are comptime; a small Mojo sweep over a handful of `(BM, BN, BK, TM, TN)` settings picks them per kernel, because the best values differ per GPU.

Stop at the rung where the kernel is no longer the budget. Only `rs_encode`, and later `residual`, should reach step 6.

**Arithmetic on bytes.** F values are 7 bits. A product is under 2^14, so a `UInt16` accumulator holds four products and an `Int32` holds thousands before a reduction. Reduce lazily: accumulate wide, reduce once per tile with `(x & 127) + (x >> 7)` twice, no division. F2 and F4 products are 4 and 16 base MACs on the same accumulators. This is the decision between fp32 and integer lanes in the open list: integer lanes with lazy reduction are the default until `rs_encode` shows fp32 wins.

**Shapes are GEMMs.** A radix-`r` DFT stage on `batch` lines is a GEMM with `M = r`, `K = r`, `N = batch` and a twiddle matrix as the constant operand; the fold is a GEMV over columns; the residual pass is a GEMM of the family table against the LDE rows. Write each with the same tile skeleton so the tuning work transfers.

## 9. Open

- Apple GPU in Mojo: shared-memory size, `barrier`, and whether Blake3 on device reaches the CPU rate. Learned from `rs_encode` and `merkle`.
- fp32 versus integer lanes for the GEMM stages. Section 10.2 assumes exact fp32 under a chunking rule; on Apple GPU integer SIMD may be the better path. Measure both in `rs_encode`.
- Whether `code` should be leaf-major from the encoder or transposed once (section 3).
- The tower constants `c_2`, `c_3`, `c_4` for `e = 16`.
