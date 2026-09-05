# Design: memory, kernels, and boundaries

Spec: `wiki/projects/caracal7/specs/caracal-prover.md` (main), `statement-layer.md` (frontend), `polynomial-mulmod.md`. This page fixes what the spec leaves open: how bytes sit in memory, which kernels exist, and where CPU and GPU meet. Section numbers below refer to the main spec.

## 1. Rules

1. **Device resident.** Every prover stage is a GPU kernel from its first commit, including the Blake3 transcript and challenge derivation (section 10: tree and transcript on device). The host only enqueues launches in stage order and reads the finished proof bytes. The verifier is a separate program and runs on the host.
2. **One buffer type.** All committed data is bytes: one `UInt8` per `F` coordinate. `F2`, `F4`, and `E` values are 2, 4, and `e` consecutive bytes. No struct-of-arrays and no per-field buffer types. Kernels are specialized on the coordinate count, not on a field type.
3. **Parameters are comptime.** One `Params` struct carries every knob. Kernels take it as a parameter, so every loop bound and stride is a constant. Exceptions: the GEMM skeleton of section 9 takes `M, N, K` and its operand strides at runtime, because one kernel serves every GEMM-shaped stage (2026-09-04); the RS encoder takes the domain (`2^b`, `M`, cosets) and the message length at runtime and the tail kernels take the level sizes and query counts at runtime, because the tail schedule is derived at runtime from `Params` and one kernel serves every level (2026-09-05). Tile shapes, reduction cadences, radices and stage strides stay comptime.
4. **No copies between stages.** Unified memory on M1 makes host and device views of one buffer cheap. A stage writes its output where the next stage reads it. There are no host reads between barriers: challenges are derived on device from the device-resident transcript, and kernels read them from device memory.
5. **Scalar reference in the test, never a CPU prover.** Each kernel's test runs a few lines of scalar Mojo on a small size and compares. The reference does not grow into a second implementation.
6. **Allocate once, at setup.** One device arena per prover instance, sized from `Params` and the IR program; every buffer of section 3 is an offset into it, assigned by a bump pointer at setup. Fixed ceilings for columns per tree, P, tail levels, and queries size the arena. No allocation after setup, and the arena is reused across proofs with the same profile. The proof output and host staging buffers follow the same rule. Threadgroup memory and register tiles are static per kernel already.

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
| `stored` | (column, slot) | mixed basis, N bytes per column, the Frobenius-real slots of 9.1 for every column; the quotient coordinate columns (A, B, Q2 as values on H, 9.2) are ordinary columns and enter the encoder as a trace |
| `packed` | (i, column, 4) | F4 symbols, N/4 per column, column fastest so the RS passes are coalesced |
| `code` | (s, column, 4) | codeword rows, leaf-major: leaf `s` is contiguous, `4 * columns` bytes (`n_cw = 1`; the split multiplies the row) |
| `tree` | (level, node, 32) | Blake3 digests, leaves first |
| `num`, `den`, `zval` | (row, e) | the Z stage per accumulator: N, D, then Z; `zval` is split into e coordinate columns as the Z tree's trace (accumulate.mojo) |
| `z2` | (accumulator, x2, e) | Z2 in the clear |
| `q3` | (2 h2, e) | Q3 on G2 in the clear (smallgrid.mojo) |
| `lde` | (column, G2, G1, 2) | evaluations on the residual grid G: witness columns, then the accumulator coordinate columns (F-valued, so 2 coordinates each) |
| `residual` | (G2, G1, e) | batched residual, E-valued |
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
| `idft2` | trace → coeff | F2 GEMM skeleton, one launch per axis | inverse 2D DFT over F2, dense per axis with the inverse tables as the constant operand (mixed radix when `h_l` grows) |
| `to_stored` | coeff → stored | one per (column, slot) | length-m DFT over F per axis on the odd digit, then the Frobenius-real slot bijection of 9.1 |
| `pack` | stored → packed | one per (column, i) | gather 4 slots on the packing digit into one F4 symbol |
| `rs_encode` | packed → code | pass A on the F4 GEMM skeleton; radix stages one per (column, butterfly) | coset twist by `g_k^i` inside the pass-A loader, then the order-L0 DFT over F4 in two passes: the power-of-two part as one F4 GEMM per Good-Thomas line (`wa` table × message rows), then the odd-part radix stages as register butterflies |
| `merkle` | code → tree | one per node per level | Blake3, 1,024-byte leaves, 32-byte nodes, one launch per level |
| `open` | stored, w_z → alpha | lane GEMM on the skeleton | contraction `<w_z, stored(c)>` in E for every column of every tree; `w_z` is built on device per (point, slot) from the pieces of 9.1 (`slot_weight`, shared with the verifier) |
| `fold` | stored, beta → fold_y | one per slot | GEMV over all columns of all three trees |
| `query_gather` | code, tree, S → proof bytes | one per query, then one per frontier node | opened leaf rows and a Merkle multiproof: the unique sibling frontier of S is computed first, each sibling emitted once |
| `tail_materialize` | tensor terms → w_tilde | one per slot | sum of the active claim batch: up to `12 P + 4 n_cw |S_1|` tensor terms at level 2, `|S_{l-1}|` plus the running claim later |
| `tail_round` | w~, y → s_i | one per row | Hadamard and reduce over all but one digit, three evaluations |
| `tail_fold` | y, r̄ → y_next | one per row | GEMV with the 8-column matrix |
| `tail_encode` | y → tail_code | the RS encoder on 32 F4 columns | the 8 E-valued columns are 32 F4 columns to `rs_encode_on`, no inverse; its scatter pass writes the `(s, 8, e)` leaf-major layout |
| `lde` | coeff → lde | GEMM skeleton, one launch per axis | forward DFT onto `G` with the twist inside the `g_l^(j k)` tables (dense; mixed radix when `h_l` grows) |
| `accumulate` | trace, gamma → zval, z2 | one thread per chain; one thread for Z2 | factors N, D per row, batched inversion and the running product along each chain, Z2 across chains (spec 10 steps 4-6) |
| `small grid` | z2, chain ends, alpha → q3 | one thread per coefficient or point, five tiny launches per accumulator | R2 in coefficient form (inverse DFT of the five line vectors, products, the gate), Q3 = R2 / (X2^h2 - 1) as a two-term sum per coefficient, evaluated on G2 (spec 7.4) |
| `residual` | lde, tables → residual | the GEMM skeleton, `C[8 lanes, point]` | the fused pass of statement-layer 5 as one GEMM of the kappa table against family rows gathered from the LDE; milestone 1 runs it on synthetic families |
| `quotient` | residual → trace of A, B, Q2 | GEMM skeleton launches | `Q1` on the coset from `R` over `G1`, inverse DFTs to the `A`, `B`, `Q2` coefficients, forward DFTs to their values on `H`; the coordinate columns then take the witness encoder path |

Milestone 1 ended with the W and Q trees, the residual and quotient on synthetic families, openings at P points, the tail, and the verifier, as the spec's build order says. Milestone 2 adds the Z tree (`accumulate`, committed through the same encoder and Merkle path, alpha after it), the seven spec points, and the small grid with Q3. Milestone 3 adds `radix_sort`.

Stage order the host enqueues for the tail, per level: commit `Mat(y_l)` (`tail_encode`, `merkle`, transcript absorb) → squeeze `S_{l-1}` → `query_gather` on the previous level → squeeze batching scalars → `tail_materialize` → three times (`tail_round`, absorb, squeeze `r_i`) → `tail_fold`. The last level sends `y_ell` in the clear.

The first kernel written is `rs_encode`, because it decides prover time and tells us what Apple GPU support in Mojo can do.

## 5. Field arithmetic

`field.mojo` holds F127 on `UInt8` and SIMD lanes: add, sub, mul with one reduction, no branches. `F2` and `F4` are fixed-width arrays of coordinates with the tower constants `i^2 = −1`, `j^2 = c_2` chosen and checked at comptime. `E` is `e` coordinates, schoolbook multiply, inversion by exponentiation; zero inversion raises. The tower constants `c_k` are recorded in `docs/decisions.md` once chosen.

Twiddles are precomputed tables in device memory: the order-`L0` subgroup generator powers as 4×4 F-matrices for the F4 DFT, and the F2 roots for the grid DFTs. One table per `Params`, built once.

## 6. Transcript, host, verifier

- `hash.mojo`: the `Hash` trait (leaf, node, absorb, squeeze; `DIGEST`, `BLOCK`). Merkle, transcript, prover, and verifier take `H: Hash` as a comptime parameter next to `Params`; Blake3 is the first implementation, and a second one (for recursion) changes nothing above the trait.
- `transcript.mojo`: device-resident. A small buffer holds the running Blake3 state; one kernel absorbs a message (tree root, clear values, sumcheck messages) with the domain separator of 9.4, one kernel squeezes challenges into a device buffer: E elements as e bytes, positions as uniform integers below L. Every kernel that needs a challenge reads it from that buffer.
- `verifier.mojo`: host program, separate from the prover. The seven steps of statement-layer section 6 for milestone 1 reduced to the Ligerito checks; builds `w_z` from the twelve tensor factors; checks consistency at opened positions with the E ⊗ F4 alphabet rule of 9.1; runs the sumcheck checks per level. (Milestone 1 evaluates the queries directly, `O(|y_l|)` per level, and materializes `w~` as a vector; the tensor form is the verifier's own ladder.)
- `proof.mojo`: the byte layout of statement-layer section 7. Exact encoding is fixed when the first proof is serialized.
- `prover.mojo`: host orchestration only. It enqueues every kernel of the pipeline in stage order on one stream and synchronizes once at the end to read the proof bytes. Barriers are ordering on the stream, not host synchronization points. `prove(profile=True)` is the measurement mode: a synchronize after every stage and the stage times recorded; `bench/bench_prover.mojo` prints the split.

## 7. Layout

```
src/caracal7/
  params.mojo       Params, derived constants
  field.mojo        F, F2, F4, E
  backend.mojo      Backend, tile_mac, the F2 and F4 GEMM skeletons and their operand loaders (section 9)
  tables.mojo       generators, twiddle tables (host, setup)
  arena.mojo        the one device allocation (rule 6)
  encode.mojo       idft2, to_stored, pack, rs_encode
  hash.mojo         Hash trait, Blake3
  merkle.mojo       tree, query_gather (multiproof)
  accumulate.mojo   the Z stage: factors, batched inversion, chain scan, Z2
  smallgrid.mojo    R2 and Q3 in coefficient form; the verifier's cyclic interpolation
  residual.mojo     lde, residual, quotient
  open.mojo         build_queries, open, fold
  tail.mojo         tail_encode, materialize, round, fold
  transcript.mojo   device-resident absorb / squeeze
  proof.mojo        Shape, tail schedule, byte layout, writer / reader
  prover.mojo       arena plan (ProverLayout) and the stage order (Prover.prove)
  verifier.mojo     host program
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

The ceiling is in the design, the polish is by measurement. Every layout, skeleton, and reduction rule above is chosen for peak throughput from the first commit and is not revisited per kernel. The per-kernel climb up the ladder is ordered by measurement, so that the kernels which are the budget (`rs_encode`, then `residual`) get the expensive rungs first, and no kernel is polished before a measurement says it is on the critical path. Nothing is left slow on purpose; work is ordered by where the time goes.

**Arithmetic on bytes.** F values are 7 bits. A product is under 2^14, so a `UInt16` accumulator holds four products and an `Int32` holds thousands before a reduction. Reduce lazily: accumulate wide, reduce once per tile with `(x & 127) + (x >> 7)` repeated until the value is below 254 (three rounds for a 32-bit accumulator of up to 2^20, two for 16 bits), then map 127 to 0; no division. Verified on the M1 GPU with a 64 × 64 × 64 byte GEMM against a scalar reference (2026-09-04). F2 and F4 products are 4 and 16 base MACs on the same accumulators. This is the decision between fp32 and integer lanes in the open list: integer lanes with lazy reduction are the default until `rs_encode` shows fp32 wins.

**Shapes are GEMMs.** A radix-`r` DFT stage on `batch` lines is a GEMM with `M = r`, `K = r`, `N = batch` and a twiddle matrix as the constant operand; the fold is a GEMV over columns; the residual pass is a GEMM of the family table against the LDE rows. Write each with the same tile skeleton so the tuning work transfers.

## 9. Backends

The design is backend agnostic. Apple and NVIDIA differ in the one place that matters, the inner product of a tile, and that difference is a comptime configuration, not a second code path.

**What differs.** NVIDIA has integer tensor cores: an `s8 × s8 → s32` MMA, exact for our 7-bit values, at several times the fp32 rate, reachable through `layout.tensor_core.TensorCore` (dtype-parametric, NVIDIA and AMD). Apple M5 and later (GPU family 10) have the same in `linalg.arch.apple.mma.MmaOpApple`: 16 × 16 × 16 tiles, `int8 → int32` instantiated, one simdgroup of 32 threads per op. Apple M1 to M4 have no matrix op reachable from Mojo: every MMA instantiation fails pipeline creation on the M1 Pro with "simdgroup_matrix operations are supported by GPUFamily10 and later" (probed 2026-09-04, int8, fp16, and fp32 inputs). The ethproofs host is an M1, so the client target runs on SIMD lanes only. Apple threadgroup memory is 32 KB, NVIDIA 48 to 228 KB. Both have 32-wide SIMD groups. So the tile op, the accumulator type, the reduction cadence, and the tile sizes differ; nothing else does.

**The configuration.** One comptime struct, selected once by `is_apple_gpu()` / `is_nvidia_gpu()` and passed as a parameter to every kernel:

```mojo
struct Backend:
    comptime simd_width: Int          # 32 on both
    comptime threadgroup_bytes: Int   # 32 KB Apple, 48 KB+ NVIDIA
    comptime mma: Bool                # tensor-core tile op available
    comptime mma_m: Int; comptime mma_n: Int; comptime mma_k: Int
    comptime in_dtype: DType          # int8 for MMA backends, uint8 on SIMD lanes
    comptime acc_dtype: DType         # int32 for MMA backends, uint16 or int32 on SIMD lanes
    comptime max_terms: Int           # products accumulated before a mod-127 reduction: 4 in uint16, 2^17 in int32
    comptime vec_bytes: Int           # bytes per thread load, 16
    comptime tile: TileParams         # BM, BN, BK, TM, TN defaults, overridden by the autotune sweep
```

**The one hot op.** `tile_mac[B: Backend](a_tile, b_tile, acc)`: multiply a `(mma_m × mma_k)` tile by a `(mma_k × mma_n)` tile and accumulate. Three implementations behind one signature: NVIDIA `TensorCore` int8 MMA, Apple M5 `MmaOpApple` int8 MMA, and SIMD lanes with lazy integer reduction (the M1 path, verified with a 64 × 64 × 64 byte GEMM). The GEMM skeleton of section 8, the buffer shapes, the transcript, and every kernel above the tile op are shared. The SIMD-lane path is also the reference the two MMA paths are tested against. Both MMA structs are importable from the stable toolchain (probed), though `MmaOpApple` lives under `linalg` and is not a documented public surface; pin the toolchain version when it is used.

**Consequences for the DFT.** MMA shapes fix `K`; a radix-`r` stage with `r` in {2, 3, 5, 7} does not fill `K = 16` or 32. Stages are therefore grouped into one matrix per pass whose order is a product of radices near the MMA `K` (for L0 = 80,640 = 2^8 · 315: the 315-point Good–Thomas pass is one 315 × 315 matrix applied as 20 blocks of 16, or three radix stages 5 · 7 · 9 padded per backend). Which grouping wins is per backend and comes out of the autotune sweep, not the design.

**Rule.** No kernel contains `is_apple_gpu()` or `is_nvidia_gpu()` directly. All dispatch goes through `Backend`. A kernel that needs something the struct does not carry adds a field to the struct.

Materialized 2026-09-04 in `backend.mojo`: `Backend`, `tile_mac` (SIMD lanes, `mma_k = 1`) and the F2 GEMM skeleton `gemm_f2`; the residual stage is written on it, the plain GEMM shares the tile op. Verified 2026-09-04: there is no single MMA surface. NVIDIA and AMD go through `layout.TensorCore`; Apple M5 goes through `linalg.arch.apple.mma.MmaOpApple`; the M1 has neither. `tile_mac` wraps all three plus the SIMD-lane path.

## 10. Open

- Apple GPU in Mojo: shared-memory size, `barrier`, and whether Blake3 on device reaches the CPU rate. Learned from `rs_encode` and `merkle`.
- The SIMD-lane tile op is the only path on the M1 client target, so it gets the full ladder of section 8. The MMA paths are written when an NVIDIA or M5 device is available to test on.
- Whether `code` should be leaf-major from the encoder or transposed once (section 3).
- The tower constants `c_2`, `c_3`, `c_4` for `e = 16`.
