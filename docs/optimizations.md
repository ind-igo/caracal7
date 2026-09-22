# Optimization candidates

This page lists the prover, verifier and proof-size changes that a read of the code found, ranked by
gain divided by effort. No item here is measured. The gains are estimates from operation counts and the stage
times in `docs/profile.md` (M1 Pro, `CLIENT` profile). Measure each change A/B, back to back, before you keep it.
The last section lists the items that were tried and lost, so that nobody proposes them again.

## Prover, wide statements

Baseline: passport 1957 ms, SOD 1378 ms, RSA-2048 593 ms. On these statements the LDE and the residual are
about 45 percent of the prove. Together, the items below may save 400 to 500 ms on the passport. The
estimates overlap, so do not add them exactly.

| # | change | where | estimate (passport) | effort |
|---|---|---|---:|---|
| 1 | LDE of the public term columns as outer products | `prover.mojo:537`, `load_public` | ~140 ms | low to medium |
| 2 | Pre-decoded entries and 32-bit index math in `k_residual` | `residual.mojo:121-219` | 70-110 ms | low |
| 3 | Horner terms in the coefficient domain; Z LDE on 10 F2 lanes | `residual.mojo:143,228` | ~95 ms | medium |
| 4 | RS gather as a twisted 8-point DFT | `encode.mojo:255-296` | ~60 ms | low |
| 5 | Cancel the odd stage of `idft2` against the odd stage of `to_packed` | `encode.mojo:91-99, 612-651` | 55-60 ms | medium |
| 6 | Pair the real W columns as a + i b through `idft2` | `dft.mojo` | ~35 ms | medium |
| 7 | `tail_materialize`: one thread per F4 symbol | `tail.mojo:108-135, 166-183` | 25-30 ms | low |
| 8 | Encode Z on E lanes | `prover.mojo:509`, `encode.mojo:607` | ~20 ms | low |
| 9 | Open: halve the zero-row block | `open.mojo:337-360` | ~15 ms | medium |
| 10 | `k_fold`: 4 slots per thread, beta staged as fp32 | `open.mojo:812-830` | ~12 ms | low |

**1. Public term columns as outer products.** The passport has 285 W, 140 Z and 264 public columns, so the LDE
transforms 689 columns. 236 of the 264 public columns are one term: a row vector times a chain set. The
interpolant of such a term is `interp(R)(X1) * interp(C)(X2)`. Do 1-D DFTs of the rows onto G1 and of the chain
indicators onto G2, then one elementwise kernel writes the LDE as the sum of the products (414 terms, F2 MACs
over 3N points). This removes 264/689 of the 18 radix passes. It also removes the per-prove host tiling of
153 MB (`tile_values`, `ir.mojo:697`), its upload and its `idft2`. The verifier's `eval_terms` uses the same
identity. Do not cache the public LDE across proves: the `lde` region lives from `ST_LDE` to `ST_RES` and other
stages reuse it, so a cache costs about 0.9 GB more arena on the passport.

**2. `k_residual` descriptors.** For each entry and each thread, `u16` does two byte loads (about 17 byte loads
in total), `e_planes` and `to_f32` convert 20 bytes of kappa, and `LdePoint.at` multiplies in 64-bit `Int`,
which the Apple GPU emulates. Let `k_fold_alpha` / `k_merge_kappa` also write a 16-byte aligned int32 record
(`col * h2 * 3W`, the A and B offsets, the flags) and kappa as fp32 planes. The largest in-buffer index is
689 x 4032 x 480, less than 2^31. The `LdePoint` docstring gives the scale: 8 more integer ops per read cost
15 to 25 percent.

**3. Horner in the coefficient domain.** No family entry reads a Z column on RSA, SOD, DSC or the passport; only
`k_horner` reads Z. The Horner term is linear in Z, and lane `l = Z_2l + i Z_2l+1` is F2-linear (the `_z_read`
formula). Build the combined lanes on the Z coefficients (7 x N E products), run the LDE on 10 lane columns in
place of 140 byte columns, and let `k_horner` read one 20-byte value. Keep the current path for statements
whose entries read Z. A smaller first step combines only the lanes: 140 to 70 columns, about 39 ms.

**4. RS gather.** Each input is multiplied by all B2 = 8 powers `gA^(i k2)`, and each power is its own table
load (`encode.mojo:283-285`). Within one thread `i = i0 + 64 M q''` with `q'' < 8`, so
`acc[k2] = gA^(i0 k2) * DFT8(x_q'')` with root `W8^M`. Use `_dif8` and seven output twists; keep the old path
for B2 < 8. About 2160 to 900 lane ops per thread. The gather on W, Z and Q is about 107 ms. The tail encoder
also gains where b = 9. The operands stay inside the bounds of `_dif8` (|x| <= 190).

**5. Odd-stage cancellation.** `idft2` ends with the inverse m-point stage on the odd digit, and `to_packed` then
applies the forward m-point DFT with `rho = omega^(2^a)`. The two compose to a diagonal:
`stored[x, r] = m * omega^(-x r) * Y2[x][r]` for each axis, where Y2 is the output after the power-of-two stages.
Compute `stored` by a twisted gather in `k_to_stored`, and run the odd stages only for the `coeff` that the LDE
reads. This removes 3 m-stages (9, 9, 7) on W and Z. `dft_axis` needs split entry points, and the `h^-1` scale
and the slot index maps move.

**6. Paired real columns.** The first stage reads two byte planes as a + i b; a split like `k_coef_columns`
(e = 2) at the end makes the coefficients of each column. Six of seven stages run at half width. This combines
with item 5: the split then goes into the `to_stored` gather.

**7. `tail_materialize`.** Each slot loads the 4 batch E values of each query byte by byte (passport:
580608 x 103 x 80 byte loads) and computes `pw` and `bj * pw` again; `bj` is a unit, so a full F4 product is
wasted. One thread per F4 symbol shares `pw`; the batch goes into 8 KB of threadgroup memory. The arithmetic
floor is about 9 ms against 40 ms now.

**8. Encode Z on E lanes.** `k_values_to_trace` splits `zval` into 140 byte columns before `idft2`. Run
`dft_axis` on the 7 E-valued columns as 10 F2 lanes (W = 10, as quotient step 4 does with `E_DFT_V`), then
`k_coef_columns` (groups = 7) writes the 140 coordinate coefficient columns. The code exists; this also removes
a strided transpose kernel.

**9. Open zero-row block.** `a_zero` has 4Q rows, but each k has at most 2 nonzero rows: x2 < H2 fills only half
0 and x2 > H2 only half 1, with the same weights. Make the zero-row transpose half-major, put the fixed slots
x2 in {0, H2} in the correct half, and run one GEMM with 2Q rows. The zero block is 22 percent of the stage-1
MACs on the passport. The risk is in the bookkeeping of the fixed slots.

**10. `k_fold`.** 22 ms against a floor of about 7.5 ms (5.6e9 FMA). Each slot converts all 20 beta bytes for
each column and loads one byte per slot. Use 4 slots per thread with one u32 load, and stage beta as fp32 for
each column chunk.

Smaller items on the wide statements:

- `k_sum_splits` with one split still makes a byte-wide transpose copy of about 17 MB (`open.mojo:806`). Let
  `k_stage2` read the (n, row) layout directly: 3 to 5 ms.
- Swap the LDE axis order (`residual.mojo:389-396`): axis 2 first, then the three axis-1 transforms. 18 to 17
  passes, about 12 percent fewer MACs. Measure it, because the stage kernels differ per radix.
- Group kappa by (family, chal): accumulate in F2 and flush once per group with one E MAC. After item 2 the net
  gain is about 8 to 11 percent of `k_residual`, since the grouping loses the cross-family merge.
- Plan h1 = 144 as one radix-16 stage on the 8x8 op in place of two radix-4 lane stages. Watch the register
  spills (`_vcap`).
- Fuse `k_pack` into `k_to_stored` through a threadgroup tile. `k_pack` is a byte-by-byte transpose. Time
  `pack` alone first; `bench/bench_encode.mojo` prints it but has no passport grid
  (`run[CLIENT.grid(144, 4032), 285]`).

## Prover, hash statements

Baseline: sha256 81 ms, keccak 58 ms. The tail is about 35 percent of the prove, but launch latency is not the
cause: the sha256 tail makes about 200 launches, 1 to 3 ms. `prove` has no host sync before `proof.finish`.
Together the items below may take sha256 to 65 to 70 ms.

- **Parallel squeeze.** `k_sample` runs on one thread and reads each Blake3 block back from global memory byte
  by byte (`transcript.mojo:84-110, 199-203`), about 0.25 us per byte. Squeeze blocks are counter mode: one
  thread per block, a prefix sum of the accepted bytes, and a compaction give the same bytes as the serial code,
  testable against `HostTranscript`. 3 to 5 ms on every statement. The same one-block merge can replace the
  serial thread-0 merge in `k_absorb_tree` (`transcript.mojo:128-131`).
- **Tail rounds and fold.** `k_fold8` computes `rbar_at` again for each row, 16 of its 24 E products
  (`tail.mojo:306-316`); round 0 multiplies by one. Compute `rbar[8]` once. Do the block reduction inside
  `k_round_partial`, and put the 64-row sum, the absorb and the squeeze in one kernel: 5 to 2 launches per round.
  About 2 ms on sha256, 5 to 9 ms on the passport.
- **Merkle frontier.** Level-1 W, Z and Q use the same positions and leaf count, but `k_frontier` runs once per
  tree (`merkle.mojo:77-166`, `prover.mojo:641-646`), 0.38 ms each on one block. Compute it once, and replace
  the O(m) scans with a prefix sum. 1.5 to 2 ms on sha256.
- **Merkle top levels.** One block does all levels below 512 nodes with barriers (`merkle.mojo:68-74`): 60 to 90
  fewer launches, 1 to 2 ms.
- **Grind threads.** `k_grind` runs 8192 x 1 (`backend.mojo:72`); 16384 and 32768 are not measured.
  0.3 to 0.5 ms per search.
- **Transcript prefix.** Each prove hashes sigma, pubf, the families and the tables on the host
  (`prover.mojo:456-466`, `proof.mojo:463`). Cache the digests at setup. Under 1 ms.

Grinding costs 1.7 to 2.1 ms per search, one search per level plus one: about 9 to 10 ms on sha256 and keccak
(12 to 16 percent). The kernel runs at about half of the ALU peak. Fewer grind bits is a ledger decision
(`docs/soundness.md`), not an engineering change.

Open questions for the next profile: `running0` (`prover.mojo:436`) has no mark, so its time is inside "tail
encode 0" (46 ms on the passport, about 4 ms estimated for `running0`). `finish` takes 7 ms on the passport for a
1.15 MB copy. The tail RS encoder at 32 columns costs about twice as much per column as the W encoder.

## Verifier

Baseline: passport 862 ms, RSA 325 ms. None of these items changes a checked value, so the ledger does not
change. Together they may take the passport to about 120 ms.

- **Boundaries (passport 240 ms, RSA 160 ms).** `horner_chain_end` (`ir.mojo:418-443`) decodes each `entry()`
  again for every row, and does one E x E product by the challenge for each row and entry, plus 143 E x E
  products by `scale`. RSA-2048 has about 1100 public factors. Move the entry decode out of the row loop and
  use the linear form `R = scale^(h1-1) start - sum_i chal_i sum_x1 scale^(h1-2-x1) v_i(x1)`, with one power table
  per accumulator. A factor then costs 4 to 12 E x E products plus F x E work. `parallelize` the factor loop.
  `verifier.mojo:259` also calls `selector_values` once per factor, and `ir.mojo:545` runs `column_offsets`
  again each time; pass the `offs` that `_check_statement` already holds, and skip the selector when a factor
  has no `col_b`. `tests/test_accumulate.mojo:322,376` compares this function with the device.
- **Clear vector (passport 284 ms).** `clear_value` (`tensor.mojo:370-405`) costs units x (m1 m2 + m2) E x E
  products per index. `parallelize` over the units (8 performance cores); replace `_scal` (an `f_pow` per
  entry) by a 126-entry table per rho; group the units by their folded s2, so a unit costs m1 = 9 products in
  place of 630. The consistency and row units have s1 and s2 in F4, so use `e_scale`.
- **Small grid (passport 68 ms, SOD 108 ms).** `interp_cyclic` (`smallgrid.mojo:263`) does one `ext_inv` per
  node on every call, about 48K inversions on the passport. Compute one Lagrange vector on G2 at z2 with one
  batch inversion; the H2 vector comes from the even-index denominators, and `L(omega2 z2)` is a rotation of
  it. The batch inversion must still raise when z2 is in G2. Make one table of alpha powers in place of
  `ext_pow(alpha, family)` per term (`verifier.mojo:300,316,318`, `ir.mojo:594`).
- **Tail and running claim (passport 85 + 57 ms).** Parallelize `Unit.fold` (`tensor.mojo:71`). `host_table`
  (`open.mojo:132`) calls `table_entry` per entry, 80 `ext_pow` from zero plus one inversion per r: build the
  powers step by step and batch the 72 denominators. Parallelize the 53 points in `_Tail.__init__`
  (`verifier.mojo:387`).
- **Hash statements.** `eval_values` (`ir.mojo:652`) builds both Lagrange vectors again for each (dense column,
  point); cache them as the term path does at `ir.mojo:682`. About 10 ms of sha256's 39 ms.
- **E arithmetic.** Every `ext_mul` converts u8 to f32, multiplies and canonicalizes. A multiply-accumulate with
  lazy reduction (centered operands, a reduction every 8 terms, inside the bound of `fp_e_mul`) gives about 1.5
  to 2x on the loops that remain.

## Proof size

- **7-bit packing.** Coordinates are uniform in F127, so 8 values fit in 7 bytes. The passport has about 990 KB
  of field bytes, so this saves about 124 KB (11 percent), and about the same fraction on every workload. The
  change stays in `ProofWriter` / `ProofReader.field_bytes` and the multiproof rows; the transcript still
  absorbs the unpacked bytes, so the device does not change. The reader must still reject 127. No soundness
  change.
- **Commit Z2 and Q3.** Put each Z2 line in the Z tree as an E column constant along X1, and Q3 in the Q tree as
  two halves of degree h2. The cost is 5 single-value opened columns and about 100 coordinate columns at 103
  level-1 queries, 50 to 90 KB, against 403 KB now: about -320 KB on the passport (28 percent), -220 KB on the
  SOD and DSC. The prover gains about 110 ms of encode on the passport, and the verifier loses its small-grid
  interpolations. This changes P2 and the small-grid term: recompute `docs/soundness.md`.
- **Fewer opened pairs.** Research only. The batched claim needs every (column, point) pair, and an aggregate of
  the unread pairs sent after beta is unsound. Batching by point class changes P4. Count the pairs that are read
  first; the gain is about 280 KB if 10 percent are read.
- **Fewer points.** Each point costs 295 x 20 = 5.9 KB on the passport. The points come from the cyclic-read
  offsets of the RSA limb lanes (`docs/rsa.md`).

Not worth it: Merkle sibling dedup (`check_multiproof` shares siblings already), one tree for W, Z and Q (they
are committed at different barriers), shorter digests (below the collision target), and dropping `Z2(1)`
(60 bytes).

## Tried and lost, or rejected

- The GEMM skeleton for the residual lost at M = 8. One body per branch in `k_residual`: -15 percent. A full
  `lde_index` per read, the two-block LDE layout and unpadded thirds: each -15 to -25 percent.
- Fused lane DFT stages lost 2 to 3x; the dense 8x8 op won. Gather64 fusion lost (41 against 31 ms). Radix 7 and
  9 are at their memory limit; radix 9 as 3 x 3 gained nothing.
- A `TileTensor` register tile is 3x slower than SIMD locals.
- Fusing the LDE into the residual: the reads span 798 (column, shift) pairs, and the axis-2 transform of length
  4032 is global. Fusing only the last radix stage multiplies the gathers by 7 to 9.
- Fusing the Merkle leaves into the last RS pass: a W leaf spans 9 column blocks, and 63 rows need 72 KB of the
  32 KB threadgroup memory. Merkle is near its Blake3 compute floor.
- The LDE of Z on E lanes, without item 3: `k_residual` reads the Z coordinate columns one by one.
- A smaller code domain: 4 x 161280 is the only domain that fits the passport.

## Order

1. Items 1, 2, 4 and 8: the best gain for the effort, each testable against the scalar references.
2. The verifier boundaries and the clear vector.
3. The 7-bit proof packing.
4. The parallel squeeze and the tail items, for the hash statements.

Then refresh `docs/profile.md` with `bench/bench_breakdown.mojo`, and the README tables with the Criterion
harness.
