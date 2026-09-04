# Decisions

Implementation decisions that the spec does not settle. Dated, newest last.

- 2026-09-04: uv with `max[all]` from PyPI (stable, Mojo 1.0.0); the pixi setup was replaced at Indigo's request. Tests run with `mojo run -I src`, one `TestSuite` runner per file, since `mojo test` no longer exists.
- 2026-09-04: Tile op per backend. Probes on the M1 Pro with Mojo 1.0.0: a byte GEMM mod 127 with threadgroup memory and int32 accumulation works (0 mismatches on 64 × 64 × 64; three reduction rounds needed for a 32-bit accumulator). Every `MmaOpApple` instantiation (int8, fp16, fp32) fails at pipeline creation: `simdgroup_matrix` needs GPU family 10 (M5). `layout.TensorCore` (NVIDIA/AMD) and `MmaOpApple` both import from the stable toolchain. Decision: `Backend` selects among NVIDIA MMA, M5 MMA, and SIMD lanes; the M1 client target is SIMD lanes only.
- 2026-09-04: Tower constants for e = 16 (`field.mojo`): i^2 = -1, j^2 = 2 + i, u^2 = j, y^2 = u. Each is a non-square in the field below (2 + i has norm 5, a non-residue mod 127; j and u follow from the (q-1)/4 argument); tests check C1..C3 by exponentiation. Elements are byte lanes `SIMD[uint8, 2^k]`, low half on 1, high half on the generator, so F4 acts coordinate-wise on E. Reductions use unsigned `min(x, x - 127)` instead of a select, because scalar comparisons return `Bool` in Mojo 1.0.
- 2026-09-04: `Params` is a trivially-register-passable struct passed as a comptime value; `REFERENCE` is the milestone-1 profile (72 × 32, L0 = 80,640, 105 queries at rate 1/140).
- 2026-09-04: `gemm.mojo`, byte GEMM mod 127 on the M1 Pro at 1024^3: naive 170 GMAC/s; threadgroup + register tiling (64/64/16, 4 x 4 per thread) 296 GMAC/s; BK = 8 gives 307. Register tiles must be `SIMD[int32, TM*TN]` locals with a runtime k loop: a `TileTensor` stack allocation with a fully unrolled k loop ran 3x slower than naive. A 128 x 128 block with 8 x 8 register tiles crashes the Metal compiler (XPC_ERROR_CONNECTION_INTERRUPTED at pipeline creation); tile sweeps stay at or below 4 x 4 per thread on this GPU until that is understood.

## GEMM rung 5: vector loads, per-row accumulators (2026-09-04)

`gemm127_vec` loads 4 bytes per thread per read from global memory, stores the A tile transposed in
shared memory so the register loads are 4-byte vectors too, and keeps the accumulator as TM separate
`SIMD[int32, TN]` rows. The single `SIMD[int32, TM*TN]` accumulator was the Metal compiler crash
(XPC_ERROR_CONNECTION_INTERRUPTED): with per-row vectors 4x8 and 8x8 register tiles compile.

Measured at 1024^3 on M1 Pro: naive 174, tiled 4x4 339, vec 4x4 360, vec 4x8 220, vec 8x8 157 GMAC/s.
Larger register tiles lose occupancy on this GPU, so the M1 backend keeps 64/64/16 with 4x4 and the
vec kernel. BK=32 also collapses (145). The next rungs on the ladder (double buffering, warp tiling)
wait until the encoder shows where the time goes.
