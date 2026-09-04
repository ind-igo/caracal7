# Decisions

Implementation decisions that the spec does not settle. Dated, newest last.

- 2026-09-04: uv with `max[all]` from PyPI (stable, Mojo 1.0.0); the pixi setup was replaced at Indigo's request. Tests run with `mojo run -I src`, one `TestSuite` runner per file, since `mojo test` no longer exists.
- 2026-09-04: Tile op per backend. Probes on the M1 Pro with Mojo 1.0.0: a byte GEMM mod 127 with threadgroup memory and int32 accumulation works (0 mismatches on 64 × 64 × 64; three reduction rounds needed for a 32-bit accumulator). Every `MmaOpApple` instantiation (int8, fp16, fp32) fails at pipeline creation: `simdgroup_matrix` needs GPU family 10 (M5). `layout.TensorCore` (NVIDIA/AMD) and `MmaOpApple` both import from the stable toolchain. Decision: `Backend` selects among NVIDIA MMA, M5 MMA, and SIMD lanes; the M1 client target is SIMD lanes only.
