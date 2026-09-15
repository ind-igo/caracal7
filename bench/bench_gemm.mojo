"""The F2 GEMM skeleton (backend.gemm_f2) at 1024^3 on plain strided operands: ms/iter and GMAC/s,
one F2 MAC = four byte MACs. Tune tiles here; every GEMM-shaped stage inherits the result."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.core.arena import Arena, Bump
from caracal7.core.backend import BACKEND, F4_TILE, Tile, Strided, Strided4, launch_gemm_f2, launch_gemm_f4, strided

comptime M = 1024
comptime N = 1024
comptime K = 1024
comptime REPS = 10


def report(name: String, ns: Int):
    var macs = Float64(M) * Float64(N) * Float64(K) * Float64(REPS)
    print(name, ": ", Float64(ns) / 1e6 / REPS, " ms/iter, ", macs / Float64(ns), " GMAC/s (F2), ",
          4 * macs / Float64(ns), " GMAC/s (byte)")


def run[T: Tile](ctx: DeviceContext, arena: Arena, a: Int, b: Int, c: Int, name: String) raises:
    var o = strided(a=a, sa_m=K * 2, sa_k=2, b=b, sb_k=N * 2, sb_hi=2, sb_lo=0, c=c, sc_m=N * 2, sc_hi=2, sc_lo=0)
    launch_gemm_f2[BACKEND, T, Strided, 1](ctx, arena, o, M, N, K)   # warm-up / compile
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        launch_gemm_f2[BACKEND, T, Strided, 1](ctx, arena, o, M, N, K)
    ctx.synchronize()
    report(name, Int(perf_counter_ns() - t0))


def run4[T: Tile](ctx: DeviceContext, arena: Arena, a: Int, b: Int, c: Int, name: String) raises:
    var o = strided(a=a, sa_m=K * 4, sa_k=4, b=b, sb_k=N * 4, sb_hi=4, sb_lo=0, c=c, sc_m=N * 4, sc_hi=4, sc_lo=0)
    launch_gemm_f4[BACKEND, T, Strided4, 1](ctx, arena, o, M, N, K)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        launch_gemm_f4[BACKEND, T, Strided4, 1](ctx, arena, o, M, N, K)
    ctx.synchronize()
    var ns = Int(perf_counter_ns() - t0)
    var macs = Float64(M) * Float64(N) * Float64(K) * Float64(REPS)
    print(name, ": ", Float64(ns) / 1e6 / REPS, " ms/iter, ", macs / Float64(ns), " GMAC/s (F4), ",
          16 * macs / Float64(ns), " GMAC/s (byte)")


def main() raises:
    var ctx = DeviceContext()
    print(ctx.name(), " M=N=K=", M)
    var bump = Bump()
    var a = bump.alloc(M * K * 4)
    var b = bump.alloc(K * N * 4)
    var c = bump.alloc(M * N * 4)
    var arena = Arena(ctx, bump.used)
    var h = ctx.enqueue_create_host_buffer[DType.uint8]((M * K + K * N) * 4)
    ctx.synchronize()
    for i in range(len(h)):
        h[i] = UInt8((i * 37 + 11) % 127)
    arena.upload(ctx, a, h)
    run[BACKEND.tile](ctx, arena, a, b, c, "default " + String(BACKEND.tile.BM) + "/" + String(BACKEND.tile.BN) + "/" + String(BACKEND.tile.BK))
    run[Tile(BM=64, BN=64, BK=8, TM=4, TN=4, mma=False)](ctx, arena, a, b, c, "64/64/8 4x4")
    run[Tile(BM=64, BN=64, BK=16, TM=4, TN=8, mma=False)](ctx, arena, a, b, c, "64/64/16 4x8")
    run4[F4_TILE](ctx, arena, a, b, c, "F4 default 32/32/8 2x2")
    run4[Tile(BM=64, BN=64, BK=8, TM=4, TN=4, mma=False)](ctx, arena, a, b, c, "F4 64/64/8 4x4")
    run4[Tile(BM=32, BN=64, BK=8, TM=2, TN=4, mma=False)](ctx, arena, a, b, c, "F4 32/64/8 2x4")
    run4[Tile(BM=32, BN=32, BK=16, TM=2, TN=2, mma=False)](ctx, arena, a, b, c, "F4 32/32/16 2x2")
