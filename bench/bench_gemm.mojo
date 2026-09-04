"""Byte GEMM mod 127 on the GPU: GMAC/s for each ladder rung. Small by default (1024^3)."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from caracal7.gemm import launch_naive, launch_tiled, launch_vec, TILE_DEFAULT, Tile

comptime M = 1024
comptime N = 1024
comptime K = 1024
comptime al = row_major[M, K]()
comptime bl = row_major[K, N]()
comptime cl = row_major[M, N]()
comptime REPS = 10


def report(name: String, ns: Int):
    var macs = Float64(M) * Float64(N) * Float64(K) * Float64(REPS)
    var bytes = Float64(M * K + K * N + M * N) * Float64(REPS)
    print(name, ": ", Float64(ns) / 1e6 / REPS, " ms/iter, ",
          macs / Float64(ns), " GMAC/s, ", bytes / Float64(ns), " GB/s (unique bytes)")


def main() raises:
    var ctx = DeviceContext()
    print(ctx.name(), " M=N=K=", M)
    var a_dev = ctx.enqueue_create_buffer[DType.uint8](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.uint8](K * N)
    var c_dev = ctx.enqueue_create_buffer[DType.uint8](M * N)
    var a_host = ctx.enqueue_create_host_buffer[DType.uint8](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.uint8](K * N)
    ctx.synchronize()
    for i in range(M * K):
        a_host[i] = UInt8((i * 37 + 11) % 127)
    for i in range(K * N):
        b_host[i] = UInt8((i * 53 + 7) % 127)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    var A = TileTensor(a_dev, al)
    var B = TileTensor(b_dev, bl)
    var C = TileTensor(c_dev, cl)

    launch_naive[M, N, K](ctx, A, B, C)   # warm-up / compile
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        launch_naive[M, N, K](ctx, A, B, C)
    ctx.synchronize()
    report("naive       ", Int(perf_counter_ns() - t0))

    launch_tiled[M, N, K, TILE_DEFAULT](ctx, A, B, C)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(REPS):
        launch_tiled[M, N, K, TILE_DEFAULT](ctx, A, B, C)
    ctx.synchronize()
    report("tiled 64/16/4", Int(perf_counter_ns() - t0))

    comptime T2 = Tile(BM=64, BN=64, BK=8, TM=4, TN=4)
    launch_tiled[M, N, K, T2](ctx, A, B, C)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(REPS):
        launch_tiled[M, N, K, T2](ctx, A, B, C)
    ctx.synchronize()
    report("tiled 64/8/4x4", Int(perf_counter_ns() - t0))


    comptime V1 = Tile(BM=64, BN=64, BK=16, TM=4, TN=4)
    launch_vec[M, N, K, V1](ctx, A, B, C)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(REPS):
        launch_vec[M, N, K, V1](ctx, A, B, C)
    ctx.synchronize()
    report("vec Tile(BM=64, BN=64, BK=16, TM=4, TN=4)", Int(perf_counter_ns() - t0))

    comptime V2 = Tile(BM=64, BN=64, BK=16, TM=4, TN=8)
    launch_vec[M, N, K, V2](ctx, A, B, C)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(REPS):
        launch_vec[M, N, K, V2](ctx, A, B, C)
    ctx.synchronize()
    report("vec Tile(BM=64, BN=64, BK=16, TM=4, TN=8)", Int(perf_counter_ns() - t0))

