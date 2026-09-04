from std.testing import assert_equal, TestSuite
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from caracal7.gemm import launch_naive, launch_tiled, TILE_DEFAULT, Tile

comptime M = 128
comptime N = 192
comptime K = 320   # crosses one REDUCE_EVERY boundary
comptime al = row_major[M, K]()
comptime bl = row_major[K, N]()
comptime cl = row_major[M, N]()


def _check[which: Int]() raises:
    var ctx = DeviceContext()
    var a_dev = ctx.enqueue_create_buffer[DType.uint8](M * K)
    var b_dev = ctx.enqueue_create_buffer[DType.uint8](K * N)
    var c_dev = ctx.enqueue_create_buffer[DType.uint8](M * N)
    var a_host = ctx.enqueue_create_host_buffer[DType.uint8](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.uint8](K * N)
    var c_host = ctx.enqueue_create_host_buffer[DType.uint8](M * N)
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
    comptime if which == 0:
        launch_naive[M, N, K](ctx, A, B, C)
    else:
        launch_tiled[M, N, K, TILE_DEFAULT](ctx, A, B, C)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()
    var bad = 0
    for i in range(M):
        for j in range(N):
            var s = 0
            for k in range(K):
                s += Int(a_host[i * K + k]) * Int(b_host[k * N + j])
            if Int(c_host[i * N + j]) != s % 127:
                bad += 1
    assert_equal(bad, 0)


def test_naive_matches_scalar() raises:
    _check[0]()


def test_tiled_matches_scalar() raises:
    _check[1]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
