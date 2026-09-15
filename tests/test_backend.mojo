"""The F2 GEMM skeleton against a scalar F2 matmul: predicated edges, a batch, and the D = 8 lane view."""

from std.testing import assert_equal, TestSuite
from max.gpu.host import DeviceContext

from caracal7.core.field import F2, F4, f_add, ext_mul
from caracal7.core.arena import Arena, Bump
from caracal7.core.backend import BACKEND, F4_TILE, Tile, Strided, Strided4, launch_gemm_f2, launch_gemm_f4, strided


def _run[D: Int, T: Tile](M: Int, N: Int, K: Int, batch: Int) raises:
    """A (z, m, k) and B (z, k, n) interleaved F2; B and C addressed through the (hi, lo) split."""
    var ctx = DeviceContext()
    var bump = Bump()
    var a = bump.alloc(batch * M * K * 2)
    var b = bump.alloc(batch * K * N * 2)
    var c = bump.alloc(batch * M * N * 2)
    var arena = Arena(ctx, bump.used)
    var h = ctx.enqueue_create_host_buffer[DType.uint8](bump.used)
    ctx.synchronize()
    for i in range(bump.used):
        h[i] = UInt8((i * 31 + 5) % 127)
    arena.upload(ctx, 0, h)

    # n = hi * D + lo: B row k holds N entries as (hi, lo) with lo fastest; same for C
    var o = strided(a, K * 2, 2, sa_z=M * K * 2,
                    b=b, sb_k=N * 2, sb_hi=D * 2, sb_lo=2, sb_z=K * N * 2,
                    c=c, sc_m=N * 2, sc_hi=D * 2, sc_lo=2, sc_z=M * N * 2)
    launch_gemm_f2[BACKEND, T, Strided, D](ctx, arena, o, M, N, K, batch)
    arena.download(ctx, 0, h)
    ctx.synchronize()

    var bad = 0
    for z in range(batch):
        for m in range(M):
            for n in range(N):
                var acc = F2(0)
                for k in range(K):
                    var x = F2(h[a + ((z * M + m) * K + k) * 2], h[a + ((z * M + m) * K + k) * 2 + 1])
                    var y = F2(h[b + ((z * K + k) * N + n) * 2], h[b + ((z * K + k) * N + n) * 2 + 1])
                    acc = f_add(acc, ext_mul[1](x, y))
                var got = F2(h[c + ((z * M + m) * N + n) * 2], h[c + ((z * M + m) * N + n) * 2 + 1])
                if acc != got:
                    bad += 1
    assert_equal(bad, 0)


def test_edges_and_batch() raises:
    _run[1, BACKEND.tile](70, 100, 200, 2)      # crosses one max_terms boundary, ragged in M, N, K


def test_lane_view() raises:
    _run[8, Tile(BM=8, BN=128, BK=16, TM=8, TN=4, mma=False)](8, 13 * 8, 13, 1)   # the residual pass shape


def test_f4() raises:
    """The F4 skeleton against a scalar F4 matmul: ragged edges, K across two F4_TERMS reductions, a batch."""
    var M = 70
    var N = 50
    var K = 75
    var batch = 2
    var ctx = DeviceContext()
    var bump = Bump()
    var a = bump.alloc(batch * M * K * 4)
    var b = bump.alloc(batch * K * N * 4)
    var c = bump.alloc(batch * M * N * 4)
    var arena = Arena(ctx, bump.used)
    var h = ctx.enqueue_create_host_buffer[DType.uint8](bump.used)
    ctx.synchronize()
    for i in range(bump.used):
        h[i] = UInt8((i * 29 + 3) % 127)
    arena.upload(ctx, 0, h)
    var o = strided(a, K * 4, 4, sa_z=M * K * 4, b=b, sb_k=N * 4, sb_hi=4, sb_lo=0, sb_z=K * N * 4,
                    c=c, sc_m=N * 4, sc_hi=4, sc_lo=0, sc_z=M * N * 4)
    launch_gemm_f4[BACKEND, F4_TILE, Strided4, 1](ctx, arena, o, M, N, K, batch)
    arena.download(ctx, 0, h)
    ctx.synchronize()
    var bad = 0
    for z in range(batch):
        for m in range(M):
            for n in range(N):
                var acc = F4(0)
                for k in range(K):
                    var x = h.unsafe_ptr().unsafe_load[width=4](a + ((z * M + m) * K + k) * 4)
                    var y = h.unsafe_ptr().unsafe_load[width=4](b + ((z * K + k) * N + n) * 4)
                    acc = f_add(acc, ext_mul[2](x, y))
                if acc != h.unsafe_ptr().unsafe_load[width=4](c + ((z * M + m) * N + n) * 4):
                    bad += 1
    assert_equal(bad, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
