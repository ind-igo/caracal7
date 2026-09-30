"""The sketched code (pcs/sketch.mojo) against scalar references: the sampler's shape, then H x and its RS code."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from core.field import F4, f_add, f_mul, ext_mul
from core.tables import RsDomain
from core.arena import Arena, Bump
from pcs.sketch import SketchMatrix, SketchLayout, upload_sketch, sketch_code


def test_sampler_is_biregular_and_seeded() raises:
    var h = SketchMatrix(1000, 16, 128, 7)
    assert_equal(h.rows, 125)
    var seen = List[Int](length=1000, fill=0)
    for i in range(len(h.col)):
        seen[h.col[i]] += 1
        assert_true(h.coef[i] >= 1 and h.coef[i] <= 126)
    for i in range(1000):
        assert_equal(seen[i], 16)
    for a in range(h.rows):
        assert_equal(h.start[a + 1] - h.start[a], 128)
    var f = SketchMatrix(1000, 16, 128, 7, 4)
    assert_equal(len(f.coef), 4 * len(f.col))
    for i in range(len(f.col)):
        assert_true(Int(f.coef[4 * i]) + Int(f.coef[4 * i + 1]) + Int(f.coef[4 * i + 2]) + Int(f.coef[4 * i + 3]) > 0)
        for b in range(4):
            assert_true(f.coef[4 * i + b] < 127)
    var g = SketchMatrix(1000, 16, 128, 7)
    var o = SketchMatrix(1000, 16, 128, 8)
    var same = True
    var differ = False
    for i in range(len(h.col)):
        same = same and g.col[i] == h.col[i] and g.coef[i] == h.coef[i]
        differ = differ or o.col[i] != h.col[i]
    assert_true(same and differ)


def _check(K: Int, cw: Int, rw: Int, cols: Int, width: Int = 1) raises:
    """Message rows kept, sketch rows = RS(H x) by direct evaluation on every coset."""
    var ctx = DeviceContext()
    var h = SketchMatrix(K, cw, rw, 11, width)
    var bump = Bump()
    var s = SketchLayout(bump, h, cols)
    var arena = Arena(ctx, bump.used)
    upload_sketch(ctx, arena, s, h)
    var xh = ctx.enqueue_create_host_buffer[DType.uint8](K * cols * 4)
    ctx.synchronize()
    for i in range(K * cols * 4):
        xh[i] = UInt8((i * 7919 + 13) % 127)
    arena.upload(ctx, s.code, xh)
    sketch_code(ctx, arena, s)
    var oh = ctx.enqueue_create_host_buffer[DType.uint8](s.leaves() * cols * 4)
    arena.download(ctx, s.code, oh)
    ctx.synchronize()
    for i in range(K * cols * 4):
        assert_equal(oh[i], xh[i])
    var dom = RsDomain(s.L0, s.m)
    for c in [0, cols - 1]:
        var u = List[F4](length=h.rows, fill=F4(0))
        for a in range(h.rows):
            for e in range(h.start[a], h.start[a + 1]):
                var x = F4(xh[(h.col[e] * cols + c) * 4], xh[(h.col[e] * cols + c) * 4 + 1],
                           xh[(h.col[e] * cols + c) * 4 + 2], xh[(h.col[e] * cols + c) * 4 + 3])
                if width == 1:
                    u[a] = f_add(u[a], f_mul(x, F4(h.coef[e])))
                else:
                    var k = F4(h.coef[4 * e], h.coef[4 * e + 1], h.coef[4 * e + 2], h.coef[4 * e + 3])
                    u[a] = f_add(u[a], ext_mul[2](x, k))
        for k in range(s.m):
            for s0 in [0, 1, s.L0 // 3, s.L0 - 1]:
                var pt = dom.point(k * s.L0 + s0)
                var acc = F4(0)
                var pw = F4(1, 0, 0, 0)
                for a in range(h.rows):
                    acc = f_add(acc, ext_mul[2](u[a], pw))
                    pw = ext_mul[2](pw, pt)
                var at = ((K + k * s.L0 + s0) * cols + c) * 4
                assert_true(F4(oh[at], oh[at + 1], oh[at + 2], oh[at + 3]) == acc,
                            "sketch code mismatch at s=" + String(k * s.L0 + s0))
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks


def test_sketch_code_matches_reference() raises:
    _check(4608, 16, 128, 35)      # 576 sketch rows, ragged columns
    _check(1000, 8, 64, 3)         # 125 rows: an odd sketch length
    _check(4608, 24, 192, 35)      # a reduction inside the row (128 of 192 entries)
    _check(4608, 16, 128, 35, 4)   # F4 coefficients: reductions every 28 entries
    _check(1000, 24, 192, 3, 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
