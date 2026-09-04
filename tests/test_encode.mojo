"""Encoder kernels against scalar references, reference profile, three columns."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext, HostBuffer

from caracal7.field import F2, F4, f_mul, f_add, ext_mul, ext_pow
from caracal7.params import REFERENCE
from caracal7.tables import Domains, TableLayout, build_tables
from caracal7.arena import Arena, Bump
from caracal7.encode import EncLayout, encode, slot_target, pack_slot

comptime p = REFERENCE
comptime COLS = 3
comptime N = p.N()
comptime h1 = p.h1()
comptime h2 = p.h2()
comptime K = N // 4


def _run() raises -> Tuple[List[UInt8], List[UInt8], List[UInt8], List[UInt8], List[UInt8], Domains]:
    """Encode COLS pseudo-random columns; return trace, coeff, stored, packed, code as host lists."""
    var ctx = DeviceContext()
    var d = Domains.__init__[p]()
    var bump = Bump()
    var e = EncLayout.__init__[p](bump, COLS)
    var tab = TableLayout.__init__[p](bump.alloc(0))
    _ = bump.alloc(tab.bytes)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, tab.base, build_tables[p](ctx, tab, d))

    var th = ctx.enqueue_create_host_buffer[DType.uint8](COLS * N)
    ctx.synchronize()
    for i in range(COLS * N):
        th[i] = UInt8((i * 7919 + 13) % 127)
    arena.upload(ctx, e.trace, th)
    encode[p](ctx, arena.base(), e, tab)

    var ch = ctx.enqueue_create_host_buffer[DType.uint8](COLS * N * 2)
    var sh = ctx.enqueue_create_host_buffer[DType.uint8](COLS * N)
    var ph = ctx.enqueue_create_host_buffer[DType.uint8](COLS * N)
    var oh = ctx.enqueue_create_host_buffer[DType.uint8](p.L0 * COLS * 4)
    arena.download(ctx, e.coeff, ch)
    arena.download(ctx, e.stored, sh)
    arena.download(ctx, e.packed, ph)
    arena.download(ctx, e.code, oh)
    ctx.synchronize()

    return (_to_list(th), _to_list(ch), _to_list(sh), _to_list(ph), _to_list(oh), d)


def _to_list(h: HostBuffer[DType.uint8]) -> List[UInt8]:
    var l = List[UInt8](capacity=len(h))
    for i in range(len(h)):
        l.append(h[i])
    return l^


def _f2(l: List[UInt8], off: Int) -> F2:
    return F2(l[off], l[off + 1])


def _f4(l: List[UInt8], off: Int) -> F4:
    return F4(l[off], l[off + 1], l[off + 2], l[off + 3])


def test_idft2_interpolates_trace() raises:
    var r = _run()
    var trace = r[0].copy()
    var coeff = r[1].copy()
    var d = r[5]
    # c(omega1^t1, omega2^t2) = sum_k coeff[k2, k1] omega1^(t1 k1) omega2^(t2 k2) must equal the trace value.
    for c in range(COLS):
        for (t1, t2) in [(0, 0), (1, 0), (0, 1), (5, 7), (71, 31), (36, 16), (13, 29)]:
            var acc = F2(0)
            for k2 in range(h2):
                var w2 = ext_pow[1](d.omega2, (t2 * k2) % h2)
                for k1 in range(h1):
                    var w1 = ext_pow[1](d.omega1, (t1 * k1) % h1)
                    acc = f_add(acc, ext_mul[1](ext_mul[1](w1, w2), _f2(coeff, ((c * h2 + k2) * h1 + k1) * 2)))
            assert_equal(Int(acc[0]), Int(trace[(c * h2 + t2) * h1 + t1]))
            assert_equal(Int(acc[1]), 0)


def test_to_stored_matches_reference_and_fixed_slots_are_real() raises:
    var r = _run()
    var coeff = r[1].copy()
    var stored = r[2].copy()
    var d = r[5]
    comptime H1 = 1 << (p.a1 - 1)
    comptime H2 = 1 << (p.a2 - 1)
    for c in range(COLS):
        for slot in range(N):
            var x1: Int
            var x2: Int
            var rr: Int
            var coord: Int
            x1, x2, rr, coord = slot_target[p](slot)
            var r1 = rr % p.m1
            var r2 = rr // p.m1
            var acc = F2(0)
            for y2 in range(p.m2):
                for y1 in range(p.m1):
                    var s = f_mul(ext_pow[0](SIMD[DType.uint8, 1](d.rho1), y1 * r1), ext_pow[0](SIMD[DType.uint8, 1](d.rho2), y2 * r2))
                    var k1 = x1 + (1 << p.a1) * y1
                    var k2 = x2 + (1 << p.a2) * y2
                    acc = f_add(acc, f_mul(_f2(coeff, ((c * h2 + k2) * h1 + k1) * 2), F2(s[0])))
            assert_equal(Int(stored[c * N + slot]), Int(acc[coord]))
            var fixed = (x1 == 0 or x1 == H1) and (x2 == 0 or x2 == H2)
            if fixed:
                assert_equal(Int(acc[1]), 0)     # spec 9.1: fixed classes lie in F


def test_pack_gathers_slots() raises:
    var r = _run()
    var stored = r[2].copy()
    var packed = r[3].copy()
    for c in range(COLS):
        for i in range(K):
            for j in range(4):
                assert_equal(Int(packed[(c * K + i) * 4 + j]), Int(stored[c * N + pack_slot[p](i, j)]))


def test_rs_encode_evaluates_message() raises:
    var r = _run()
    var packed = r[3].copy()
    var code = r[4].copy()
    var d = r[5]
    for c in range(COLS):
        for s in [0, 1, 2, 255, 256, 315, 4097, 40320, 80639, 12345]:
            var pt = ext_pow[2](d.g, s)
            var acc = F4(0)
            var pw = F4(1, 0, 0, 0)
            for i in range(K):
                acc = f_add(acc, ext_mul[2](_f4(packed, (c * K + i) * 4), pw))
                pw = ext_mul[2](pw, pt)
            var got = _f4(code, (s * COLS + c) * 4)
            assert_true(got == acc, "code mismatch at s=" + String(s) + " col " + String(c))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
