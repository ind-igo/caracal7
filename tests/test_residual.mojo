"""Residual stage on the synthetic families, reference profile: LDE against direct evaluation, the
residual vanishing on H and matching a host walk of the entry table on G, the DEEP identity
R(z) = (A(z) + z2^h2 B(z)) (z1^h1 - 1) + Q2(z) (z2^h2 - 1) at a random z in E^2, the degree bounds
on B and Q2, and the coordinate columns as values on H."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext, HostBuffer

from caracal7.field import F2, E, f_add, f_sub, f_mul, ext_mul, ext_pow, ext_embed
from caracal7.params import REFERENCE
from caracal7.tables import Domains, TableLayout, build_tables
from caracal7.arena import Arena, Bump
from caracal7.encode import EncLayout, to_packed
from caracal7.residual import ENTRY, NONE, SYNTHETIC_COLUMNS, synthetic_families, synthetic_trace, entry, residual_at
from caracal7.residual import lde, residual, quotient, quotient_elems
from caracal7.bytes import list_e

comptime p = REFERENCE
comptime C = SYNTHETIC_COLUMNS
comptime N = p.N()
comptime h1 = p.h1()
comptime h2 = p.h2()
comptime G1 = 2 * h1
comptime G2 = 2 * h2
comptime G = G1 * G2


struct Run:
    var coeff: List[UInt8]
    var lde: List[UInt8]
    var res: List[UInt8]
    var scratch: List[UInt8]
    var stored: List[UInt8]
    var fam: List[UInt8]
    var alpha: E
    var chals: List[UInt8]
    var d: Domains

    def __init__(out self) raises:
        var ctx = DeviceContext()
        self.d = Domains.__init__[p]()
        var f = synthetic_families(with_accumulator=False)
        self.fam = f.bytes.copy()
        var bump = Bump()
        var enc = EncLayout.__init__[p](bump, C)
        var tab = TableLayout.__init__[p](bump.alloc(0))
        _ = bump.alloc(tab.bytes)
        var families = bump.alloc(f.count * ENTRY)
        var ltmp = bump.alloc(C * h2 * G1 * 2)
        var lde_buf = bump.alloc(C * G * 2)
        var res_buf = bump.alloc(G * p.e)
        var scratch = bump.alloc(quotient_elems[p]() * p.e)
        var stored = bump.alloc(3 * p.e * N)
        var alpha = bump.alloc(p.e)
        var chals = bump.alloc(3 * p.e)
        var arena = Arena(ctx, bump.used)
        arena.upload(ctx, tab.base, build_tables[p](ctx, tab, self.d))
        arena.upload(ctx, enc.trace, _host(ctx, synthetic_trace[p](1)))
        arena.upload(ctx, families, _host(ctx, f.bytes))
        self.alpha = E(0)
        for i in range(p.e):
            self.alpha[i] = UInt8((i * 29 + 3) % 127)
        var ah = ctx.enqueue_create_host_buffer[DType.uint8](p.e)
        ctx.synchronize()
        for i in range(p.e):
            ah[i] = self.alpha[i]
        arena.upload(ctx, alpha, ah)
        self.chals = List[UInt8](capacity=3 * p.e)
        for i in range(3 * p.e):
            self.chals.append(UInt8((i * 53 + 11) % 127))
        arena.upload(ctx, chals, _host(ctx, self.chals))

        to_packed[p](ctx, arena.base(), enc, tab)
        lde[p](ctx, arena.base(), enc.coeff, C, tab, ltmp, lde_buf)
        residual[p](ctx, arena.base(), lde_buf, families, f.count, tab, alpha, chals, res_buf)
        quotient[p](ctx, arena.base(), res_buf, tab, scratch, stored)

        var ch = ctx.enqueue_create_host_buffer[DType.uint8](C * N * 2)
        var lh = ctx.enqueue_create_host_buffer[DType.uint8](C * G * 2)
        var rh = ctx.enqueue_create_host_buffer[DType.uint8](G * p.e)
        var qh = ctx.enqueue_create_host_buffer[DType.uint8](quotient_elems[p]() * p.e)
        var sh = ctx.enqueue_create_host_buffer[DType.uint8](3 * p.e * N)
        arena.download(ctx, enc.coeff, ch)
        arena.download(ctx, lde_buf, lh)
        arena.download(ctx, res_buf, rh)
        arena.download(ctx, scratch, qh)
        arena.download(ctx, stored, sh)
        ctx.synchronize()
        self.coeff = _to_list(ch)
        self.lde = _to_list(lh)
        self.res = _to_list(rh)
        self.scratch = _to_list(qh)
        self.stored = _to_list(sh)

    def f2(self, l: List[UInt8], off: Int) -> F2:
        return F2(l[off], l[off + 1])

    def e(self, l: List[UInt8], off: Int) -> E:
        return list_e(l, off // p.e)

    def col_at(self, c: Int, z1: E, z2: E) -> E:
        """The column c at z from the monomial coefficients, Horner on both axes."""
        var acc = E(0)
        for k2 in range(h2 - 1, -1, -1):
            var inner = E(0)
            for k1 in range(h1 - 1, -1, -1):
                inner = f_add(ext_mul[4](inner, z1), ext_embed[4](self.f2(self.coeff, ((c * h2 + k2) * h1 + k1) * 2)))
            acc = f_add(ext_mul[4](acc, z2), inner)
        return acc

    def poly_at(self, off: Int, rows: Int, z1: E, z2: E) -> E:
        """An E-valued (k2, k1, e) coefficient table at z."""
        var acc = E(0)
        for k2 in range(rows - 1, -1, -1):
            var inner = E(0)
            for k1 in range(h1 - 1, -1, -1):
                inner = f_add(ext_mul[4](inner, z1), self.e(self.scratch, off + (k2 * h1 + k1) * p.e))
            acc = f_add(ext_mul[4](acc, z2), inner)
        return acc

    def gates(self) -> Tuple[F2, F2]:
        return (ext_pow[1](self.d.omega1, h1 - 1), ext_pow[1](self.d.omega2, h2 - 1))


def _host(ctx: DeviceContext, l: List[UInt8]) raises -> HostBuffer[DType.uint8]:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](len(l))
    ctx.synchronize()
    for i in range(len(l)):
        h[i] = l[i]
    return h^


def _to_list(h: HostBuffer[DType.uint8]) -> List[UInt8]:
    var l = List[UInt8](capacity=len(h))
    for i in range(len(h)):
        l.append(h[i])
    return l^


def _random_e(seed: Int) -> E:
    var v = E(0)
    var s = seed
    for i in range(p.e):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        v[i] = UInt8((s >> 8) % 127)
    return v


def test_lde_matches_direct_evaluation() raises:
    var r = Run()
    var points: List[Int] = [0, 1, G1 + 1, 5 * G1 + 7, (G2 - 1) * G1 + G1 - 1, 17 * G1 + 100]
    for c in range(0, C, 3):
        for n in points:
            var j1 = n % G1
            var j2 = n // G1
            var z1 = ext_embed[4](ext_pow[1](r.d.g1, j1))
            var z2 = ext_embed[4](ext_pow[1](r.d.g2, j2))
            var want = r.col_at(c, z1, z2)
            var got = ext_embed[4](r.f2(r.lde, ((c * G2 + j2) * G1 + j1) * 2))
            assert_true(want == got, "lde mismatch")


def test_residual_vanishes_on_h_and_matches_host_on_g() raises:
    var r = Run()
    var e1: F2
    var e2: F2
    e1, e2 = r.gates()
    var bad = 0
    for j2 in range(0, G2, 2):
        for j1 in range(0, G1, 2):
            if r.e(r.res, (j2 * G1 + j1) * p.e) != E(0):
                bad += 1
    assert_equal(bad, 0)
    var points: List[Int] = [1, G1, G1 + 1, 9 * G1 + 50, (G2 - 1) * G1 + G1 - 1]
    for n in points:
        var j1 = n % G1
        var j2 = n // G1
        var reads = List[E]()
        for k in range(len(r.fam) // ENTRY):
            var en = entry(r.fam, k)
            var a1 = (j1 + en.dj1_a) % G1
            var a2 = (j2 + en.dj2_a) % G2
            reads.append(ext_embed[4](r.f2(r.lde, ((en.col_a * G2 + a2) * G1 + a1) * 2)))
            var b1 = (j1 + en.dj1_b) % G1
            var b2 = (j2 + en.dj2_b) % G2
            var cb = 0 if en.col_b == NONE else en.col_b
            reads.append(ext_embed[4](r.f2(r.lde, ((cb * G2 + b2) * G1 + b1) * 2)))
        var z1 = ext_embed[4](ext_pow[1](r.d.g1, j1))
        var z2 = ext_embed[4](ext_pow[1](r.d.g2, j2))
        var want = residual_at(r.fam, r.alpha, r.chals, z1, z2, e1, e2, reads)
        assert_true(want == r.e(r.res, (j2 * G1 + j1) * p.e), "residual mismatch on G")


def test_deep_identity_at_random_z() raises:
    var r = Run()
    var e1: F2
    var e2: F2
    e1, e2 = r.gates()
    var z1 = _random_e(77)
    var z2 = _random_e(91)
    var reads = List[E]()
    for k in range(len(r.fam) // ENTRY):
        var en = entry(r.fam, k)
        reads.append(r.col_at(en.col_a, ext_mul[4](z1, ext_embed[4](ext_pow[1](r.d.g1, en.dj1_a))),
                              ext_mul[4](z2, ext_embed[4](ext_pow[1](r.d.g2, en.dj2_a)))))
        if en.col_b == NONE:
            reads.append(E(0))
        else:
            reads.append(r.col_at(en.col_b, ext_mul[4](z1, ext_embed[4](ext_pow[1](r.d.g1, en.dj1_b))),
                                  ext_mul[4](z2, ext_embed[4](ext_pow[1](r.d.g2, en.dj2_b)))))
    var rz = residual_at(r.fam, r.alpha, r.chals, z1, z2, e1, e2, reads)
    var q1coef = 2 * h1 * G2 * p.e
    var q2coef = q1coef + G2 * h1 * p.e
    var a = r.poly_at(q1coef, h2, z1, z2)
    var b = r.poly_at(q1coef + h2 * h1 * p.e, h2, z1, z2)
    var q2 = r.poly_at(q2coef, h2, z1, z2)
    var one = E(0)
    one[0] = 1
    var z1h = f_sub(ext_pow[4](z1, h1), one)
    var z2h = ext_pow[4](z2, h2)
    var rhs = f_add(ext_mul[4](f_add(a, ext_mul[4](z2h, b)), z1h), ext_mul[4](q2, f_sub(z2h, one)))
    assert_true(rz == rhs, "DEEP identity fails at z")
    assert_true(rz != E(0), "R(z) is zero: the check is vacuous")
    # degree bounds: B has X2-degree <= h2 - 2, Q2 likewise (spec 8)
    var top = 0
    for k1 in range(h1):
        for i in range(p.e):
            top += Int(r.scratch[q1coef + ((G2 - 1) * h1 + k1) * p.e + i])
            top += Int(r.scratch[q2coef + ((h2 - 1) * h1 + k1) * p.e + i])
    assert_equal(top, 0)


def test_quotient_trace_is_values_on_h() raises:
    var r = Run()
    var q1coef = 2 * h1 * G2 * p.e
    var srcs: List[Int] = [q1coef, q1coef + h2 * h1 * p.e, q1coef + G2 * h1 * p.e]
    var rows: List[Int] = [0, 1, 5, h2 - 1]
    var cols: List[Int] = [0, 1, 40, h1 - 1]
    for q in range(3):
        for x2 in rows:
            for x1 in cols:
                var v = r.poly_at(srcs[q], h2, ext_embed[4](ext_pow[1](r.d.omega1, x1)), ext_embed[4](ext_pow[1](r.d.omega2, x2)))
                for tau in range(p.e):
                    assert_equal(Int(r.stored[(q * p.e + tau) * N + x2 * h1 + x1]), Int(v[tau]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
