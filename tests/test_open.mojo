"""Open stage on a real proof run: <w_z, stored(c)> = c(z_p) for every witness column and point
against direct evaluation from the coefficients, the quotient openings against their coefficient
tables, and the fold against a host sum. The challenges are read back from the arena, which only
a test may do."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext, HostBuffer

from caracal7.field import F2, E, f_add, f_mul, ext_mul, ext_pow, ext_embed
from caracal7.verifier import encode_at
from caracal7.params import REFERENCE
from caracal7.hash import Blake3
from caracal7.proof import Shape
from caracal7.prover import Prover, load_trace
from caracal7.ir import shift_points, point_coord
from caracal7.synthetic import SYNTHETIC_COLUMNS, synthetic_families, synthetic_trace
from caracal7.bytes import list_e

comptime p = REFERENCE
comptime N = p.N()
comptime h1 = p.h1()
comptime h2 = p.h2()
comptime G2 = 2 * h2


def _dl(ctx: DeviceContext, prover: Prover[p, Blake3], off: Int, bytes: Int) raises -> List[UInt8]:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](bytes)
    prover.arena.download(ctx, off, h)
    ctx.synchronize()
    var l = List[UInt8](capacity=bytes)
    for i in range(bytes):
        l.append(h[i])
    return l^


def _horner(coef: List[UInt8], off: Int, stride: Int, ext: Int, z1: E, z2: E, rows: Int) -> E:
    """sum_{k2 < rows, k1} coef[off + (k2 h1 + k1) stride] z1^k1 z2^k2, coefficient width ext (2 or 16)."""
    var acc = E(0)
    for k2 in range(rows - 1, -1, -1):
        var inner = E(0)
        for k1 in range(h1 - 1, -1, -1):
            var at = off + (k2 * h1 + k1) * stride
            var c = E(0)
            for t in range(ext):
                c[t] = coef[at + t]
            inner = f_add(ext_mul[4](inner, z1), c)
        acc = f_add(ext_mul[4](acc, z2), inner)
    return acc


def test_openings_and_fold() raises:
    var ctx = DeviceContext()
    var f = synthetic_families()
    var shape = Shape.__init__[p](SYNTHETIC_COLUMNS, f.bytes, f.accs)
    var prover = Prover[p, Blake3](ctx, Shape.__init__[p](SYNTHETIC_COLUMNS, f.bytes, f.accs), f.bytes.copy())
    load_trace[p, Blake3](ctx, prover, synthetic_trace[p](1))
    _ = prover.prove(ctx, List[UInt8]())
    ref L = prover.layout
    var C = shape.columns()
    var P = shape.points
    var z = _dl(ctx, prover, L.z, 2 * p.e)
    var coeff = _dl(ctx, prover, L.enc_w.coeff, SYNTHETIC_COLUMNS * N * 2)
    var openings = _dl(ctx, prover, L.openings, P * C * p.e)
    var scratch = _dl(ctx, prover, L.quotient, 14 * N * p.e)
    var beta = _dl(ctx, prover, L.beta_gamma, C * p.e)
    var stored_w = _dl(ctx, prover, L.enc_w.stored, SYNTHETIC_COLUMNS * N)
    var stored_z = _dl(ctx, prover, L.enc_z.stored, shape.columns_z * N)
    var stored_q = _dl(ctx, prover, L.enc_q.stored, shape.columns_q * N)
    var y = _dl(ctx, prover, L.fold_y, N * p.e)
    var code_w = _dl(ctx, prover, L.enc_w.code, p.L() * SYNTHETIC_COLUMNS * 4)
    var code_z = _dl(ctx, prover, L.enc_z.code, p.L() * shape.columns_z * 4)
    var code_q = _dl(ctx, prover, L.enc_q.code, p.L() * shape.columns_q * 4)
    var pts = shift_points(f.bytes)
    var d = prover.domains

    # witness columns at every point
    for pt in range(P):
        var dj1 = Int(pts[pt * 4]) | Int(pts[pt * 4 + 1]) << 8
        var dj2 = Int(pts[pt * 4 + 2]) | Int(pts[pt * 4 + 3]) << 8
        var z1 = point_coord(list_e(z, 0), dj1, d.g1, p.h1())
        var z2 = point_coord(list_e(z, 1), dj2, d.g2, p.h2())
        for c in range(SYNTHETIC_COLUMNS):
            var want = _horner(coeff, c * N * 2, 2, 2, z1, z2, h2)
            assert_true(want == list_e(openings, pt * C + c), "witness opening mismatch")
    # quotient coordinate columns at z: sum_tau e_tau alpha_tau = Q(z) from the coefficient tables
    var q1coef = 2 * h1 * G2 * p.e
    var srcs: List[Int] = [q1coef, q1coef + h2 * h1 * p.e, q1coef + G2 * h1 * p.e]
    for q in range(3):
        var want = _horner(scratch, srcs[q], p.e, p.e, list_e(z, 0), list_e(z, 1), h2)
        var got = E(0)
        for tau in range(p.e):
            var basis = E(0)
            basis[tau] = 1
            got = f_add(got, ext_mul[4](basis, list_e(openings, SYNTHETIC_COLUMNS + shape.columns_z + q * p.e + tau)))
        assert_true(want == got, "quotient opening mismatch")
    # fold at a few slots
    var slots: List[Int] = [0, 1, 77, N // 2, N - 1]
    for slot in slots:
        var acc = E(0)
        for c in range(C):
            var v = stored_w[c * N + slot] if c < SYNTHETIC_COLUMNS else (stored_z[(c - SYNTHETIC_COLUMNS) * N + slot] if c < SYNTHETIC_COLUMNS + shape.columns_z else stored_q[(c - SYNTHETIC_COLUMNS - shape.columns_z) * N + slot])
            acc = f_add(acc, f_mul(list_e(beta, c), E(v)))
        assert_true(acc == list_e(y, slot), "fold mismatch")
    # the alphabet rule at a few code positions: Enc(y)(g^s) = sum_c beta_c X[s, c] coordinate-wise in E (x) F4
    for s in [0, 1, 4097, p.L() - 1]:
        var enc = encode_at[p](y, d.level1.point(s))
        var nonzero = False
        for tau in range(4):
            var want = E(0)
            for c in range(C):
                var sym = code_w[(s * SYNTHETIC_COLUMNS + c) * 4 + tau] if c < SYNTHETIC_COLUMNS else (code_z[(s * shape.columns_z + c - SYNTHETIC_COLUMNS) * 4 + tau] if c < SYNTHETIC_COLUMNS + shape.columns_z else code_q[(s * shape.columns_q + c - SYNTHETIC_COLUMNS - shape.columns_z) * 4 + tau])
                want = f_add(want, f_mul(list_e(beta, c), E(sym)))
            assert_true(enc[tau] == want, "alphabet rule mismatch")
            nonzero = nonzero or want != E(0)
        assert_true(nonzero, "consistency check is vacuous")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
