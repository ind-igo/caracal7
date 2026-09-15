"""The small grid kernels against host polynomial arithmetic, reference profile: random line vectors
on H2, R2 by host convolution and long division by X2^h2 - 1, against the device Q3 on G2; and the
cyclic interpolation against Horner on the coefficients."""

from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext, HostBuffer

from caracal7.core.field import F2, E, f_add, f_sub, ext_mul, ext_pow, ext_embed, ext_inv, E_LEVEL, E_BYTES
from caracal7.core.params import CLIENT
from caracal7.core.tables import Domains, TableLayout, build_tables
from caracal7.core.arena import Arena, Bump
from caracal7.core.bytes import list_e
from caracal7.relations.smallgrid import SmallGridLayout, small_grid_product, small_grid_values, interp_cyclic

comptime p = CLIENT.grid(72, 32)
comptime h2 = p.h2()


def _rand(n: Int, seed: Int) -> List[UInt8]:
    var l = List[UInt8](capacity=n)
    var s = seed
    for _ in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        l.append(UInt8((s >> 8) % 127))
    return l^


def _host(ctx: DeviceContext, l: List[UInt8]) raises -> HostBuffer[DType.uint8]:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](len(l))
    ctx.synchronize()
    for i in range(len(l)):
        h[i] = l[i]
    return h^



def _coeffs(vals: List[UInt8], off: Int, w2_inv: F2) -> List[E]:
    """Inverse DFT on H2 by the definition: c_k = h2^-1 sum_t v_t omega2^(-t k)."""
    var inv = E(0)
    inv[0] = 1
    for _ in range(125):
        inv[0] = UInt8((Int(inv[0]) * h2) % 127)
    var out = List[E]()
    for k in range(h2):
        var acc = E(0)
        for t in range(h2):
            acc = f_add(acc, ext_mul[E_LEVEL](list_e(vals, off + t), ext_embed[E_LEVEL](ext_pow[1](w2_inv, (t * k) % h2))))
        out.append(ext_mul[E_LEVEL](acc, inv))
    return out^


def _mul(a: List[E], b: List[E]) -> List[E]:
    var out = List[E](length=len(a) + len(b) - 1, fill=E(0))
    for i in range(len(a)):
        for j in range(len(b)):
            out[i + j] = f_add(out[i + j], ext_mul[E_LEVEL](a[i], b[j]))
    return out^


def _horner(c: List[E], z: E) -> E:
    var acc = E(0)
    for k in range(len(c) - 1, -1, -1):
        acc = f_add(ext_mul[E_LEVEL](acc, z), c[k])
    return acc


def test_q3_matches_host_division_and_interpolation() raises:
    var ctx = DeviceContext()
    var d = Domains.__init__[p]()
    var z2 = _rand(h2 * E_BYTES, 1)
    for t in range(E_BYTES):   # the trailing Z2(omega2^h2) = Z2(1) entry (accumulate.k_z2)
        z2.append(z2[t])
    var zend = _rand(h2 * E_BYTES, 2)
    var nend = _rand(h2 * E_BYTES, 3)
    var dend = _rand(h2 * E_BYTES, 4)
    for t in range(h2):     # b d = a c n on H2, so R2 vanishes there and the quotient is exact (the device divides on a coset)
        var v = ext_mul[E_LEVEL](ext_mul[E_LEVEL](list_e(z2, t), list_e(zend, t)), ext_mul[E_LEVEL](list_e(nend, t), ext_inv[E_LEVEL](list_e(z2, (t + 1) % h2))))
        for i in range(E_BYTES):
            dend[t * E_BYTES + i] = v[i]
    var alpha = _rand(E_BYTES, 5)
    var bump = Bump()
    var tab = TableLayout.__init__[p](bump.alloc(0))
    _ = bump.alloc(tab.bytes)
    var o_z2 = bump.alloc((h2 + 1) * E_BYTES)
    var o_zend = bump.alloc(h2 * E_BYTES)
    var o_nend = bump.alloc(h2 * E_BYTES)
    var o_dend = bump.alloc(h2 * E_BYTES)
    var o_alpha = bump.alloc(E_BYTES)
    var sg = SmallGridLayout.__init__[p](bump)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, tab.base, build_tables[p](ctx, tab, d))
    arena.upload(ctx, o_z2, _host(ctx, z2))
    arena.upload(ctx, o_zend, _host(ctx, zend))
    arena.upload(ctx, o_nend, _host(ctx, nend))
    arena.upload(ctx, o_dend, _host(ctx, dend))
    arena.upload(ctx, o_alpha, _host(ctx, alpha))
    var e2 = ext_pow[1](d.omega2, h2 - 1)
    # two accumulators sharing the lines: R2 = (1 + alpha) (X2 - e2)(b d - a c n)
    small_grid_product[p](ctx, arena, tab, o_z2, [(o_zend, E_BYTES), (o_nend, E_BYTES)], [(o_dend, E_BYTES)], sg, o_alpha, 0, e2, True)
    small_grid_product[p](ctx, arena, tab, o_z2, [(o_zend, E_BYTES), (o_nend, E_BYTES)], [(o_dend, E_BYTES)], sg, o_alpha, 1, e2, False)
    small_grid_values[p](ctx, arena, tab, sg)
    var qh = ctx.enqueue_create_host_buffer[DType.uint8](2 * h2 * E_BYTES)
    arena.download(ctx, sg.q3, qh)
    ctx.synchronize()
    var q3 = List[UInt8](capacity=2 * h2 * E_BYTES)
    for i in range(2 * h2 * E_BYTES):
        q3.append(qh[i])

    # host: R2 in coefficients, then the exact quotient q_k = r_{k+h2} + r_{k+2h2} (R2 vanishes on H2)
    var w2_inv = ext_pow[1](d.omega2, h2 - 1)
    var a = _coeffs(z2, 0, w2_inv)
    var zs = List[UInt8](capacity=h2 * E_BYTES)
    for t in range(h2):
        for i in range(E_BYTES):
            zs.append(z2[((t + 1) % h2) * E_BYTES + i])
    var b = _coeffs(zs, 0, w2_inv)
    var c = _coeffs(zend, 0, w2_inv)
    var n = _coeffs(nend, 0, w2_inv)
    var dd = _coeffs(dend, 0, w2_inv)
    var p1 = _mul(b, dd)
    var p2 = _mul(_mul(a, c), n)
    var s = List[E](length=3 * h2 - 2, fill=E(0))
    for i in range(len(p1)):
        s[i] = p1[i]
    for i in range(len(p2)):
        s[i] = f_sub(s[i], p2[i])
    var e2e = ext_embed[E_LEVEL](e2)
    var r = List[E](length=4 * h2, fill=E(0))     # zero past 3 h2 - 1, as the device reads it
    for m in range(3 * h2):
        var lo = s[m - 1] if m >= 1 and m - 1 < len(s) else E(0)
        var hi = s[m] if m < len(s) else E(0)
        r[m] = f_sub(lo, ext_mul[E_LEVEL](e2e, hi))
    var one = E(0)
    one[0] = 1
    var scale = f_add(one, list_e(alpha, 0))
    var q = List[E](length=2 * h2, fill=E(0))
    for k in range(2 * h2):
        q[k] = ext_mul[E_LEVEL](scale, f_add(r[k + h2], r[k + 2 * h2]))
    for j in [0, 1, 2, 7, 2 * h2 - 1]:
        var pt = ext_embed[E_LEVEL](ext_pow[1](d.g2, j))
        assert_true(list_e(q3, j) == _horner(q, pt), "Q3 on G2 differs from the host quotient")
    # interpolation: a random point against Horner on the coefficients
    var zr = E(0)
    for t in range(E_BYTES):
        zr[t] = UInt8((t * 41 + 9) % 127)
    assert_true(interp_cyclic(q3, 0, 2 * h2, d.g2, zr) == _horner(q, zr), "cyclic interpolation of Q3 differs from Horner")
    assert_true(interp_cyclic(z2, 0, h2, d.omega2, zr) == _horner(a, zr), "cyclic interpolation of Z2 differs from Horner")
    assert_true(_horner(q, zr) != E(0), "vacuous")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
