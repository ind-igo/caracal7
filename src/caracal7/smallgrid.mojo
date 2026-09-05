"""The small grid (spec 7.3, 7.4): the chain-end residual R2 on H2 and its quotient Q3 on G2, sent
in the clear. For a permutation accumulator k, with a = Z2, b = Z2(omega2 X2), c = Z(e1, X2),
n = N(e1, X2), d = D(e1, X2) (all polynomials of degree < h2 from their values on H2):

    R2 = sum_k alpha^k (X2 - e2) (b d - a c n),   deg < 3 h2;   Q3 = R2 / (X2^h2 - 1),   deg < 2 h2.

Everything is in coefficient form and O(h2^2): inverse DFT of the five line vectors (winv2),
coefficient products, the division as q_k = r_{k + h2} + r_{k + 2 h2} (exact for an honest R2),
then Q3 evaluated on G2 through wfwd2 and g2^h2 = -1. One thread per output element per launch.

Buffers (bytes; slowest ... fastest): lines (5, h2, e) coefficients a, b, c, n, d; pac (2 h2, e);
p1 (2 h2, e); p2 (3 h2, e); q3c (2 h2, e) the Q3 coefficients summed over accumulators; q3 (2 h2, e)
its values on G2 in generator order, the proof's clear vector.

ponytail: chain-end families are the permutation pair (W) only; lookup and memory chain-end rules
(6.3, 6.4) add terms to R2 here when they exist.
"""

from std.math import ceildiv
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext

from caracal7.field import F2, E, f_add, f_sub, f_mul, f_pow, ext_mul, ext_pow, ext_embed, ext_inv
from caracal7.params import Params
from caracal7.tables import TableLayout

comptime BLOCK = 128


@always_inline
def _gid() -> Int:
    return Int(block_idx.x * block_dim.x + thread_idx.x)


@always_inline
def _e(base: Pointer[UInt8, MutAnyOrigin], off: Int) -> E:
    return base.unsafe_load[width=16](off)


@always_inline
def _f2(base: Pointer[UInt8, MutAnyOrigin], off: Int) -> E:
    return ext_embed[4](base.unsafe_load[width=2](off))


def k_idft_h2[p: Params](base: Pointer[UInt8, MutAnyOrigin], src: Int64, stride: Int32, shift: Int32, winv2: Int64, dst: Int64):
    """dst[k] = sum_t v[(t + shift) mod h2] winv2[k, t]; v[t] at src + t stride."""
    comptime h2 = p.h2()
    var k = _gid()
    if k >= h2:
        return
    var acc = E(0)
    for t in range(h2):
        var tt = (t + Int(shift)) % h2
        acc = f_add(acc, ext_mul[4](_e(base, Int(src) + tt * Int(stride)), _f2(base, Int(winv2) + (k * h2 + t) * 2)))
    base.unsafe_store[width=16](Int(dst) + k * 16, acc)


def k_polymul(base: Pointer[UInt8, MutAnyOrigin], a: Int64, na: Int32, b: Int64, nb: Int32, dst: Int64):
    """dst[k] = sum_i a[i] b[k - i], k < na + nb - 1."""
    var k = _gid()
    if k >= Int(na) + Int(nb) - 1:
        return
    var acc = E(0)
    var lo = max(0, k - Int(nb) + 1)
    var hi = min(k, Int(na) - 1)
    for i in range(lo, hi + 1):
        acc = f_add(acc, ext_mul[4](_e(base, Int(a) + i * 16), _e(base, Int(b) + (k - i) * 16)))
    base.unsafe_store[width=16](Int(dst) + k * 16, acc)


@always_inline
def _s[p: Params](base: Pointer[UInt8, MutAnyOrigin], p1: Int, p2: Int, m: Int) -> E:
    """Coefficient m of p1 - p2, p1 of 2 h2 - 1 and p2 of 3 h2 - 2 coefficients, zero past them."""
    comptime h2 = p.h2()
    var v = E(0)
    if m < 2 * h2 - 1:
        v = _e(base, p1 + m * 16)
    if m < 3 * h2 - 2:
        v = f_sub(v, _e(base, p2 + m * 16))
    return v


@always_inline
def _r[p: Params](base: Pointer[UInt8, MutAnyOrigin], p1: Int, p2: Int, e2: E, m: Int) -> E:
    """Coefficient m of (X2 - e2)(p1 - p2)."""
    return f_sub(_s[p](base, p1, p2, m - 1), ext_mul[4](e2, _s[p](base, p1, p2, m)))


def k_q3[p: Params](base: Pointer[UInt8, MutAnyOrigin], p1: Int64, p2: Int64, e2a: UInt8, e2b: UInt8,
                    alpha: Int64, power: Int32, dst: Int64, accumulate: Int32):
    """q_k = alpha^power (r_{k + h2} + r_{k + 2 h2}), r = (X2 - e2)(p1 - p2); k < 2 h2. Adds into dst
    when `accumulate` is nonzero."""
    comptime h2 = p.h2()
    var k = _gid()
    if k >= 2 * h2:
        return
    var e2 = ext_embed[4](F2(e2a, e2b))
    var q = ext_mul[4](ext_pow[4](_e(base, Int(alpha)), Int(power)),
                       f_add(_r[p](base, Int(p1), Int(p2), e2, k + h2), _r[p](base, Int(p1), Int(p2), e2, k + 2 * h2)))
    if accumulate != 0:
        q = f_add(q, _e(base, Int(dst) + k * 16))
    base.unsafe_store[width=16](Int(dst) + k * 16, q)


def k_eval_g2[p: Params](base: Pointer[UInt8, MutAnyOrigin], q3c: Int64, wfwd2: Int64, dst: Int64):
    """dst[j] = Q3(g2^j) = sum_{k < h2} (q_k + (-1)^j q_{k + h2}) g2^(j k), j < 2 h2 (g2^h2 = -1)."""
    comptime h2 = p.h2()
    var j = _gid()
    if j >= 2 * h2:
        return
    var acc = E(0)
    for k in range(h2):
        var lo = _e(base, Int(q3c) + k * 16)
        var hi = _e(base, Int(q3c) + (k + h2) * 16)
        var c = f_sub(lo, hi) if j % 2 == 1 else f_add(lo, hi)
        acc = f_add(acc, ext_mul[4](c, _f2(base, Int(wfwd2) + (j * h2 + k) * 2)))
    base.unsafe_store[width=16](Int(dst) + j * 16, acc)


def small_grid_accumulator[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], tab: TableLayout,
                                      z2: Int, z_end: Int, z_end_stride: Int, n_end: Int, d_end: Int,
                                      lines: Int, pac: Int, p1: Int, p2: Int, alpha: Int, power: Int, e2: F2,
                                      q3c: Int, first: Bool) raises:
    """Add accumulator `power`'s term of R2 / (X2^h2 - 1) into q3c (coefficients)."""
    comptime h2 = p.h2()
    var w = tab.base + tab.winv2
    var g = ceildiv(h2, BLOCK)
    var specs = [(z2, 16, 0), (z2, 16, 1), (z_end, z_end_stride, 0), (n_end, 16, 0), (d_end, 16, 0)]   # a, b, c, n, d
    for i in range(5):
        var line = specs[i]
        ctx.enqueue_function[k_idft_h2[p]](base, Int64(line[0]), Int32(line[1]), Int32(line[2]), Int64(w), Int64(lines + i * h2 * 16),
                                           grid_dim=g, block_dim=BLOCK)
    var a = lines
    var b = lines + h2 * 16
    var c = lines + 2 * h2 * 16
    var n = lines + 3 * h2 * 16
    var d = lines + 4 * h2 * 16
    ctx.enqueue_function[k_polymul](base, Int64(b), Int32(h2), Int64(d), Int32(h2), Int64(p1), grid_dim=ceildiv(2 * h2, BLOCK), block_dim=BLOCK)
    ctx.enqueue_function[k_polymul](base, Int64(a), Int32(h2), Int64(c), Int32(h2), Int64(pac), grid_dim=ceildiv(2 * h2, BLOCK), block_dim=BLOCK)
    ctx.enqueue_function[k_polymul](base, Int64(pac), Int32(2 * h2 - 1), Int64(n), Int32(h2), Int64(p2), grid_dim=ceildiv(3 * h2, BLOCK), block_dim=BLOCK)
    ctx.enqueue_function[k_q3[p]](base, Int64(p1), Int64(p2), e2[0], e2[1], Int64(alpha), Int32(power), Int64(q3c), Int32(0 if first else 1),
                                  grid_dim=ceildiv(2 * h2, BLOCK), block_dim=BLOCK)


def small_grid_values[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], tab: TableLayout, q3c: Int, q3: Int) raises:
    """Q3 on G2 from its coefficients."""
    ctx.enqueue_function[k_eval_g2[p]](base, Int64(q3c), Int64(tab.base + tab.wfwd2), Int64(q3),
                                       grid_dim=ceildiv(2 * p.h2(), BLOCK), block_dim=BLOCK)


# ---- host side ----

def interp_cyclic(vals: List[UInt8], off: Int, n: Int, w: F2, z: E) raises -> E:
    """P(z) for the polynomial of degree < n with values vals[off + i] on the cyclic group <w> of order
    n, by the barycentric formula: (z^n - 1) / n * sum_i v_i w^i / (z - w^i). Raises when z is in the group."""
    var acc = E(0)
    var wi = ext_embed[4](F2(1, 0))
    var we = ext_embed[4](w)
    for i in range(n):
        var v = E(0)
        for t in range(16):
            v[t] = vals[(off + i) * 16 + t]
        acc = f_add(acc, ext_mul[4](ext_mul[4](v, wi), ext_inv[4](f_sub(z, wi))))
        wi = ext_mul[4](wi, we)
    var one = E(0)
    one[0] = 1
    var n_inv = E(0)
    n_inv[0] = f_pow(SIMD[DType.uint8, 1](n % 127), 125)[0]
    return ext_mul[4](ext_mul[4](f_sub(ext_pow[4](z, n), one), n_inv), acc)
