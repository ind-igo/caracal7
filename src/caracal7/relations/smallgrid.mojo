"""The small grid (spec 7.3, 7.4): the chain-end residual R2 on H2 and its quotient Q3 on G2, sent
in the clear. A grand-product term (small_grid_product) is, with a = Z2, b = Z2(omega2 X2) and lines of
degree < h2 from their values on H2,

    alpha^family (X2 - e2) (b prod den - a prod num),   deg < 3 h2 with up to two lines per side;
    R2 = sum of the terms;   Q3 = R2 / (X2^h2 - 1),   deg < 2 h2.

For a permutation accumulator num = [Z(e1, X2), N(e1, X2)], den = [D(e1, X2)]; for a wiring product
(accumulate.k_wire_factors) num and den are the two slots' factor lines. Everything is in coefficient form
and O(h2^2). The inverse DFTs (winv2) and the evaluation of Q3 on G2 (wfwd2, the full 2 h2-point table) are
lane GEMMs on the skeleton: the E vector is the 8-lane A operand, the F2 table is B. The coefficient
products and the division q_k = r_{k + h2} + r_{k + 2 h2} (exact for an honest R2) are one thread per
output coefficient. The shifted line b is the Z2 buffer one element on: accumulate.k_z2 stores
Z2(omega2^h2) = 1 after the last chain.

Scratch (SG_* offsets in units of h2 e): lines (6, h2, e) the coefficients of a, b and up to four lines;
pac (2 h2, e); p1 (3 h2, e); p2 (3 h2, e); q3c (2 h2, e) the Q3 coefficients summed over the terms;
q3 (2 h2, e) its values on G2 in generator order, the proof's clear vector.

The lookup (6.3) needs no term of its own: its chain-end D_end(x2) is the factor kernel's cyclic
row + 1 read (accumulate.mojo), so its line is the same (W) pair.

Chain-end terms (Shape.ends, END bytes: col_a, col_b, family u16; coef, chal, gate u8): coef chal
A(e1, X2) [B(e1, X2)] [(X2 - e2)] with A, B Z blocks (their line is zval at the chain's last row, stride
h1 e), weighted by alpha^family like the (W) pairs by theirs. The mulmod check R_A R_B = R_C is three
such terms. Degree < 3 h2 as for (W), so the same division into Q3.
TODO(memory): the memory chain-end rule of 6.4 adds a term to R2 here when a profile has memory.
"""

from std.math import ceildiv
from max.gpu.host import DeviceContext

from caracal7.core.field import F2, E, f_add, f_sub, f_mul, f_pow, ext_mul, ext_pow, ext_embed, ext_inv, ext_one
from caracal7.core.params import Params
from caracal7.core.tables import TableLayout
from caracal7.core.backend import BACKEND, Tile, Strided, launch_gemm_f2, strided

comptime SG_LINES = 0       # scratch offsets in units of h2 e (module docstring)
comptime SG_PAC = 6
comptime SG_P1 = 8
comptime SG_P2 = 11
comptime SG_Q3C = 14
comptime SG_TOTAL = 16

comptime DFT_TILE = Tile(BM=8, BN=64, BK=32, TM=1, TN=2)   # 8 lanes x a short N: 256 threads per block, unlike LANE_TILE's 32
from caracal7.core.bytes import Base, Buf, get_u16
from caracal7.relations.ir import END, NONE
from caracal7.core.arena import Arena
from std.gpu import global_idx


def lane_dft[p: Params](ctx: DeviceContext, arena: Arena, src: Int, stride: Int, table: Int, n: Int, k: Int, dst: Int) raises:
    """dst[j] = sum_i v[i] table[j, i] in E, j < n, i < k: v[i] at src + i stride, table (n, k, 2) F2, dst (n, e)."""
    launch_gemm_f2[BACKEND, DFT_TILE, Strided, 1](ctx, arena, strided(
        a=src, sa_m=2, sa_k=stride, b=table, sb_k=2, sb_hi=k * 2, sb_lo=0,
        c=dst, sc_m=2, sc_hi=p.e, sc_lo=0), p.e // 2, n, k)


def k_polymul(base: Base, a: Buf[16], na: Int32, b: Buf[16], nb: Int32, dst: Buf[16]):
    """dst[k] = sum_i a[i] b[k - i], k < na + nb - 1."""
    var k = global_idx.x
    if k >= Int(na) + Int(nb) - 1:
        return
    var acc = E(0)
    var lo = max(0, k - Int(nb) + 1)
    var hi = min(k, Int(na) - 1)
    for i in range(lo, hi + 1):
        acc = f_add(acc, ext_mul[4](a.load(base, i), b.load(base, k - i)))
    dst.store(base, k, acc)


@always_inline
def _s(base: Base, p1: Buf[16], n1: Int, p2: Buf[16], n2: Int, m: Int) -> E:
    """Coefficient m of p1 - p2, p1 of n1 and p2 of n2 coefficients, zero past them."""
    var v = E(0)
    if m < n1:
        v = p1.load(base, m)
    if m < n2:
        v = f_sub(v, p2.load(base, m))
    return v


@always_inline
def _r(base: Base, p1: Buf[16], n1: Int, p2: Buf[16], n2: Int, e2: E, m: Int) -> E:
    """Coefficient m of (X2 - e2)(p1 - p2)."""
    return f_sub(_s(base, p1, n1, p2, n2, m - 1), ext_mul[4](e2, _s(base, p1, n1, p2, n2, m)))


def k_q3[p: Params](base: Base, p1: Buf[16], n1: Int32, p2: Buf[16], n2: Int32, e2a: UInt8, e2b: UInt8,
                    alpha: Buf[16], power: Int32, dst: Buf[16], accumulate: Int32):
    """q_k = alpha^power (r_{k + h2} + r_{k + 2 h2}), r = (X2 - e2)(p1 - p2) with p1, p2 of n1, n2 coefficients
    (at most 3 h2 - 2); k < 2 h2. Adds into dst when `accumulate` is nonzero."""
    comptime h2 = p.h2()
    var k = global_idx.x
    if k >= 2 * h2:
        return
    var e2 = ext_embed[4](F2(e2a, e2b))
    var q = ext_mul[4](ext_pow[4](alpha.load(base, 0), Int(power)),
                       f_add(_r(base, p1, Int(n1), p2, Int(n2), e2, k + h2), _r(base, p1, Int(n1), p2, Int(n2), e2, k + 2 * h2)))
    if accumulate != 0:
        q = f_add(q, dst.load(base, k))
    dst.store(base, k, q)


def _line_product[p: Params](ctx: DeviceContext, arena: Arena, first: Int, others: List[Int], pac: Int, dst: Int) raises -> Int:
    """dst = the line at `first` times every line in `others` (one or two), all of h2 coefficients; returns
    the coefficient count. `pac` holds the two-line intermediate."""
    comptime h2 = p.h2()
    comptime B = BACKEND.block
    var src = first
    var n = h2
    for i in range(len(others)):
        var out = dst if i == len(others) - 1 else pac
        ctx.enqueue_function[k_polymul](arena.buf, Buf[16](src), Int32(n), Buf[16](others[i]), Int32(h2), Buf[16](out),
                                        grid_dim=ceildiv(n + h2, B), block_dim=B)
        src = out
        n += h2 - 1
    return n


def small_grid_product[p: Params](ctx: DeviceContext, arena: Arena, tab: TableLayout, z2: Int,
                                  num: List[Tuple[Int, Int]], den: List[Tuple[Int, Int]],
                                  sg: Int, alpha: Int, power: Int, e2: F2, first: Bool) raises:
    """Add the grand-product term alpha^power (X2 - e2) (b prod den - a prod num) into q3c (coefficients);
    a, b are the Z2 line at z2 and its shift; num, den are one or two (offset, stride) lines of h2 values each."""
    comptime h2 = p.h2()
    comptime B = BACKEND.block
    comptime e = p.e
    if len(num) == 0 or len(num) > 2 or len(den) == 0 or len(den) > 2:
        raise Error("a grand-product term has one or two lines per side (degree < 3 h2)")
    var w = tab.base + tab.winv2
    var lines = sg + SG_LINES * h2 * e
    var specs: List[Tuple[Int, Int]] = [(z2, 16), (z2 + 16, 16)]
    specs.extend(num.copy())
    specs.extend(den.copy())
    for i in range(len(specs)):
        lane_dft[p](ctx, arena, specs[i][0], specs[i][1], w, h2, h2, lines + i * h2 * e)
    var nums = List[Int]()
    var dens = List[Int]()
    for i in range(len(num)):
        nums.append(lines + (2 + i) * h2 * e)
    for i in range(len(den)):
        dens.append(lines + (2 + len(num) + i) * h2 * e)
    var pac = sg + SG_PAC * h2 * e
    var p1 = sg + SG_P1 * h2 * e
    var p2 = sg + SG_P2 * h2 * e
    var n1 = _line_product[p](ctx, arena, lines + h2 * e, dens, pac, p1)
    var n2 = _line_product[p](ctx, arena, lines, nums, pac, p2)
    ctx.enqueue_function[k_q3[p]](arena.buf, Buf[16](p1), Int32(n1), Buf[16](p2), Int32(n2), e2[0], e2[1], Buf[16](alpha), Int32(power),
                                  Buf[16](sg + SG_Q3C * h2 * e), Int32(0 if first else 1), grid_dim=ceildiv(2 * h2, B), block_dim=B)


def k_end_term[p: Params](base: Base, src: Buf[16], n: Int32, e2a: UInt8, e2b: UInt8, gate: Int32, alpha: Buf[16], power: Int32,
                          chals: Buf[16], chal: Int32, coef: Int32, dst: Buf[16], accumulate: Int32):
    """q_k += coef chal alpha^power (r_{k + h2} + r_{k + 2 h2}), r = [(X2 - e2)] P with P the n coefficients
    at src; k < 2 h2."""
    comptime h2 = p.h2()
    var k = global_idx.x
    if k >= 2 * h2:
        return
    var kappa = f_mul(ext_pow[4](alpha.load(base, 0), Int(power)), E(UInt8(coef)))
    if chal != 0:
        kappa = ext_mul[4](kappa, chals.load(base, Int(chal) - 1))
    var e2 = ext_embed[4](F2(e2a, e2b))
    var r = E(0)
    for m in [k + h2, k + 2 * h2]:
        var v = src.load(base, m) if m < Int(n) else E(0)
        if gate != 0:
            v = f_sub(src.load(base, m - 1) if m - 1 < Int(n) else E(0), ext_mul[4](e2, v))
        r = f_add(r, v)
    var q = ext_mul[4](kappa, r)
    if accumulate != 0:
        q = f_add(q, dst.load(base, k))
    dst.store(base, k, q)


def small_grid_end[p: Params](ctx: DeviceContext, arena: Arena, tab: TableLayout, zval: Int, columns_w: Int,
                              ends: Span[UInt8, _], i: Int, sg: Int, alpha: Int, chals: Int, e2: F2, first: Bool) raises:
    """Add chain-end term i of `ends` into q3c (coefficients)."""
    comptime h2 = p.h2()
    comptime h1 = p.h1()
    comptime B = BACKEND.block
    var w = tab.base + tab.winv2
    var lines = sg + SG_LINES * h2 * p.e
    var pac = sg + SG_PAC * h2 * p.e
    var ca = get_u16(ends, i * END)
    var cb = get_u16(ends, i * END + 2)
    var line_of = zval + (h1 - 1) * p.e
    lane_dft[p](ctx, arena, line_of + (ca - columns_w) * p.N(), h1 * p.e, w, h2, h2, lines)
    var src = lines
    var n = h2
    if cb != NONE:
        lane_dft[p](ctx, arena, line_of + (cb - columns_w) * p.N(), h1 * p.e, w, h2, h2, lines + h2 * p.e)
        ctx.enqueue_function[k_polymul](arena.buf, Buf[16](lines), Int32(h2), Buf[16](lines + h2 * p.e), Int32(h2), Buf[16](pac),
                                        grid_dim=ceildiv(2 * h2, B), block_dim=B)
        src = pac
        n = 2 * h2 - 1
    ctx.enqueue_function[k_end_term[p]](arena.buf, Buf[16](src), Int32(n), e2[0], e2[1], Int32(ends[i * END + 8]), Buf[16](alpha),
                                        Int32(get_u16(ends, i * END + 4)), Buf[16](chals), Int32(ends[i * END + 7]), Int32(ends[i * END + 6]),
                                        Buf[16](sg + SG_Q3C * h2 * p.e), Int32(0 if first else 1), grid_dim=ceildiv(2 * h2, B), block_dim=B)


def small_grid_values[p: Params](ctx: DeviceContext, arena: Arena, tab: TableLayout, sg: Int, q3: Int) raises:
    """Q3 on G2 from its coefficients: the 2 h2-point DFT."""
    lane_dft[p](ctx, arena, sg + SG_Q3C * p.h2() * p.e, 16, tab.base + tab.wfwd2, 2 * p.h2(), 2 * p.h2(), q3)


# ---- host side ----

def interp_cyclic(vals: Span[UInt8, _], off: Int, n: Int, w: F2, z: E, width: Int = 16) raises -> E:
    """P(z) for the polynomial of degree < n with values vals[off + i] (`width` coordinates each) on the cyclic
    group <w> of order n, by the barycentric formula: (z^n - 1) / n * sum_i v_i w^i / (z - w^i). Raises when z
    is in the group."""
    var acc = E(0)
    var wi = ext_one[4]()
    var we = ext_embed[4](w)
    for i in range(n):
        var v = E(0)
        for t in range(width):
            v[t] = vals[(off + i) * width + t]
        acc = f_add(acc, ext_mul[4](ext_mul[4](v, wi), ext_inv[4](f_sub(z, wi))))
        wi = ext_mul[4](wi, we)
    var n_inv = E(0)
    n_inv[0] = f_pow(SIMD[DType.uint8, 1](n % 127), 125)[0]
    return ext_mul[4](ext_mul[4](f_sub(ext_pow[4](z, n), ext_one[4]()), n_inv), acc)
