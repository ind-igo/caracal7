"""The small grid (spec 7.3, 7.4): the chain-end residual R2 on H2 and its quotient Q3 on G2, sent
in the clear. A grand-product term (small_grid_product) is, with a = Z2, b = Z2(omega2 X2) and lines of
degree < h2 from their values on H2,

    alpha^family (X2 - e2) (b prod den - a prod num),   deg < 3 h2 with up to two lines per side;
    R2 = sum of the terms;   Q3 = R2 / (X2^h2 - 1),   deg < 2 h2.

For a permutation accumulator num = [Z(e1, X2), N(e1, X2)], den = [D(e1, X2)]; for a wiring product
(accumulate.k_wire_factors) num and den are the two slots' factor lines. Every line is evaluated on
the coset c_t = gamma2 g2^t of G2 (tables c2p, cfwd2: winv2 to coefficients, then to the 2 h2 coset values),
the terms are pointwise products there, and Q3 = R2 / (X2^h2 - 1) is a pointwise division (c^h2 - 1 is
never zero off G2; exact for an honest R2, which vanishes on H2), then cinv2 to coefficients and wfwd2
to values on G2. The DFTs are lane GEMMs on the skeleton: the E vector is the 8-lane A operand, the F2
table is B. The shifted line b is the Z2 buffer one element on: accumulate.k_z2 stores Z2(omega2^h2) = 1
after the last chain. Before 2026-09-09 the products were coefficient convolutions, O(h2^2) serial per
output; ECDSA's six wiring products took 260 ms.

Scratch (SG_* offsets in units of h2 e): lines (6, 2 h2, e) the coset values of a, b and up to four lines;
coef (h2, e); r2 (2 h2, e) R2 on the coset summed over the terms; q3c (2 h2, e) the Q3 coefficients;
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

from caracal7.core.field import F2, E, f_add, f_sub, f_mul, f_pow, ext_mul, ext_pow, ext_embed, ext_inv, ext_inv0, ext_one
from caracal7.core.params import Params
from caracal7.core.tables import TableLayout
from caracal7.core.backend import BACKEND, Tile, Strided, launch_gemm_f2, strided

comptime SG_LINES = 0       # scratch offsets in units of h2 e: six lines on the coset, 2 h2 values each
comptime SG_COEF = 12       # a line's h2 coefficients on the way to the coset
comptime SG_R2 = 13         # R2 on the coset, summed over the terms (2 h2)
comptime SG_Q3C = 15        # the Q3 coefficients (2 h2)
comptime SG_TOTAL = 17

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


def _line_on_coset[p: Params](ctx: DeviceContext, arena: Arena, tab: TableLayout, src: Int, stride: Int, sg: Int, dst: Int) raises:
    """h2 values on H2 at src (stride bytes apart) -> the line's 2 h2 values on the coset c: winv2, then cfwd2."""
    comptime h2 = p.h2()
    var coef = sg + SG_COEF * h2 * p.e
    lane_dft[p](ctx, arena, src, stride, tab.base + tab.winv2, h2, h2, coef)
    lane_dft[p](ctx, arena, coef, 16, tab.base + tab.cfwd2, 2 * h2, h2, dst)


def k_product_term[p: Params](base: Base, lines: Buf[16], nn: Int32, nd: Int32, c2p: Buf[2], e2a: UInt8, e2b: UInt8,
                              alpha: Buf[16], power: Int32, dst: Buf[16], accumulate: Int32):
    """dst[t] (+)= alpha^power (c_t - e2) (b prod den - a prod num) at coset point t < 2 h2; lines (6, 2 h2, e)
    hold a, b, the nn num lines, the nd den lines."""
    comptime n = 2 * p.h2()
    var t = global_idx.x
    if t >= n:
        return
    var a = lines.load(base, t)
    var b = lines.load(base, n + t)
    for i in range(Int(nn)):
        a = ext_mul[4](a, lines.load(base, (2 + i) * n + t))
    for i in range(Int(nd)):
        b = ext_mul[4](b, lines.load(base, (2 + Int(nn) + i) * n + t))
    var g = f_sub(ext_embed[4](c2p.load(base, t)), ext_embed[4](F2(e2a, e2b)))
    var q = ext_mul[4](ext_pow[4](alpha.load(base, 0), Int(power)), ext_mul[4](g, f_sub(b, a)))
    if accumulate != 0:
        q = f_add(q, dst.load(base, t))
    dst.store(base, t, q)


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
    var lines = sg + SG_LINES * h2 * e
    var specs: List[Tuple[Int, Int]] = [(z2, 16), (z2 + 16, 16)]
    specs.extend(num.copy())
    specs.extend(den.copy())
    for i in range(len(specs)):
        _line_on_coset[p](ctx, arena, tab, specs[i][0], specs[i][1], sg, lines + i * 2 * h2 * e)
    ctx.enqueue_function[k_product_term[p]](arena.buf, Buf[16](lines), Int32(len(num)), Int32(len(den)), Buf[2](tab.base + tab.c2p),
                                            e2[0], e2[1], Buf[16](alpha), Int32(power), Buf[16](sg + SG_R2 * h2 * e), Int32(0 if first else 1),
                                            grid_dim=ceildiv(2 * h2, B), block_dim=B)


def k_end_term[p: Params](base: Base, lines: Buf[16], two: Int32, c2p: Buf[2], e2a: UInt8, e2b: UInt8, gate: Int32,
                          alpha: Buf[16], power: Int32, chals: Buf[16], chal: Int32, coef: Int32, dst: Buf[16], accumulate: Int32):
    """dst[t] (+)= coef chal alpha^power [(c_t - e2)] A(c_t) [B(c_t)] at coset point t < 2 h2; A, B at lines."""
    comptime n = 2 * p.h2()
    var t = global_idx.x
    if t >= n:
        return
    var kappa = f_mul(ext_pow[4](alpha.load(base, 0), Int(power)), E(UInt8(coef)))
    if chal != 0:
        kappa = ext_mul[4](kappa, chals.load(base, Int(chal) - 1))
    var v = lines.load(base, t)
    if two != 0:
        v = ext_mul[4](v, lines.load(base, n + t))
    if gate != 0:
        v = ext_mul[4](v, f_sub(ext_embed[4](c2p.load(base, t)), ext_embed[4](F2(e2a, e2b))))
    var q = ext_mul[4](kappa, v)
    if accumulate != 0:
        q = f_add(q, dst.load(base, t))
    dst.store(base, t, q)


def small_grid_end[p: Params](ctx: DeviceContext, arena: Arena, tab: TableLayout, zval: Int, columns_w: Int,
                              ends: Span[UInt8, _], i: Int, sg: Int, alpha: Int, chals: Int, e2: F2, first: Bool) raises:
    """Add chain-end term i of `ends` into R2 on the coset."""
    comptime h2 = p.h2()
    comptime h1 = p.h1()
    comptime B = BACKEND.block
    var lines = sg + SG_LINES * h2 * p.e
    var ca = get_u16(ends, i * END)
    var cb = get_u16(ends, i * END + 2)
    var line_of = zval + (h1 - 1) * p.e
    _line_on_coset[p](ctx, arena, tab, line_of + (ca - columns_w) * p.N(), h1 * p.e, sg, lines)
    if cb != NONE:
        _line_on_coset[p](ctx, arena, tab, line_of + (cb - columns_w) * p.N(), h1 * p.e, sg, lines + 2 * h2 * p.e)
    ctx.enqueue_function[k_end_term[p]](arena.buf, Buf[16](lines), Int32(0 if cb == NONE else 1), Buf[2](tab.base + tab.c2p), e2[0], e2[1],
                                        Int32(ends[i * END + 8]), Buf[16](alpha), Int32(get_u16(ends, i * END + 4)), Buf[16](chals),
                                        Int32(ends[i * END + 7]), Int32(ends[i * END + 6]),
                                        Buf[16](sg + SG_R2 * h2 * p.e), Int32(0 if first else 1), grid_dim=ceildiv(2 * h2, B), block_dim=B)


def k_q3_coset[p: Params](base: Base, r2: Buf[16], c2p: Buf[2], dst: Buf[16]):
    """Q3(c_t) = R2(c_t) / (c_t^h2 - 1); c_t^h2 = gamma2^h2 (-1)^t is never 1 off G2."""
    comptime h2 = p.h2()
    var t = global_idx.x
    if t >= 2 * h2:
        return
    var d = f_sub(ext_embed[4](ext_pow[1](c2p.load(base, t), h2)), ext_one[4]())
    dst.store(base, t, ext_mul[4](r2.load(base, t), ext_inv0[4](d)))


def small_grid_values[p: Params](ctx: DeviceContext, arena: Arena, tab: TableLayout, sg: Int, q3: Int) raises:
    """Q3 on G2 from R2 on the coset: the pointwise division, the coset's inverse DFT to coefficients,
    the 2 h2-point DFT to values."""
    comptime h2 = p.h2()
    comptime B = BACKEND.block
    var q3v = sg + SG_LINES * h2 * p.e
    ctx.enqueue_function[k_q3_coset[p]](arena.buf, Buf[16](sg + SG_R2 * h2 * p.e), Buf[2](tab.base + tab.c2p), Buf[16](q3v),
                                        grid_dim=ceildiv(2 * h2, B), block_dim=B)
    lane_dft[p](ctx, arena, q3v, 16, tab.base + tab.cinv2, 2 * h2, 2 * h2, sg + SG_Q3C * h2 * p.e)
    lane_dft[p](ctx, arena, sg + SG_Q3C * h2 * p.e, 16, tab.base + tab.wfwd2, 2 * h2, 2 * h2, q3)


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
