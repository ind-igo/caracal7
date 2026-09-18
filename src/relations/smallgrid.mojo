"""The small grid (spec 7.3, 7.4): the chain-end residual R2 on H2 and its quotient Q3 on G2, sent
in the clear. A grand-product term (small_grid_product) is, with a = Z2, b = Z2(omega2 X2) and lines of
degree < h2 from their values on H2,

    alpha^family (X2 - e2) (b prod den - a prod num),   deg < 3 h2 with up to two lines per side;
    R2 = sum of the terms;   Q3 = R2 / (X2^h2 - 1),   deg < 2 h2.

For a permutation accumulator num = [Z(e1, X2), N(e1, X2)], den = [D(e1, X2)]; for a wiring product
(accumulate.k_wire_factors) num and den are the two slots' factor lines. Every line is evaluated on
the coset c_t = gamma2 g2^t of G2 (plans inv2 to coefficients, then cfwd2p to the 2 h2 coset values),
the terms are pointwise products there, and Q3 = R2 / (X2^h2 - 1) is a pointwise division (c^h2 - 1 is
never zero off G2; exact for an honest R2, which vanishes on H2), then cinv2p to coefficients and gfwd2p
to values on G2. The DFTs are dft_axis plans over one line of e / 2 F2 lanes, the coset twists folded into
the stage tables (tables.mojo); before 2026-09-11 they were dense h2^2 tables.
The shifted line b is the Z2 buffer one element on: accumulate.k_z2 stores Z2(omega2^h2) = 1
after the last chain. Before 2026-09-09 the products were coefficient convolutions, O(h2^2) serial per
output; ECDSA's six wiring products took 260 ms.

Scratch (SmallGridLayout): lines (6, 2 h2, e) the coset values of a, b and up to four lines;
coef (h2, e); r2 (2 h2, e) R2 on the coset summed over the terms; q3c (2 h2, e) the Q3 coefficients;
q3 (2 h2, e) its values on G2 in generator order, the proof's clear vector; scr (2 h2, e) the gathered input
line and the plans' scratch.

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

from core.field import F2, E, f_add, f_sub, f_mul, f_pow, ext_mul, ext_pow, ext_embed, ext_inv, ext_inv0, ext_one, fp_ext_mul, fp_ext_pow, fp_reduce, fp_canonical, E_LEVEL, E_BYTES, EF, to_f32, E_DFT_V
from core.params import Params
from core.tables import TableLayout
from core.backend import BACKEND
from core.dft import DftPlan, dft_axis

from core.bytes import Base, Buf, u16, get_u16
from relations.ir import END, NONE, WIRE
from relations.accumulate import AccLayout
from core.arena import Arena, Bump
from std.gpu import global_idx


struct SmallGridLayout(TrivialRegisterPassable):
    """Arena offsets of the small-grid stage, all E-valued and sized by h2. The stage's callers name
    only `q3`, the output; the rest is its scratch."""
    var lines: Int      # (nl, 2 h2, e) lines on the coset: per term Z2 and its shift, then up to two numerator and two denominator lines
    var coef: Int       # (nl, h2, e)   the lines' coefficients on the way to the coset
    var r2: Int         # (2 h2, e)     R2 on the coset, summed over the terms
    var q3c: Int        # (2 h2, e)     the Q3 coefficients
    var scr: Int        # (nl, 2 h2, e) the gathered lines (h2 each) or a plan's scratch
    var q3: Int         # (2 h2, e)     Q3 on G2 in the clear, sent with the Q root

    def __init__[p: Params](out self, mut bump: Bump, lines: Int = 6):
        """`lines` the most lines a batched term set gathers (6 per wiring product); one product at a time needs 6."""
        comptime u = p.h2() * p.e
        var nl = max(6, lines)
        self.lines = bump.alloc(2 * nl * u)
        self.coef = bump.alloc(nl * u)
        self.r2 = bump.alloc(2 * u)
        self.q3c = bump.alloc(2 * u)
        self.scr = bump.alloc(2 * nl * u)
        self.q3 = bump.alloc(2 * u)


def k_gather_line[p: Params](base: Base, src: Buf[1], stride: Int32, dst: Buf[E_BYTES]):
    """dst[t] = the E value at src + t stride, t < h2: a line as contiguous rows for dft_axis."""
    var t = global_idx.x
    if t >= p.h2():
        return
    dst.store(base, t, Buf[E_BYTES](src.at(t * Int(stride))).load(base, 0))


def _line_on_coset[p: Params](ctx: DeviceContext, arena: Arena, tab: TableLayout, src: Int, stride: Int, sg: SmallGridLayout, dst: Int) raises:
    """h2 values on H2 at src (stride bytes apart) -> the line's 2 h2 values on the coset c: inv2 to
    coefficients, then cfwd2p (the twist gamma2^k inside the plan)."""
    comptime h2 = p.h2()
    comptime B = BACKEND.block
    ctx.enqueue_function[k_gather_line[p]](arena.buf, Buf[1](src), Int32(stride), Buf[E_BYTES](sg.scr), grid_dim=ceildiv(h2, B), block_dim=B)
    dft_axis[DftPlan(h2, h2), E_DFT_V](ctx, arena, sg.scr, sg.coef, sg.scr, E_BYTES // 2, 1, tab.base + tab.inv2)
    dft_axis[DftPlan(2 * h2, h2), E_DFT_V](ctx, arena, sg.coef, dst, sg.scr, E_BYTES // 2, 1, tab.base + tab.cfwd2p)


def k_product_term[p: Params](base: Base, lines: Buf[E_BYTES], nn: Int32, nd: Int32, c2p: Buf[2], e2a: UInt8, e2b: UInt8,
                              alpha: Buf[E_BYTES], power: Int32, dst: Buf[E_BYTES], accumulate: Int32):
    """dst[t] (+)= alpha^power (c_t - e2) (b prod den - a prod num) at coset point t < 2 h2; lines (6, 2 h2, e)
    hold a, b, the nn num lines, the nd den lines."""
    comptime n = 2 * p.h2()
    var t = global_idx.x
    if t >= n:
        return
    var a = to_f32(lines.load(base, t))          # fp32 lanes, every product reduced
    var b = to_f32(lines.load(base, n + t))
    for i in range(Int(nn)):
        a = fp_reduce(fp_ext_mul[E_LEVEL](a, to_f32(lines.load(base, (2 + i) * n + t))))
    for i in range(Int(nd)):
        b = fp_reduce(fp_ext_mul[E_LEVEL](b, to_f32(lines.load(base, (2 + Int(nn) + i) * n + t))))
    var g = to_f32(f_sub(ext_embed[E_LEVEL](c2p.load(base, t)), ext_embed[E_LEVEL](F2(e2a, e2b))))
    var q = fp_ext_mul[E_LEVEL](fp_ext_pow[E_LEVEL](alpha.load(base, 0), Int(power)), fp_reduce(fp_ext_mul[E_LEVEL](g, fp_reduce(b - a))))
    if accumulate != 0:
        q += to_f32(dst.load(base, t))
    dst.store(base, t, fp_canonical(q))


def small_grid_product[p: Params](ctx: DeviceContext, arena: Arena, tab: TableLayout, z2: Int,
                                  num: List[Tuple[Int, Int]], den: List[Tuple[Int, Int]],
                                  sg: SmallGridLayout, alpha: Int, power: Int, e2: F2, first: Bool) raises:
    """Add the grand-product term alpha^power (X2 - e2) (b prod den - a prod num) into q3c (coefficients);
    a, b are the Z2 line at z2 and its shift; num, den are one or two (offset, stride) lines of h2 values each."""
    comptime h2 = p.h2()
    comptime B = BACKEND.block
    comptime e = p.e
    if len(num) == 0 or len(num) > 2 or len(den) == 0 or len(den) > 2:
        raise Error("a grand-product term has one or two lines per side (degree < 3 h2)")
    var specs: List[Tuple[Int, Int]] = [(z2, e), (z2 + e, e)]
    specs.extend(num.copy())
    specs.extend(den.copy())
    for i in range(len(specs)):
        _line_on_coset[p](ctx, arena, tab, specs[i][0], specs[i][1], sg, sg.lines + i * 2 * h2 * e)
    ctx.enqueue_function[k_product_term[p]](arena.buf, Buf[E_BYTES](sg.lines), Int32(len(num)), Int32(len(den)), Buf[2](tab.base + tab.c2p),
                                            e2[0], e2[1], Buf[E_BYTES](alpha), Int32(power), Buf[E_BYTES](sg.r2), Int32(0 if first else 1),
                                            grid_dim=ceildiv(2 * h2, B), block_dim=B)


def small_grid_accumulator[p: Params](ctx: DeviceContext, arena: Arena, tab: TableLayout, A: AccLayout, k: Int, pi: Int,
                                      sg: SmallGridLayout, alpha: Int, power: Int, e2: F2, first: Bool) raises:
    """The grand-product term of accumulator k (product index pi): Z on the last row of every chain and
    N(e1, x2) against D(e1, x2)."""
    small_grid_product[p](ctx, arena, tab, A.z2_at(pi), [(A.zval_at(k) + (p.h1() - 1) * p.e, p.h1() * p.e), (A.n_end_at(pi), p.e)],
                          [(A.d_end_at(pi), p.e)], sg, alpha, power, e2, first)


def k_gather_wire_lines[p: Params](base: Base, z2: Buf[E_BYTES], wlines: Buf[E_BYTES], dst: Buf[E_BYTES], count: Int32):
    """dst (6 count, h2, e): per wiring product g its Z2 line, the shift, n0, n1, d0, d1 as contiguous rows."""
    comptime h2 = p.h2()
    var t = global_idx.x
    if t >= Int(count) * 6 * h2:
        return
    var l = t // h2
    var i = t % h2
    var g = l // 6
    var s = l % 6
    var v: E
    if s < 2:
        v = z2.load(base, g * h2 + i + s)
    else:
        v = wlines.load(base, (4 * g + s - 2) * h2 + i)
    dst.store(base, l * h2 + i, v)


def k_wire_terms[p: Params](base: Base, lines: Buf[E_BYTES], wires: Buf[1], c2p: Buf[2], e2a: UInt8, e2b: UInt8,
                            alpha: Buf[E_BYTES], dst: Buf[E_BYTES], count: Int32, accumulate: Int32):
    """dst[t] (+)= sum over the wiring products g of alpha^family(g) (c_t - e2) (b d0 d1 - a n0 n1) at coset
    point t < 2 h2; lines (6 count, 2 h2, e) as k_gather_wire_lines lays them out, on the coset."""
    comptime n = 2 * p.h2()
    var t = global_idx.x
    if t >= n:
        return
    var g_t = to_f32(f_sub(ext_embed[E_LEVEL](c2p.load(base, t)), ext_embed[E_LEVEL](F2(e2a, e2b))))
    var total = EF(0)
    for g in range(Int(count)):
        var a = to_f32(lines.load(base, (6 * g) * n + t))          # fp32 lanes, every product reduced
        var b = to_f32(lines.load(base, (6 * g + 1) * n + t))
        a = fp_reduce(fp_ext_mul[E_LEVEL](a, to_f32(lines.load(base, (6 * g + 2) * n + t))))
        a = fp_reduce(fp_ext_mul[E_LEVEL](a, to_f32(lines.load(base, (6 * g + 3) * n + t))))
        b = fp_reduce(fp_ext_mul[E_LEVEL](b, to_f32(lines.load(base, (6 * g + 4) * n + t))))
        b = fp_reduce(fp_ext_mul[E_LEVEL](b, to_f32(lines.load(base, (6 * g + 5) * n + t))))
        var power = u16(base, wires.at(g * WIRE + 4))
        total = fp_reduce(total + fp_ext_mul[E_LEVEL](fp_ext_pow[E_LEVEL](alpha.load(base, 0), power), fp_reduce(fp_ext_mul[E_LEVEL](g_t, fp_reduce(b - a)))))
    if accumulate != 0:
        total += to_f32(dst.load(base, t))
    dst.store(base, t, fp_canonical(total))


def small_grid_wiring[p: Params](ctx: DeviceContext, arena: Arena, tab: TableLayout, A: AccLayout, count: Int, pi0: Int, wires: Int,
                                 sg: SmallGridLayout, alpha: Int, e2: F2, first: Bool) raises:
    """The grand-product terms of the `count` wiring products (product indices from pi0), their factor
    lines n0 n1 against d0 d1, weighted by alpha^family each: one gather, one plan pair over all the
    lines, one term kernel (one product at a time was 40 launches of 1,152 threads per product)."""
    comptime h2 = p.h2()
    comptime B = BACKEND.block
    comptime e = p.e
    var nl = 6 * count
    ctx.enqueue_function[k_gather_wire_lines[p]](arena.buf, Buf[E_BYTES](A.z2_at(pi0)), Buf[E_BYTES](A.wlines), Buf[E_BYTES](sg.scr), Int32(count),
                                                 grid_dim=ceildiv(nl * h2, B), block_dim=B)
    dft_axis[DftPlan(h2, h2), E_DFT_V](ctx, arena, sg.scr, sg.coef, sg.scr, E_BYTES // 2, nl, tab.base + tab.inv2)
    dft_axis[DftPlan(2 * h2, h2), E_DFT_V](ctx, arena, sg.coef, sg.lines, sg.scr, E_BYTES // 2, nl, tab.base + tab.cfwd2p)
    ctx.enqueue_function[k_wire_terms[p]](arena.buf, Buf[E_BYTES](sg.lines), Buf[1](wires), Buf[2](tab.base + tab.c2p), e2[0], e2[1],
                                          Buf[E_BYTES](alpha), Buf[E_BYTES](sg.r2), Int32(count), Int32(0 if first else 1),
                                          grid_dim=ceildiv(2 * h2, B), block_dim=B)


def k_end_term[p: Params](base: Base, lines: Buf[E_BYTES], two: Int32, c2p: Buf[2], e2a: UInt8, e2b: UInt8, gate: Int32,
                          alpha: Buf[E_BYTES], power: Int32, chals: Buf[E_BYTES], chal: Int32, coef: Int32, dst: Buf[E_BYTES], accumulate: Int32):
    """dst[t] (+)= coef chal alpha^power [(c_t - e2)] A(c_t) [B(c_t)] at coset point t < 2 h2; A, B at lines."""
    comptime n = 2 * p.h2()
    var t = global_idx.x
    if t >= n:
        return
    var kappa = fp_reduce(fp_ext_pow[E_LEVEL](alpha.load(base, 0), Int(power)) * Float32(coef))   # fp32 lanes, every product reduced
    if chal != 0:
        kappa = fp_reduce(fp_ext_mul[E_LEVEL](kappa, to_f32(chals.load(base, Int(chal) - 1))))
    var v = to_f32(lines.load(base, t))
    if two != 0:
        v = fp_reduce(fp_ext_mul[E_LEVEL](v, to_f32(lines.load(base, n + t))))
    if gate != 0:
        v = fp_reduce(fp_ext_mul[E_LEVEL](v, to_f32(f_sub(ext_embed[E_LEVEL](c2p.load(base, t)), ext_embed[E_LEVEL](F2(e2a, e2b))))))
    var q = fp_ext_mul[E_LEVEL](kappa, v)
    if accumulate != 0:
        q += to_f32(dst.load(base, t))
    dst.store(base, t, fp_canonical(q))


def small_grid_end[p: Params](ctx: DeviceContext, arena: Arena, tab: TableLayout, zval: Int, columns_w: Int,
                              ends: Span[UInt8, _], i: Int, sg: SmallGridLayout, alpha: Int, chals: Int, e2: F2, first: Bool) raises:
    """Add chain-end term i of `ends` into R2 on the coset."""
    comptime h2 = p.h2()
    comptime h1 = p.h1()
    comptime B = BACKEND.block
    var lines = sg.lines
    var ca = get_u16(ends, i * END)
    var cb = get_u16(ends, i * END + 2)
    var line_of = zval + (h1 - 1) * p.e
    _line_on_coset[p](ctx, arena, tab, line_of + (ca - columns_w) * p.N(), h1 * p.e, sg, lines)
    if cb != NONE:
        _line_on_coset[p](ctx, arena, tab, line_of + (cb - columns_w) * p.N(), h1 * p.e, sg, lines + 2 * h2 * p.e)
    ctx.enqueue_function[k_end_term[p]](arena.buf, Buf[E_BYTES](lines), Int32(0 if cb == NONE else 1), Buf[2](tab.base + tab.c2p), e2[0], e2[1],
                                        Int32(ends[i * END + 8]), Buf[E_BYTES](alpha), Int32(get_u16(ends, i * END + 4)), Buf[E_BYTES](chals),
                                        Int32(ends[i * END + 7]), Int32(ends[i * END + 6]),
                                        Buf[E_BYTES](sg.r2), Int32(0 if first else 1), grid_dim=ceildiv(2 * h2, B), block_dim=B)


def k_q3_coset[p: Params](base: Base, r2: Buf[E_BYTES], c2p: Buf[2], dst: Buf[E_BYTES]):
    """Q3(c_t) = R2(c_t) / (c_t^h2 - 1); c_t^h2 = gamma2^h2 (-1)^t is never 1 off G2."""
    comptime h2 = p.h2()
    var t = global_idx.x
    if t >= 2 * h2:
        return
    var d = f_sub(ext_embed[E_LEVEL](ext_pow[1](c2p.load(base, t), h2)), ext_one[E_LEVEL]())
    dst.store(base, t, fp_canonical(fp_ext_mul[E_LEVEL](to_f32(r2.load(base, t)), to_f32(ext_inv0[E_LEVEL](d)))))


def small_grid_values[p: Params](ctx: DeviceContext, arena: Arena, tab: TableLayout, sg: SmallGridLayout) raises:
    """Q3 on G2 (sg.q3) from R2 on the coset: the pointwise division, the coset's inverse DFT to coefficients,
    the 2 h2-point DFT to values."""
    comptime h2 = p.h2()
    comptime B = BACKEND.block
    var q3v = sg.lines
    ctx.enqueue_function[k_q3_coset[p]](arena.buf, Buf[E_BYTES](sg.r2), Buf[2](tab.base + tab.c2p), Buf[E_BYTES](q3v),
                                        grid_dim=ceildiv(2 * h2, B), block_dim=B)
    dft_axis[DftPlan(2 * h2, 2 * h2), E_DFT_V](ctx, arena, q3v, sg.q3c, sg.scr, E_BYTES // 2, 1, tab.base + tab.cinv2p)
    dft_axis[DftPlan(2 * h2, 2 * h2), E_DFT_V](ctx, arena, sg.q3c, sg.q3, sg.scr, E_BYTES // 2, 1, tab.base + tab.gfwd2p)


# ---- host side ----

def interp_cyclic(vals: Span[UInt8, _], off: Int, n: Int, w: F2, z: E, width: Int = E_BYTES) raises -> E:
    """P(z) for the polynomial of degree < n with values vals[off + i] (`width` coordinates each) on the cyclic
    group <w> of order n, by the barycentric formula: (z^n - 1) / n * sum_i v_i w^i / (z - w^i). Raises when z
    is in the group."""
    var acc = E(0)
    var wi = ext_one[E_LEVEL]()
    var we = ext_embed[E_LEVEL](w)
    for i in range(n):
        var v = E(0)
        for t in range(width):
            v[t] = vals[(off + i) * width + t]
        acc = f_add(acc, ext_mul[E_LEVEL](ext_mul[E_LEVEL](v, wi), ext_inv[E_LEVEL](f_sub(z, wi))))
        wi = ext_mul[E_LEVEL](wi, we)
    var n_inv = E(0)
    n_inv[0] = f_pow(SIMD[DType.uint8, 1](n % 127), 125)[0]
    return ext_mul[E_LEVEL](ext_mul[E_LEVEL](f_sub(ext_pow[E_LEVEL](z, n), ext_one[E_LEVEL]()), n_inv), acc)
