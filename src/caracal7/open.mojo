"""Openings and the level-1 fold (design section 4, spec 9.1).

    w_z       (P, slot, e)      the evaluation query of every opening point, on the Frobenius-real slots
    openings  (P, column, e)    alpha_{c,p} = <w_{z_p}, stored(c)>, witness columns then quotient columns
    fold_y    (slot, e)         y = sum_c beta_c stored(c) over both trees, the level-2 message

Opening point p is (g1^dj1 z1, g2^dj2 z2) for the (dj1, dj2) pair at `shifts + POINT p`
(residual.shift_points); point 0 is z itself. The weight of a slot (t, x1', x2, r) is L(r) times
Mon(x) + Par(x, r) at t = 0 and i (Mon(x) - Par(x, r)) at t = 1 for a pair representative x, and
Mon(x) for a fixed slot (9.1). One thread per (point, slot) computes it from scratch; the
verifier evaluates the same pieces as tensor factors.

`open` and `fold` are lane GEMMs on backend.gemm_f2: C[8 F2 lanes, n] with the F byte operand
read through the Bytes loader.
"""

from std.math import ceildiv
from max.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim

from caracal7.field import F2, E, f_add, f_sub, f_mul, f_pow, ext_mul, ext_pow, ext_inv0, ext_embed
from caracal7.params import Params
from caracal7.tables import TableLayout, Domains
from caracal7.encode import slot_target
from caracal7.backend import BACKEND, LANE_TILE, Bytes, launch_gemm_f2, strided

comptime BLOCK = 256


@always_inline
def _e(base: Pointer[UInt8, MutAnyOrigin], off: Int) -> E:
    return base.unsafe_load[width=16](off)


@always_inline
def _lagrange(zeta: E, m: Int, rr: E) -> E:
    """L(r) = (r / m) (zeta^m - 1) / (zeta - r); the delta if zeta = r."""
    var one = ext_embed[4](SIMD[DType.uint8, 1](1))
    var d = f_sub(zeta, rr)
    if d == E(0):
        return one
    var num = ext_mul[4](f_mul(rr, E(UInt8(f_pow_inv_m(m)))), f_sub(ext_pow[4](zeta, m), one))
    return ext_mul[4](num, ext_inv0[4](d))


@always_inline
def f_pow_inv_m(m: Int) -> Int:
    """m^-1 in F for the tiny odd m of the grid: search, comptime-foldable."""
    for x in range(1, 127):
        if (m * x) % 127 == 1:
            return x
    return 0


@always_inline
def slot_weight[p: Params](slot: Int, z1: E, z2: E, rho1: UInt8, rho2: UInt8) -> E:
    """w_z[slot] of spec 9.1 for the point (z1, z2); shared by the kernel and the verifier."""
    comptime H1 = 1 << (p.a1 - 1)
    comptime H2 = 1 << (p.a2 - 1)
    var x1: Int
    var x2: Int
    var r: Int
    var coord: Int
    x1, x2, r, coord = slot_target[p](slot)
    var r1 = r % p.m1
    var r2 = r // p.m1
    var rr1 = _rho(rho1, r1)                                  # the odd-digit point r as a field element
    var rr2 = _rho(rho2, r2)
    var L = ext_mul[4](_lagrange(ext_pow[4](z1, 1 << p.a1), p.m1, rr1), _lagrange(ext_pow[4](z2, 1 << p.a2), p.m2, rr2))
    var mon = ext_mul[4](ext_pow[4](z1, x1), ext_pow[4](z2, x2))

    var x1p = (slot >> 1) % H1
    var x2s = ((slot >> 1) // H1) % (1 << p.a2)
    var w: E
    if x1p == 0 and (x2s == 0 or x2s == H2):
        w = mon                                              # fixed slot: c_x(r) lies in F
    else:
        # Par(x, r) = z1^xbar1 r1^s1(x1) z2^xbar2 r2^s2(x2), with xbar = (2^a - x) mod 2^a, s(x) = (2^(7-a) x - 1) mod m
        var par = ext_embed[4](SIMD[DType.uint8, 1](1))
        if x1 != 0:
            var s1 = ((1 << (7 - p.a1)) * x1 - 1) % p.m1
            par = ext_mul[4](ext_pow[4](z1, (1 << p.a1) - x1), _rho(rho1, (r1 * s1) % p.m1))
        if x2 != 0:
            var s2 = ((1 << (7 - p.a2)) * x2 - 1) % p.m2
            par = ext_mul[4](par, ext_mul[4](ext_pow[4](z2, (1 << p.a2) - x2), _rho(rho2, (r2 * s2) % p.m2)))
        if coord == 0:
            w = f_add(mon, par)
        else:
            var i = E(0)
            i[1] = 1
            w = ext_mul[4](i, f_sub(mon, par))
    return ext_mul[4](w, L)


@always_inline
def _rho(rho: UInt8, k: Int) -> E:
    return ext_embed[4](f_pow(SIMD[DType.uint8, 1](rho), k))


def k_build_queries[p: Params](base: Pointer[UInt8, MutAnyOrigin], z: Int64, shifts: Int64, points: Int32,
                                g1p: Int64, g2p: Int64, rho1: UInt8, rho2: UInt8, w_z: Int64):
    comptime N = p.N()
    var gid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if gid >= Int(points) * N:
        return
    var pt = gid // N
    var sh = Int(shifts) + pt * 4
    var dj1 = Int(base[unsafe_offset=sh]) | Int(base[unsafe_offset=sh + 1]) << 8
    var dj2 = Int(base[unsafe_offset=sh + 2]) | Int(base[unsafe_offset=sh + 3]) << 8
    var z1 = ext_mul[4](_e(base, Int(z)), ext_embed[4](base.unsafe_load[width=2](Int(g1p) + dj1 * 2)))
    var z2 = ext_mul[4](_e(base, Int(z) + p.e), ext_embed[4](base.unsafe_load[width=2](Int(g2p) + dj2 * 2)))
    base.unsafe_store[width=16](Int(w_z) + gid * p.e, slot_weight[p](gid % N, z1, z2, rho1, rho2))


def build_queries[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                             z: Int, shifts: Int, points: Int, tab: TableLayout, d: Domains, w_z: Int) raises:
    """w_z (P, slot, e) for the P opening points derived from z."""
    comptime k = k_build_queries[p]
    ctx.enqueue_function[k](base, Int64(z), Int64(shifts), Int32(points), Int64(tab.base + tab.g1p), Int64(tab.base + tab.g2p),
                            d.rho1, d.rho2, Int64(w_z),
                            grid_dim=ceildiv(points * p.N(), BLOCK), block_dim=BLOCK)


comptime OPEN_SPLITS = 64   # K chunks of one opening GEMM; the grid is P x splits blocks instead of P


def open_splits[p: Params]() -> Int:
    """The largest power of two <= OPEN_SPLITS dividing N, so every chunk has the same length."""
    var s = OPEN_SPLITS
    while p.N() % s != 0:
        s //= 2
    return s


def k_sum_splits(base: Pointer[UInt8, MutAnyOrigin], src: Int64, splits: Int32, elems: Int32, dst: Int64, dst_stride: Int32, total: Int32):
    """dst[o * dst_stride + i] = sum_s src[(o * splits + s) * elems + i] over F, for o * elems + i < total."""
    var gid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if gid >= Int(total):
        return
    var o = gid // Int(elems)
    var i = gid % Int(elems)
    var acc: UInt32 = 0
    for k in range(Int(splits)):
        acc += UInt32(base[unsafe_offset=Int(src) + (o * Int(splits) + k) * Int(elems) + i])
    base[unsafe_offset=Int(dst) + o * Int(dst_stride) + i] = UInt8(acc % 127)


def open[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                    w_z: Int, points: Int, stored: Int, columns: Int, partial: Int, dst: Int, row_columns: Int) raises:
    """dst[p, c] = <w_z[p], stored(c)> for one tree; rows of the openings buffer hold `row_columns`.
    Split-K: block (p, s) reduces slots [s K, (s + 1) K) into `partial` (p, s, c, e), then one sum."""
    comptime N = p.N()
    comptime e = p.e
    var splits = open_splits[p]()
    var K = N // splits
    launch_gemm_f2[BACKEND, LANE_TILE, Bytes, 1](ctx, base, strided(
        a=w_z, sa_m=2, sa_k=e, sa_z=K * e, b=stored, sb_k=1, sb_hi=N, sb_lo=0, sb_z=K, sb_zd=splits,
        c=partial, sc_m=2, sc_hi=e, sc_lo=0, sc_z=columns * e), e // 2, columns, K, batch=points * splits)
    var total = points * columns * e
    ctx.enqueue_function[k_sum_splits](base, Int64(partial), Int32(splits), Int32(columns * e), Int64(dst), Int32(row_columns * e), Int32(total),
                                       grid_dim=ceildiv(total, BLOCK), block_dim=BLOCK)


def fold[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                    beta: Int, stored_w: Int, columns_w: Int, stored_q: Int, columns_q: Int, y: Int) raises:
    """y (slot, e) = sum_c beta_c stored(c) over both trees: one GEMV per tree, the second accumulating."""
    comptime N = p.N()
    comptime e = p.e
    launch_gemm_f2[BACKEND, LANE_TILE, Bytes, 1](ctx, base, strided(
        a=beta, sa_m=2, sa_k=e, b=stored_w, sb_k=N, sb_hi=1, sb_lo=0,
        c=y, sc_m=2, sc_hi=e, sc_lo=0), e // 2, N, columns_w)
    launch_gemm_f2[BACKEND, LANE_TILE, Bytes, 1, acc=True](ctx, base, strided(
        a=beta + columns_w * e, sa_m=2, sa_k=e, b=stored_q, sb_k=N, sb_hi=1, sb_lo=0,
        c=y, sc_m=2, sc_hi=e, sc_lo=0), e // 2, N, columns_q)
