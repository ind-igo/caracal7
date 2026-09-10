"""Openings and the level-1 fold (design section 4, spec 9.1).

    w_z       (slot, P, e)      the evaluation query of every opening point, on the Frobenius-real slots
    openings  (P, column, e)    alpha_{c,p} = <w_{z_p}, stored(c)>, witness columns then quotient columns
    fold_y    (slot, e)         y = sum_c beta_c stored(c) over both trees, the level-2 message

Opening point p is (g1^dj1 z1, g2^dj2 z2) for the (dj1, dj2) pair at `shifts + POINT p`, or a fixed coordinate (residual.mojo)
(residual.shift_points); point 0 is z itself. The weight of a slot (t, x1', x2, r) is L(r) times
Mon(x) + Par(x, r) at t = 0 and i (Mon(x) - Par(x, r)) at t = 1 for a pair representative x, and
Mon(x) for a fixed slot (9.1). Every factor depends on the point and one digit of the slot, so a
per-point table (z^x on each binary axis, L(r) on each odd axis; `table_entry`) is built first and a
slot costs four E products from it. The verifier reads the same table from the host (`host_table`).

`open` is one GEMM on backend.gemm_f2 with the 8 F2 lanes of every point as its rows, C[(point, lane),
column]; `fold` is k_fold, one thread per slot with the 8 E lanes in fp32 registers (the lane GEMM
at M = 8 paid for its shared-memory staging, like the residual).
"""

from std.math import ceildiv
from max.gpu.host import DeviceContext

from caracal7.core.field import F2, E, f_add, f_sub, f_mul, f_pow, ext_mul, ext_pow, ext_inv0, ext_embed, ext_one
from caracal7.core.field import fp_reduce, fp_center, fp_canonical, fp_ext_mul
from caracal7.relations.ir import FIX_ONE, FIX_E
from caracal7.core.params import Params
from caracal7.core.tables import TableLayout, Domains
from caracal7.pcs.encode import slot_target
from caracal7.core.backend import BACKEND, Bytes, launch_gemm_f2, strided
from caracal7.core.bytes import Base, Buf, u16
from caracal7.core.arena import Arena
from std.gpu import global_idx



@always_inline
def _lagrange(zeta: E, m: Int, rr: E) -> E:
    """L(r) = (r / m) (zeta^m - 1) / (zeta - r); the delta if zeta = r."""
    var one = ext_one[4]()
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


def table_len[p: Params]() -> Int:
    """Per-point table: z1^x for x < 2^a1, z2^x for x < 2^a2, L1(r1) for r1 < m1, L2(r2) for r2 < m2."""
    return (1 << p.a1) + (1 << p.a2) + p.m1 + p.m2


@always_inline
def table_entry[p: Params](i: Int, z1: E, z2: E, rho1: UInt8, rho2: UInt8) -> E:
    """Entry i of the point table for (z1, z2): one definition for the kernel and the verifier."""
    comptime A1 = 1 << p.a1
    comptime A2 = 1 << p.a2
    if i < A1:
        return ext_pow[4](z1, i)
    if i < A1 + A2:
        return ext_pow[4](z2, i - A1)
    if i < A1 + A2 + p.m1:
        return _lagrange(ext_pow[4](z1, A1), p.m1, _rho(rho1, i - A1 - A2))
    return _lagrange(ext_pow[4](z2, A2), p.m2, _rho(rho2, i - A1 - A2 - p.m1))


def host_table[p: Params](z1: E, z2: E, rho1: UInt8, rho2: UInt8) -> List[UInt8]:
    """The point table as host bytes (table_len, e); `host_base` of it is the Base `slot_weight` reads."""
    var t = List[UInt8](capacity=table_len[p]() * 16)
    for i in range(table_len[p]()):
        var v = table_entry[p](i, z1, z2, rho1, rho2)
        for j in range(16):
            t.append(v[j])
    return t^


@always_inline
def slot_weight[p: Params](slot: Int, base: Base, tab: Buf[16], off: Int, rho1: UInt8, rho2: UInt8) -> E:
    """w_z[slot] of spec 9.1 for the point whose table starts at element `off` of `tab`; shared by the kernel
    and the verifier (`host_table` + `host_base`). Every factor is a table read or a power of the odd-digit
    generator; four E products per slot."""
    comptime H1 = 1 << (p.a1 - 1)
    comptime H2 = 1 << (p.a2 - 1)
    comptime A1 = 1 << p.a1
    comptime A2 = 1 << p.a2
    var x1: Int
    var x2: Int
    var r: Int
    var coord: Int
    x1, x2, r, coord = slot_target[p](slot)
    var r1 = r % p.m1
    var r2 = r // p.m1
    var L = ext_mul[4](tab.load(base, off + A1 + A2 + r1), tab.load(base, off + A1 + A2 + p.m1 + r2))
    var mon = ext_mul[4](tab.load(base, off + x1), tab.load(base, off + A1 + x2))

    var x1p = (slot >> 1) % H1
    var x2s = ((slot >> 1) // H1) % A2
    var w: E
    if x1p == 0 and (x2s == 0 or x2s == H2):
        w = mon                                              # fixed slot: c_x(r) lies in F
    else:
        # Par(x, r) = z1^xbar1 r1^s1(x1) z2^xbar2 r2^s2(x2), with xbar = (2^a - x) mod 2^a, s(x) = (2^(7-a) x - 1) mod m
        var par = ext_one[4]()
        if x1 != 0:
            var s1 = ((1 << (7 - p.a1)) * x1 - 1) % p.m1
            par = f_mul(tab.load(base, off + A1 - x1), E(_scal(rho1, (r1 * s1) % p.m1)))
        if x2 != 0:
            var s2 = ((1 << (7 - p.a2)) * x2 - 1) % p.m2
            par = ext_mul[4](par, f_mul(tab.load(base, off + A1 + A2 - x2), E(_scal(rho2, (r2 * s2) % p.m2))))
        if coord == 0:
            w = f_add(mon, par)
        else:
            var i = E(0)
            i[1] = 1
            w = ext_mul[4](i, f_sub(mon, par))
    return ext_mul[4](w, L)


comptime EF = SIMD[DType.float32, 16]      # E on float lanes
comptime V8 = SIMD[DType.float32, 8]


@always_inline
def _slot_weight_fp[p: Params](slot: Int, base: Base, tab: Buf[16], off: Int, rho1: UInt8, rho2: UInt8) -> E:
    """`slot_weight` for the device on fp32 lanes: the same factors, every product reduced (canonical
    table values, |x| <= 190 between products, E products below 4.7 M)."""
    comptime H1 = 1 << (p.a1 - 1)
    comptime H2 = 1 << (p.a2 - 1)
    comptime A1 = 1 << p.a1
    comptime A2 = 1 << p.a2
    var x1: Int
    var x2: Int
    var r: Int
    var coord: Int
    x1, x2, r, coord = slot_target[p](slot)
    var r1 = r % p.m1
    var r2 = r // p.m1
    var L = fp_reduce(fp_ext_mul[4](tab.load(base, off + A1 + A2 + r1).cast[DType.float32](),
                                    tab.load(base, off + A1 + A2 + p.m1 + r2).cast[DType.float32]()))
    var mon = fp_reduce(fp_ext_mul[4](tab.load(base, off + x1).cast[DType.float32](),
                                      tab.load(base, off + A1 + x2).cast[DType.float32]()))
    var x1p = (slot >> 1) % H1
    var x2s = ((slot >> 1) // H1) % A2
    var w: EF
    if x1p == 0 and (x2s == 0 or x2s == H2):
        w = mon                                              # fixed slot: c_x(r) lies in F
    else:
        var par = EF(0)
        par[0] = 1
        if x1 != 0:
            var s1 = ((1 << (7 - p.a1)) * x1 - 1) % p.m1
            par = fp_reduce(tab.load(base, off + A1 - x1).cast[DType.float32]() * Float32(Int(_scal(rho1, (r1 * s1) % p.m1))))
        if x2 != 0:
            var s2 = ((1 << (7 - p.a2)) * x2 - 1) % p.m2
            var f2 = fp_reduce(tab.load(base, off + A1 + A2 - x2).cast[DType.float32]() * Float32(Int(_scal(rho2, (r2 * s2) % p.m2))))
            par = fp_reduce(fp_ext_mul[4](par, f2))
        if coord == 0:
            w = mon + par
        else:                                                # i (mon - par): (re, im) -> (-im, re) per F2 pair
            var d = (mon - par).deinterleave()
            w = (-d[1]).interleave(d[0])
    return fp_canonical(fp_ext_mul[4](w, L))


@always_inline
def _scal(rho: UInt8, k: Int) -> UInt8:
    return f_pow(SIMD[DType.uint8, 1](rho), k)[0]


@always_inline
def _rho(rho: UInt8, k: Int) -> E:
    return ext_embed[4](_scal(rho, k))


@always_inline
def _coord(base: Base, z: E, dj: Int, gtab: Buf[2], h: Int) -> E:
    """`point_coord` on the device: z g^dj from the power table, or the fixed 1 / e_l."""
    if dj == FIX_ONE:
        return ext_one[4]()
    var j = 2 * (h - 1) if dj == FIX_E else dj
    var g = ext_embed[4](gtab.load(base, j))
    return g if dj == FIX_E else ext_mul[4](z, g)


def k_point_tables[p: Params](base: Base, z: Buf[16], shifts: Buf[1], points: Int32,
                               g1p: Buf[2], g2p: Buf[2], rho1: UInt8, rho2: UInt8, tab: Buf[16]):
    comptime T = table_len[p]()
    var gid = Int(global_idx.x)
    if gid >= Int(points) * T:
        return
    var pt = gid // T
    var sh = shifts.at(pt * 4)
    var dj1 = u16(base, sh)
    var dj2 = u16(base, sh + 2)
    var z1 = _coord(base, z.load(base, 0), dj1, g1p, p.h1())
    var z2 = _coord(base, z.load(base, 1), dj2, g2p, p.h2())
    tab.store(base, gid, table_entry[p](gid % T, z1, z2, rho1, rho2))


def k_build_queries[p: Params](base: Base, points: Int32, rho1: UInt8, rho2: UInt8, tab: Buf[16], w_z: Buf[16]):
    comptime N = p.N()
    var gid = Int(global_idx.x)
    if gid >= Int(points) * N:
        return
    w_z.store(base, (gid % N) * Int(points) + gid // N, _slot_weight_fp[p](gid % N, base, tab, (gid // N) * table_len[p](), rho1, rho2))


def build_queries[p: Params](ctx: DeviceContext, arena: Arena,
                             z: Int, shifts: Int, points: Int, tab: TableLayout, d: Domains, w_tab: Int, w_z: Int) raises:
    """w_z (slot, P, e) for the P opening points derived from z: the per-point tables (P, table_len, e) into
    `w_tab`, then the slots."""
    comptime kt = k_point_tables[p]
    comptime k = k_build_queries[p]
    ctx.enqueue_function[kt](arena.buf, Buf[16](z), Buf[1](shifts), Int32(points), Buf[2](tab.base + tab.g1p), Buf[2](tab.base + tab.g2p),
                             d.rho1, d.rho2, Buf[16](w_tab),
                             grid_dim=ceildiv(points * table_len[p](), BACKEND.block), block_dim=BACKEND.block)
    ctx.enqueue_function[k](arena.buf, Int32(points), d.rho1, d.rho2, Buf[16](w_tab), Buf[16](w_z),
                            grid_dim=ceildiv(points * p.N(), BACKEND.block), block_dim=BACKEND.block)


comptime OPEN_SPLITS = 64   # K chunks of the opening GEMM: batch = splits, so the grid is not just the column and row blocks


def open_splits[p: Params]() -> Int:
    """The largest power of two <= OPEN_SPLITS dividing N, so every chunk has the same length."""
    var s = OPEN_SPLITS
    while p.N() % s != 0:
        s //= 2
    return s


def k_sum_splits(base: Base, src: Buf[1], splits: Int32, columns: Int32, points: Int32, dst: Buf[1], dst_stride: Int32, total: Int32):
    """dst (point, column, e) at row stride `dst_stride` = sum over s of src (s, column, point, e)."""
    var gid = Int(global_idx.x)
    if gid >= Int(total):
        return
    var i = gid % 16
    var c = (gid // 16) % Int(columns)
    var pt = gid // (16 * Int(columns))
    var acc: UInt32 = 0
    for s in range(Int(splits)):
        acc += UInt32(src.load(base, ((s * Int(columns) + c) * Int(points) + pt) * 16 + i))
    dst.store(base, pt * Int(dst_stride) + c * 16 + i, UInt8(acc % 127))


def open[p: Params](ctx: DeviceContext, arena: Arena,
                    w_z: Int, points: Int, stored: Int, columns: Int, partial: Int, dst: Int, row_columns: Int) raises:
    """dst[p, c] = <w_z[p], stored(c)> for one tree; rows of the openings buffer hold `row_columns`.
    One GEMM, rows (point, lane) from w_z (slot, P, e), split-K: batch s reduces slots [s K, (s + 1) K)
    into `partial` (s, c, p, e), then one sum."""
    comptime N = p.N()
    comptime e = p.e
    comptime assert e == 16, "k_sum_splits walks 16-byte E values"
    var splits = open_splits[p]()
    var K = N // splits
    launch_gemm_f2[BACKEND, BACKEND.tile, Bytes[kfast_=True], 1](ctx, arena, strided(
        a=w_z, sa_m=2, sa_k=points * e, sa_z=K * points * e, b=stored, sb_k=1, sb_hi=N, sb_lo=0, sb_z=K,
        c=partial, sc_m=2, sc_hi=points * e, sc_lo=0, sc_z=columns * points * e), points * 8, columns, K, batch=splits)
    var total = points * columns * e
    ctx.enqueue_function[k_sum_splits](arena.buf, Buf[1](partial), Int32(splits), Int32(columns), Int32(points), Buf[1](dst),
                                       Int32(row_columns * e), Int32(total),
                                       grid_dim=ceildiv(total, BACKEND.block), block_dim=BACKEND.block)


def k_fold[acc: Bool](base: Base, beta: Buf[16], stored: Buf[1], columns: Int32, n: Int32, y: Buf[16]):
    """y[slot] (+)= sum_c beta_c stored(c)[slot], one thread per slot with the 8 E lanes in fp32
    registers: stored is one F byte per (column, slot), centered to |v| <= 63, so a term is below 8 K
    and 1,024 columns stay exact; beta_c is a uniform load per column."""
    var slot = Int(global_idx.x)
    if slot >= Int(n):
        return
    var re = V8(0)
    var im = V8(0)
    comptime if acc:
        var d = y.load(base, slot).deinterleave()
        re = d[0].cast[DType.float32]()
        im = d[1].cast[DType.float32]()
    for c in range(Int(columns)):
        var b = beta.load(base, c).deinterleave()
        var v = V8(fp_center(stored.load(base, c * Int(n) + slot)))
        re = b[0].cast[DType.float32]().fma(v, re)
        im = b[1].cast[DType.float32]().fma(v, im)
    y.store(base, slot, fp_canonical(re).interleave(fp_canonical(im)))


def fold[p: Params, acc: Bool](ctx: DeviceContext, arena: Arena,
                               beta: Int, stored: Int, columns: Int, y: Int) raises:
    """y (slot, e) += sum_c beta_c stored(c) over one tree (beta at the tree's first column): one
    launch per tree, every tree after the first accumulating."""
    comptime N = p.N()
    if columns > 1024:
        raise Error("fold: the fp32 lanes hold 1,024 columns")
    comptime kf = k_fold[acc]
    ctx.enqueue_function[kf](arena.buf, Buf[16](beta), Buf[1](stored), Int32(columns), Int32(N), Buf[16](y),
                                 grid_dim=ceildiv(N, BACKEND.block), block_dim=BACKEND.block)
