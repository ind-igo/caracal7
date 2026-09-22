"""Openings and the level-1 fold (design section 4, spec 9.1).

    w_tab     (P, table_len, e) then (P, factor_len, e): the per-point factors of the evaluation query w_z
    open_full (P, column, e)    <w_{z_p}, stored(c)> per stored column, the three trees in order (the factored openings below)
    openings  (P, opened, e)    alpha_{c,p}: a witness column's value, an E-valued column's one value sum_t b_t <w_{z_p}, stored(c_t)>
    fold_y    (slot, e)         y = sum_c beta_c stored(c) over the trees, beta_v b_t on a coordinate column, the level-2 message

Opening point p is (g1^dj1 z1, g2^dj2 z2) for the (dj1, dj2) pair at `shifts + POINT p`, or a fixed coordinate (residual.mojo)
(residual.shift_points); point 0 is z itself. The weight of a slot (t, x1', x2, r) is L(r) times
Mon(x) + Par(x, r) at t = 0 and i (Mon(x) - Par(x, r)) at t = 1 for a pair representative x, and
Mon(x) for a fixed slot (9.1). Every factor depends on the point and one digit of the slot, so a
per-point table (z^x on each binary axis, L(r) on each odd axis; `table_entry`) is built first and a
slot costs four E products from it. The verifier reads the same table from the host (`host_table`).

`open` is one GEMM on backend.gemm_f2 with the e / 2 F2 lanes of every row as its rows, C[(row, lane),
column], split-K over slot chunks (`gemm_splits`); `fold` is k_fold, one thread per slot with the e / 2 F2 lanes of E
in fp32 registers (the lane GEMM at M = 8 paid for its shared-memory staging, like the residual).
"""

from std.math import ceildiv
from max.gpu.host import DeviceContext

from core.field import F2, E, f_add, f_sub, f_mul, f_pow, ext_mul, ext_pow, ext_inv0, ext_embed, ext_one, E_LEVEL, E_BYTES, EF, VH, e_planes, e_merge, ef_planes, ef_merge, to_f32
from core.field import fp_reduce, fp_center, fp_canonical, fp_ext_mul
from relations.ir import FIX_ONE, FIX_E, POINT
from core.params import Params
from core.tables import TableLayout, Domains
from pcs.encode import slot_target
from core.backend import BACKEND, APPLE8, Bytes, launch_gemm_f2, strided, frag8, mma8
from core.bytes import Base, Buf, u16, set_u16
from core.arena import Arena
from std.gpu import global_idx, thread_idx, block_idx, lane_id
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import row_major, stack_allocation



@always_inline
def _lagrange(zeta: E, m: Int, rr: E) -> E:
    """L(r) = (r / m) (zeta^m - 1) / (zeta - r); the delta if zeta = r."""
    var one = ext_one[E_LEVEL]()
    var d = f_sub(zeta, rr)
    if d == E(0):
        return one
    var num = ext_mul[E_LEVEL](f_mul(rr, E(UInt8(f_pow_inv_m(m)))), f_sub(ext_pow[E_LEVEL](zeta, m), one))
    return ext_mul[E_LEVEL](num, ext_inv0[E_LEVEL](d))


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
        return ext_pow[E_LEVEL](z1, i)
    if i < A1 + A2:
        return ext_pow[E_LEVEL](z2, i - A1)
    if i < A1 + A2 + p.m1:
        return _lagrange(ext_pow[E_LEVEL](z1, A1), p.m1, _rho(rho1, i - A1 - A2))
    return _lagrange(ext_pow[E_LEVEL](z2, A2), p.m2, _rho(rho2, i - A1 - A2 - p.m1))


def factor_len[p: Params]() -> Int:
    """Per-point factor table, built from the point table on the device (`k_factor_tables`): Za(x1, r1) =
    z1^x1 L1(r1) and Zb(x2, r2) = z2^x2 L2(r2) for the monomial term, Pa(x1, r1) = P1(x1, r1) L1(r1) and
    Pb(x2, r2) = P2(x2, r2) L2(r2) for the parity term. A slot pair's two weights are then the two E
    products Za Zb and Pa Pb (`k_build_queries`)."""
    comptime A1 = 1 << p.a1
    comptime A2 = 1 << p.a2
    return 2 * (A1 * p.m1 + A2 * p.m2)


def k_factor_tables[p: Params](base: Base, points: Int32, rho1: UInt8, rho2: UInt8, tab: Buf[E_BYTES], ftab: Buf[E_BYTES]):
    """One thread per factor entry of every point, from the point's table: Za[x1 m1 + r1], Zb[x2 m2 + r2],
    Pa[x1 m1 + r1], Pb[x2 m2 + r2] (`factor_len`); P(0, r) = 1."""
    comptime A1 = 1 << p.a1
    comptime A2 = 1 << p.a2
    comptime NA = A1 * p.m1
    comptime NB = A2 * p.m2
    comptime FT = factor_len[p]()
    comptime T = table_len[p]()
    var gid = Int(global_idx.x)
    if gid >= Int(points) * FT:
        return
    var pt = gid // FT
    var i = gid % FT
    var off = pt * T
    var v: E
    if i < NA:
        v = ext_mul[E_LEVEL](tab.load(base, off + i // p.m1), tab.load(base, off + A1 + A2 + i % p.m1))
    elif i < NA + NB:
        var j = i - NA
        v = ext_mul[E_LEVEL](tab.load(base, off + A1 + j // p.m2), tab.load(base, off + A1 + A2 + p.m1 + j % p.m2))
    elif i < 2 * NA + NB:
        var j = i - NA - NB
        var x1 = j // p.m1
        var r1 = j % p.m1
        var L1 = tab.load(base, off + A1 + A2 + r1)
        if x1 == 0:
            v = L1
        else:
            var s1 = ((1 << (7 - p.a1)) * x1 - 1) % p.m1
            v = ext_mul[E_LEVEL](f_mul(tab.load(base, off + A1 - x1), E(_scal(rho1, (r1 * s1) % p.m1))), L1)
    else:
        var j = i - 2 * NA - NB
        var x2 = j // p.m2
        var r2 = j % p.m2
        var L2 = tab.load(base, off + A1 + A2 + p.m1 + r2)
        if x2 == 0:
            v = L2
        else:
            var s2 = ((1 << (7 - p.a2)) * x2 - 1) % p.m2
            v = ext_mul[E_LEVEL](f_mul(tab.load(base, off + A1 + A2 - x2), E(_scal(rho2, (r2 * s2) % p.m2))), L2)
    ftab.store(base, gid, v)


def host_table[p: Params](z1: E, z2: E, rho1: UInt8, rho2: UInt8) -> List[UInt8]:
    """The point table as host bytes (table_len, e); `host_base` of it is the Base `slot_weight` reads."""
    var t = List[UInt8](capacity=table_len[p]() * E_BYTES)
    for i in range(table_len[p]()):
        var v = table_entry[p](i, z1, z2, rho1, rho2)
        for j in range(E_BYTES):
            t.append(v[j])
    return t^


@always_inline
def slot_weight[p: Params](slot: Int, base: Base, tab: Buf[E_BYTES], off: Int, rho1: UInt8, rho2: UInt8) -> E:
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
    var L = ext_mul[E_LEVEL](tab.load(base, off + A1 + A2 + r1), tab.load(base, off + A1 + A2 + p.m1 + r2))
    var mon = ext_mul[E_LEVEL](tab.load(base, off + x1), tab.load(base, off + A1 + x2))

    var x1p = (slot >> 1) % H1
    var x2s = ((slot >> 1) // H1) % A2
    var w: E
    if x1p == 0 and (x2s == 0 or x2s == H2):
        w = mon                                              # fixed slot: c_x(r) lies in F
    else:
        # Par(x, r) = z1^xbar1 r1^s1(x1) z2^xbar2 r2^s2(x2), with xbar = (2^a - x) mod 2^a, s(x) = (2^(7-a) x - 1) mod m
        var par = ext_one[E_LEVEL]()
        if x1 != 0:
            var s1 = ((1 << (7 - p.a1)) * x1 - 1) % p.m1
            par = f_mul(tab.load(base, off + A1 - x1), E(_scal(rho1, (r1 * s1) % p.m1)))
        if x2 != 0:
            var s2 = ((1 << (7 - p.a2)) * x2 - 1) % p.m2
            par = ext_mul[E_LEVEL](par, f_mul(tab.load(base, off + A1 + A2 - x2), E(_scal(rho2, (r2 * s2) % p.m2))))
        if coord == 0:
            w = f_add(mon, par)
        else:
            var i = E(0)
            i[1] = 1
            w = ext_mul[E_LEVEL](i, f_sub(mon, par))
    return ext_mul[E_LEVEL](w, L)




@always_inline
def _scal(rho: UInt8, k: Int) -> UInt8:
    return f_pow(SIMD[DType.uint8, 1](rho), k)[0]


@always_inline
def _rho(rho: UInt8, k: Int) -> E:
    return ext_embed[E_LEVEL](_scal(rho, k))


@always_inline
def _coord(base: Base, z: E, dj: Int, gtab: Buf[2], h: Int) -> E:
    """`point_coord` on the device: z g^dj from the power table, or the fixed 1 / e_l."""
    if dj == FIX_ONE:
        return ext_one[E_LEVEL]()
    var j = 2 * (h - 1) if dj == FIX_E else dj
    var g = ext_embed[E_LEVEL](gtab.load(base, j))
    return g if dj == FIX_E else ext_mul[E_LEVEL](z, g)


def k_point_tables[p: Params](base: Base, z: Buf[E_BYTES], shifts: Buf[1], points: Int32,
                               g1p: Buf[2], g2p: Buf[2], rho1: UInt8, rho2: UInt8, tab: Buf[E_BYTES]):
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


def point_tables[p: Params](ctx: DeviceContext, arena: Arena,
                            z: Int, shifts: Int, points: Int, tab: TableLayout, d: Domains, w_tab: Int) raises:
    """The per-point tables (P, table_len, e) into `w_tab`, then the factor tables (P, factor_len, e) after
    them (`ftab_at`), for the P opening points derived from z."""
    comptime kt = k_point_tables[p]
    comptime kf = k_factor_tables[p]
    ctx.enqueue_function[kt](arena.buf, Buf[E_BYTES](z), Buf[1](shifts), Int32(points), Buf[2](tab.base + tab.g1p), Buf[2](tab.base + tab.g2p),
                             d.rho1, d.rho2, Buf[E_BYTES](w_tab),
                             grid_dim=ceildiv(points * table_len[p](), BACKEND.block), block_dim=BACKEND.block)
    ctx.enqueue_function[kf](arena.buf, Int32(points), d.rho1, d.rho2, Buf[E_BYTES](w_tab), Buf[E_BYTES](ftab_at[p](w_tab, points)),
                             grid_dim=ceildiv(points * factor_len[p](), BACKEND.block), block_dim=BACKEND.block)


def ftab_at[p: Params](w_tab: Int, points: Int) -> Int:
    """The factor tables follow the point tables in the `w_tab` region."""
    return w_tab + points * table_len[p]() * p.e


# ---- the factored openings ----
#
# w_z[slot] for a pair (x1, x2, r) is Za(x1, r1) Zb(x2, r2) on c_x(r) plus Pa(x1, r1) Pb(x2, r2) on its conjugate,
# and Zb, Pb depend on the point only through z2. The points fall into Q classes by their dj2 (`point_classes`),
# so <w_z, stored(c)> is contracted in two stages: over the (x2, r2, coord) slots once per class, into
# T_Z[q][c, x1, r1] = sum Zb_q (u + i v) and T_P[q][c, x1, r1] = sum Pb_q (u - i v) (`open_stage1`, the GEMM of
# `open` on a transposed copy of the stored column, K = 2 h2), then over the h1 / 2 + 1 values of (x1, r1) per
# point (`open_stage2`, E by E). The x1 = 0 pairs of `slot_target` hold two classes of coordinates (the fixed
# c_x(r) of x1 = 0 and x1 = H1, and the x1 = H1 pairs above x2 = H2), so they contract with their own weight rows
# (`k_class_weights`: 2 Q rows for x1 >= 1, 4 Q rows for x1 = 0). The level-2 running query sum_p gamma_p w_z
# factors the same way (`running0`) on every statement. The openings take the factored path when the class
# count is small against the point count (`open_factored`; the SOD's 52 points fall into 14 classes, RSA's 18 into
# 10) and otherwise the direct GEMM over the full w_z (`build_queries`, `open_direct`).

comptime KIND_Z = 0
comptime KIND_P = 1


def point_classes(point_list: List[UInt8], points: Int) -> Tuple[List[UInt8], Int]:
    """[class of every point][a representative point of every class], u16 each: the distinct dj2 of the
    point list in order of first appearance. Returns the 4 P bytes (the second half padded) and the class count."""
    var out = List[UInt8](length=4 * points, fill=0)
    var seen = List[Int]()
    for i in range(points):
        var dj2 = Int(point_list[i * POINT + 2]) | Int(point_list[i * POINT + 3]) << 8
        var q = -1
        for j in range(len(seen)):
            if seen[j] == dj2:
                q = j
        if q < 0:
            q = len(seen)
            seen.append(dj2)
            set_u16(out, 2 * (points + q), i)
        set_u16(out, 2 * i, q)
    return (out^, len(seen))


@always_inline
def _class_of(base: Base, classes: Buf[1], pt: Int) -> Int:
    return u16(base, classes.at(2 * pt))


@always_inline
def _rep_of(base: Base, classes: Buf[1], P: Int, q: Int) -> Int:
    return u16(base, classes.at(2 * (P + q)))


def open_factored[p: Params](P: Int, Q: Int) -> Bool:
    """Whether the factored openings beat the direct GEMM: they cost about 2 Q (H1 + 1) / H1 rows against the
    direct P (the x1 = 0 block's 4 Q rows cover 1 / H1 of the transposed columns), so when the point list
    has few classes."""
    comptime H1 = 1 << (p.a1 - 1)
    return 2 * Q * (H1 + 1) < P * H1


def stage1_k[p: Params]() -> Int:
    """K of the first contraction: the (x2, r2, coord) slots of one (x1, r1), 2 h2."""
    return 2 * (1 << p.a2) * p.m2


def main_rows[p: Params]() -> Int:
    """(x1, r1) rows of a column in the transposed store with x1 >= 1."""
    return ((1 << (p.a1 - 1)) - 1) * p.m1


def k_class_weights[p: Params](base: Base, ftab: Buf[E_BYTES], classes: Buf[1], P: Int32, Q: Int32, a_main: Buf[E_BYTES], a_zero: Buf[E_BYTES]):
    """The stage-1 weights per class from its representative point's factor table, one thread per
    (q, k), k = coord + 2 (x2 + A2 r2). a_main (k, 2 Q, e): rows (q, kind), Zb / i Zb and Pb / -i Pb by coord.
    a_zero (k, 4 Q, e): rows (q, kind, half) for the x1 = 0 pairs, which hold (slot_target) the fixed real
    c_x(r) of x1 = 0 (coord 0) and x1 = H1 (coord 1) at x2 in {0, H2}, the pairs of x1 = 0 below H2, and the
    pairs of x1 = H1 at x2 - H2 above it: half 0 collects x1 = 0, half 1 x1 = H1."""
    comptime A1 = 1 << p.a1
    comptime A2 = 1 << p.a2
    comptime H2 = A2 // 2
    comptime NA = A1 * p.m1
    comptime NB = A2 * p.m2
    comptime FT = factor_len[p]()
    comptime K = 2 * NB
    var gid = Int(global_idx.x)
    if gid >= Int(Q) * K:
        return
    var q = gid // K
    var k = gid % K
    var coord = k & 1
    var x2 = (k >> 1) % A2
    var r2 = (k >> 1) // A2
    var off = _rep_of(base, classes, Int(P), q) * FT
    var i = E(0)
    i[1] = 1
    var zb = ftab.load(base, off + NA + x2 * p.m2 + r2)
    var pb = ftab.load(base, off + 2 * NA + NB + x2 * p.m2 + r2)
    var wz = zb
    var wp = pb
    if coord == 1:
        wz = ext_mul[E_LEVEL](i, zb)
        wp = f_sub(E(0), ext_mul[E_LEVEL](i, pb))
    a_main.store(base, k * 2 * Int(Q) + 2 * q + KIND_Z, wz)
    a_main.store(base, k * 2 * Int(Q) + 2 * q + KIND_P, wp)
    var z0 = E(0)
    var z1 = E(0)
    var p0 = E(0)
    var p1 = E(0)
    if x2 == 0 or x2 == H2:
        if coord == 0:
            z0 = zb
        else:
            z1 = zb
    elif x2 < H2:
        z0 = wz
        p0 = wp
    else:
        var x2t = x2 - H2
        z1 = ftab.load(base, off + NA + x2t * p.m2 + r2)
        p1 = ftab.load(base, off + 2 * NA + NB + x2t * p.m2 + r2)
        if coord == 1:
            z1 = ext_mul[E_LEVEL](i, z1)
            p1 = f_sub(E(0), ext_mul[E_LEVEL](i, p1))
    var row = k * 4 * Int(Q) + 4 * q
    a_zero.store(base, row + 2 * KIND_Z, z0)
    a_zero.store(base, row + 2 * KIND_Z + 1, z1)
    a_zero.store(base, row + 2 * KIND_P, p0)
    a_zero.store(base, row + 2 * KIND_P + 1, p1)


comptime TR_THREADS = 256
comptime TR_TILE = 8192          # bytes of `stored` one transpose block moves


def transpose_rb[p: Params]() -> Int:
    """Values of r per transpose block: TR_TILE bytes of pairs, at most all m1 m2."""
    comptime H1 = 1 << (p.a1 - 1)
    comptime A2 = 1 << p.a2
    return max(1, min(p.m1 * p.m2, TR_TILE // (A2 * 2 * H1)))


def k_transpose_stored[p: Params](base: Base, stored: Buf[1], columns: Int32, main: Buf[1], zero: Buf[1]):
    """stored (column, slot) -> main (column, x1 - 1, r1, k) for x1 >= 1 and zero (column, r1, k) for x1 = 0,
    k = coord + 2 (x2 + A2 r2). One block per (column, run of RB values of r): the block's pairs are one
    contiguous run of `stored` ((r, x2) major, x1 minor), read into threadgroup memory and written out with x2
    minor, both sides in 8-byte lanes (four pairs: four x1 of one (x2, r) in, four x2 of one (x1, r) out). The
    first version scattered two-byte stores: 144 ms on the passport's W tree, more than the contraction it
    fed; the second moved 512 bytes a block and was launch-bound; the third moved two bytes a lane and ran
    at 2 GB/s (Metal lanes want 4 or 8 bytes)."""
    comptime H1 = 1 << (p.a1 - 1)
    comptime A2 = 1 << p.a2
    comptime M = p.m1 * p.m2
    comptime RB = transpose_rb[p]()
    comptime PB = RB * A2 * H1               # pairs per block
    comptime K = stage1_k[p]()
    comptime NM = main_rows[p]()
    comptime N = p.N()
    comptime assert A2 % 4 == 0 and H1 % 4 == 0, "the transpose moves four pairs a lane"
    var tile = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[2 * PB]())
    var blk = Int(block_idx.x)
    var rb = blk % ceildiv(M, RB)
    var c = blk // ceildiv(M, RB)
    if c >= Int(columns):
        return
    var tid = Int(thread_idx.x)
    var r0 = rb * RB
    var src = c * N + 2 * H1 * A2 * r0
    for q in range(tid, PB // 4, TR_THREADS):
        if r0 + (4 * q) // (A2 * H1) < M:
            tile.ptr.unsafe_store[width=8](8 * q, base.unsafe_load[width=8](Int(stored.at(src + 8 * q))))
    barrier()
    for q in range(tid, PB // 4, TR_THREADS):
        var rl = q // (A2 * H1 // 4)
        var r = r0 + rl
        if r >= M:
            break
        var rem = q % (A2 * H1 // 4)
        var x1 = rem // (A2 // 4)
        var x2 = 4 * (rem % (A2 // 4))
        var v = SIMD[DType.uint8, 8](0)
        comptime for i in range(4):
            var pr = tile.ptr.unsafe_load[width=2](2 * ((rl * A2 + x2 + i) * H1 + x1))
            v[2 * i] = pr[0]
            v[2 * i + 1] = pr[1]
        var r1 = r % p.m1
        var r2 = r // p.m1
        var kk = 2 * (x2 + A2 * r2)
        if x1 >= 1:
            base.unsafe_store[width=8](main.at((c * NM + (x1 - 1) * p.m1 + r1) * K + kk), v)
        else:
            base.unsafe_store[width=8](zero.at((c * p.m1 + r1) * K + kk), v)


def k_stage2[p: Params](base: Base, ftab: Buf[E_BYTES], classes: Buf[1], t_main: Buf[E_BYTES], t_zero: Buf[E_BYTES],
                        P: Int32, columns: Int32, Q: Int32, dst: Buf[E_BYTES], row_columns: Int32):
    """dst[p, c] = sum over (x1, r1) of Za_p T_Z[q(p)] + Pa_p T_P[q(p)], one thread per (point, column): the
    x1 >= 1 rows from t_main (2 Q, columns NM, e), the x1 = 0 and x1 = H1 rows from t_zero (4 Q, columns m1, e).
    fp32 lanes: E by E products of canonical values, reduced every four."""
    comptime A1 = 1 << p.a1
    comptime H1 = A1 // 2
    comptime A2 = 1 << p.a2
    comptime NA = A1 * p.m1
    comptime NB = A2 * p.m2
    comptime NM = main_rows[p]()
    comptime FT = factor_len[p]()
    var gid = Int(global_idx.x)
    if gid >= Int(P) * Int(columns):
        return
    var pt = gid // Int(columns)
    var c = gid % Int(columns)
    var q = _class_of(base, classes, pt)
    var off = pt * FT
    var n_main = Int(columns) * NM
    var n_zero = Int(columns) * p.m1
    var acc = EF(0)
    var n = 0
    for x1 in range(1, H1):
        for r1 in range(p.m1):
            var j = x1 * p.m1 + r1
            var i = c * NM + (x1 - 1) * p.m1 + r1
            acc += fp_ext_mul[E_LEVEL](to_f32(ftab.load(base, off + j)), to_f32(t_main.load(base, (2 * q + KIND_Z) * n_main + i)))
            acc += fp_ext_mul[E_LEVEL](to_f32(ftab.load(base, off + NA + NB + j)), to_f32(t_main.load(base, (2 * q + KIND_P) * n_main + i)))
            n += 1
            if n % 2 == 0:
                acc = fp_reduce(acc)
    for r1 in range(p.m1):
        comptime for half in range(2):
            var j = half * H1 * p.m1 + r1
            var i = c * p.m1 + r1
            acc += fp_ext_mul[E_LEVEL](to_f32(ftab.load(base, off + j)), to_f32(t_zero.load(base, (4 * q + 2 * KIND_Z + half) * n_zero + i)))
            acc += fp_ext_mul[E_LEVEL](to_f32(ftab.load(base, off + NA + NB + j)), to_f32(t_zero.load(base, (4 * q + 2 * KIND_P + half) * n_zero + i)))
            acc = fp_reduce(acc)
    dst.store(base, pt * Int(row_columns) + c, fp_canonical(acc))


def k_class_sums[p: Params](base: Base, ftab: Buf[E_BYTES], classes: Buf[1], gamma: Buf[E_BYTES], P: Int32, Q: Int32, s_tab: Buf[E_BYTES]):
    """s_tab[(q, kind), j] = sum over the points of class q of gamma_p Za_p[j] (kind Z) or Pa_p[j] (kind P),
    j < NA: the point sums of the running query, one thread per entry."""
    comptime A1 = 1 << p.a1
    comptime A2 = 1 << p.a2
    comptime NA = A1 * p.m1
    comptime NB = A2 * p.m2
    comptime FT = factor_len[p]()
    var gid = Int(global_idx.x)
    if gid >= 2 * Int(Q) * NA:
        return
    var q = gid // (2 * NA)
    var kind = (gid // NA) % 2
    var j = gid % NA
    var acc = EF(0)
    var n = 0
    for pt in range(Int(P)):
        if _class_of(base, classes, pt) != q:
            continue
        acc += fp_ext_mul[E_LEVEL](to_f32(gamma.load(base, pt)), to_f32(ftab.load(base, pt * FT + kind * (NA + NB) + j)))
        n += 1
        if n % 4 == 0:
            acc = fp_reduce(acc)
    s_tab.store(base, gid, fp_canonical(acc))


def k_running0[p: Params](base: Base, ftab: Buf[E_BYTES], classes: Buf[1], s_tab: Buf[E_BYTES], P: Int32, Q: Int32, dst: Buf[E_BYTES]):
    """dst[slot] = sum_p gamma_p w_{z_p}[slot], one thread per pair as the weights were built before
    (2026-09-17 `k_build_queries`): u = sum_q S_Z[q](x1, r1) Zb_q(x2, r2), v likewise with S_P and Pb, the
    pair's weights u + v and i (u - v); a fixed pair takes S_Z at x1 = 0 and x1 = H1 times Zb."""
    comptime H1 = 1 << (p.a1 - 1)
    comptime H2 = 1 << (p.a2 - 1)
    comptime A1 = 1 << p.a1
    comptime A2 = 1 << p.a2
    comptime NA = A1 * p.m1
    comptime NB = A2 * p.m2
    comptime FT = factor_len[p]()
    comptime PAIRS = p.N() // 2
    var k = Int(global_idx.x)
    if k >= PAIRS:
        return
    var x1 = k % H1
    var rest = k // H1
    var x2 = rest % A2
    var r = rest // A2
    var r1 = r % p.m1
    var r2 = r // p.m1
    var fixed = False
    if x1 == 0:
        if x2 == 0 or x2 == H2:
            fixed = True
        elif x2 > H2:
            x1 = H1
            x2 -= H2
    var u = EF(0)
    var v = EF(0)
    for q in range(Int(Q)):
        var off = _rep_of(base, classes, Int(P), q) * FT
        var zb = to_f32(ftab.load(base, off + NA + x2 * p.m2 + r2))
        if fixed:
            u += fp_ext_mul[E_LEVEL](to_f32(s_tab.load(base, (2 * q + KIND_Z) * NA + r1)), zb)
            v += fp_ext_mul[E_LEVEL](to_f32(s_tab.load(base, (2 * q + KIND_Z) * NA + H1 * p.m1 + r1)), zb)
        else:
            var pb = to_f32(ftab.load(base, off + 2 * NA + NB + x2 * p.m2 + r2))
            u += fp_ext_mul[E_LEVEL](to_f32(s_tab.load(base, (2 * q + KIND_Z) * NA + x1 * p.m1 + r1)), zb)
            v += fp_ext_mul[E_LEVEL](to_f32(s_tab.load(base, (2 * q + KIND_P) * NA + x1 * p.m1 + r1)), pb)
        if q % 4 == 3:
            u = fp_reduce(u)
            v = fp_reduce(v)
    u = fp_reduce(u)
    v = fp_reduce(v)
    var w0: EF
    var w1: EF
    if fixed:
        w0 = u
        w1 = v
    else:
        w0 = u + v
        var d = ef_planes(u - v)                                 # i (u - v): (re, im) -> (-im, re) per F2 pair
        w1 = ef_merge(-d[1], d[0])
    dst.store(base, 2 * k, fp_canonical(w0))
    dst.store(base, 2 * k + 1, fp_canonical(w1))


def class_weights[p: Params](ctx: DeviceContext, arena: Arena, ftab: Int, classes: Int, P: Int, Q: Int, a_main: Int, a_zero: Int) raises:
    comptime k = k_class_weights[p]
    ctx.enqueue_function[k](arena.buf, Buf[E_BYTES](ftab), Buf[1](classes), Int32(P), Int32(Q), Buf[E_BYTES](a_main), Buf[E_BYTES](a_zero),
                            grid_dim=ceildiv(Q * stage1_k[p](), BACKEND.block), block_dim=BACKEND.block)


def running0[p: Params](ctx: DeviceContext, arena: Arena, ftab: Int, classes: Int, P: Int, Q: Int, gamma: Int, s_tab: Int, dst: Int) raises:
    """dst (slot, e) = sum_p gamma_p w_{z_p}, the level-2 running query, through the class sums in s_tab."""
    comptime A1 = 1 << p.a1
    comptime NA = A1 * p.m1
    comptime ks = k_class_sums[p]
    comptime kr = k_running0[p]
    ctx.enqueue_function[ks](arena.buf, Buf[E_BYTES](ftab), Buf[1](classes), Buf[E_BYTES](gamma), Int32(P), Int32(Q), Buf[E_BYTES](s_tab),
                             grid_dim=ceildiv(2 * Q * NA, BACKEND.block), block_dim=BACKEND.block)
    ctx.enqueue_function[kr](arena.buf, Buf[E_BYTES](ftab), Buf[1](classes), Buf[E_BYTES](s_tab), Int32(P), Int32(Q), Buf[E_BYTES](dst),
                             grid_dim=ceildiv(p.N() // 2, BACKEND.block), block_dim=BACKEND.block)


def open_transpose[p: Params](ctx: DeviceContext, arena: Arena, stored: Int, columns: Int, stored_t: Int) raises:
    """One tree's stored columns into the transposed layout of `k_transpose_stored`: the x1 >= 1 rows, then the
    x1 = 0 rows (`zero_rows_at`)."""
    comptime kt = k_transpose_stored[p]
    ctx.enqueue_function[kt](arena.buf, Buf[1](stored), Int32(columns), Buf[1](stored_t), Buf[1](zero_rows_at[p](stored_t, columns)),
                             grid_dim=columns * ceildiv(p.m1 * p.m2, transpose_rb[p]()), block_dim=TR_THREADS)


def zero_rows_at[p: Params](stored_t: Int, columns: Int) -> Int:
    return stored_t + columns * main_rows[p]() * stage1_k[p]()


def open_stage1[p: Params](ctx: DeviceContext, arena: Arena, Q: Int, columns: Int,
                           stored_t: Int, a_main: Int, a_zero: Int, t_main: Int, t_zero: Int, partial: Int) raises:
    """The two stage-1 GEMMs of one tree: the x1 >= 1 rows against a_main into t_main, the x1 = 0 rows against
    a_zero into t_zero."""
    comptime K = stage1_k[p]()
    comptime NM = main_rows[p]()
    open[p](ctx, arena, a_main, 2 * Q, stored_t, columns * NM, K, partial, t_main, columns * NM)
    open[p](ctx, arena, a_zero, 4 * Q, zero_rows_at[p](stored_t, columns), columns * p.m1, K, partial, t_zero, columns * p.m1)


def k_build_queries[p: Params](base: Base, points: Int32, ftab: Buf[E_BYTES], w_z: Buf[E_BYTES]):
    """The direct path's full w_z (slot, P, e): `slot_weight` on the device, one thread per (point, slot pair):
    the pair (2k, 2k + 1) holds the two coordinates of one c_x(r) (`slot_target`), so with u = Za Zb and
    v = Pa Pb from the point's factor table the weights are u + v and i (u - v); a fixed pair holds
    coordinate 0 of x1 = 0 and x1 = H1, two Za Zb. Lane bound: u + v below 2.8 M, within fp_canonical's range."""
    comptime H1 = 1 << (p.a1 - 1)
    comptime H2 = 1 << (p.a2 - 1)
    comptime A1 = 1 << p.a1
    comptime A2 = 1 << p.a2
    comptime NA = A1 * p.m1
    comptime NB = A2 * p.m2
    comptime PAIRS = p.N() // 2
    var gid = Int(global_idx.x)
    var P = Int(points)
    if gid >= P * PAIRS:
        return
    var pt = gid // PAIRS
    var k = gid % PAIRS
    var off = pt * factor_len[p]()
    var x1 = k % H1
    var rest = k // H1
    var x2 = rest % A2
    var r = rest // A2
    var fixed = False
    if x1 == 0:
        if x2 == 0 or x2 == H2:
            fixed = True
        elif x2 > H2:
            x1 = H1
            x2 -= H2
    var ia = off + x1 * p.m1 + r % p.m1
    var ib = off + NA + x2 * p.m2 + r // p.m1
    var zb = to_f32(ftab.load(base, ib))
    var u = fp_ext_mul[E_LEVEL](to_f32(ftab.load(base, ia)), zb)
    var w0: EF
    var w1: EF
    if fixed:
        w0 = u
        w1 = fp_ext_mul[E_LEVEL](to_f32(ftab.load(base, ia + H1 * p.m1)), zb)
    else:
        var v = fp_ext_mul[E_LEVEL](to_f32(ftab.load(base, ia + NA + NB)), to_f32(ftab.load(base, ib + NA + NB)))
        w0 = u + v
        var d = ef_planes(u - v)                                 # i (u - v): (re, im) -> (-im, re) per F2 pair
        w1 = ef_merge(-d[1], d[0])
    w_z.store(base, (2 * k) * P + pt, fp_canonical(w0))
    w_z.store(base, (2 * k + 1) * P + pt, fp_canonical(w1))


def build_queries[p: Params](ctx: DeviceContext, arena: Arena, ftab: Int, points: Int, w_z: Int) raises:
    """The direct path: w_z (slot, P, e) from the factor tables."""
    comptime k = k_build_queries[p]
    ctx.enqueue_function[k](arena.buf, Int32(points), Buf[E_BYTES](ftab), Buf[E_BYTES](w_z),
                            grid_dim=ceildiv(points * p.N() // 2, BACKEND.block), block_dim=BACKEND.block)


def open_direct[p: Params](ctx: DeviceContext, arena: Arena, w_z: Int, P: Int, stored: Int, columns: Int, partial: Int, dst: Int, row_columns: Int) raises:
    """dst[p, c] = <w_{z_p}, stored(c)> for one tree's columns as one GEMM of P e rows over the N slots."""
    open[p](ctx, arena, w_z, P, stored, columns, p.N(), partial, dst, row_columns)


def open_stage2[p: Params](ctx: DeviceContext, arena: Arena, ftab: Int, classes: Int, P: Int, Q: Int, columns: Int,
                           t_main: Int, t_zero: Int, dst: Int, row_columns: Int) raises:
    """dst[p, c] = <w_{z_p}, stored(c)> for one tree's columns from its stage-1 contractions."""
    comptime k2 = k_stage2[p]
    ctx.enqueue_function[k2](arena.buf, Buf[E_BYTES](ftab), Buf[1](classes), Buf[E_BYTES](t_main), Buf[E_BYTES](t_zero),
                             Int32(P), Int32(columns), Int32(Q), Buf[E_BYTES](dst), Int32(row_columns),
                             grid_dim=ceildiv(P * columns, BACKEND.block), block_dim=BACKEND.block)


comptime OPEN_SPLITS = 1024  # at most this many K chunks of the opening GEMM: batch = splits
comptime OPEN_BLOCKS = 2048  # and no more chunks than it takes to reach about this many blocks


def gemm_splits(rows: Int, columns: Int, K: Int) -> Int:
    """Split-K count of `open`: a power of two dividing K, at most OPEN_SPLITS, and no more than the grid
    needs for OPEN_BLOCKS blocks of OPEN8_BM rows by 64 columns."""
    var s = OPEN_SPLITS
    while K % s != 0:
        s //= 2
    var blocks = ceildiv(rows * E_BYTES, OPEN8_BM) * ceildiv(columns, 64)
    while s > 1 and s * blocks > OPEN_BLOCKS:
        s //= 2
    return s


def partial_bytes(rows: Int, columns: Int, K: Int) -> Int:
    """Bytes of the split-K partials of one `open` call."""
    return gemm_splits(rows, columns, K) * rows * columns * E_BYTES


def k_sum_splits(base: Base, src: Buf[1], splits: Int32, columns: Int32, points: Int32, dst: Buf[1], dst_stride: Int32, total: Int32):
    """dst (point, column, e) at row stride `dst_stride` = sum over s of src (s, column, point, e)."""
    var gid = Int(global_idx.x)
    if gid >= Int(total):
        return
    var i = gid % E_BYTES
    var c = (gid // E_BYTES) % Int(columns)
    var pt = gid // (E_BYTES * Int(columns))
    var acc: UInt32 = 0
    for s in range(Int(splits)):
        acc += UInt32(src.load(base, ((s * Int(columns) + c) * Int(points) + pt) * E_BYTES + i))
    dst.store(base, pt * Int(dst_stride) + c * E_BYTES + i, UInt8(acc % 127))


comptime OPEN8_BK = 32        # slots per staging chunk
comptime OPEN8_BM = 64        # rows per block: 8 simdgroups, one 8-row tile each
comptime OPEN8_THREADS = 256


def k_open_apple8[BN: Int](base: Base, a: Int64, b: Int64, c: Int64, mrows: Int32, columns: Int32, points: Int32, n_total: Int32, kb_: Int32):
    """`open` on the 8x8 fp16 simdgroup op (backend.mma8): C[z, n, M'] = sum over the kb_ slots of split z of
    A'[M', k] B[k, n] mod 127. A'[M', k] = base[a + k mrows + M'] is the query weights (w_z, or a class weight table) with their (row, lane, plane) bytes as rows,
    so an A tile is contiguous; B[k, n] = base[b + n n_total + k] the stored column; C lands in the partials
    (s, c, p, e) at c + ((z columns + n) points + M' // e) e + M' % e. A block covers BN columns (grid z), 56 or 64 (`open`
    picks the one that pads less); each simdgroup owns 8 rows x BN columns (BN // 8 accumulator pairs); a chunk
    stages 64 x 32 A and 32 x BN B bytes as fp16 in threadgroup memory. Sums of up to 1056 products are exact
    in fp32: the accumulators reduce to the centered residue every 32 chunks. Counters (2026-09-17, alone at
    the chain grid): ALU limiter 82%, fp32 utilization 42%, threadgroup loads 9%: the op is fp32 ALU work on
    the M1, the padded multiply-adds are 2.9 ms of the 6.0 at the measured 2.3 T MAC/s; 128-row blocks, 16- and
    64-slot chunks were all slower."""
    comptime BM = OPEN8_BM
    comptime BK = OPEN8_BK
    comptime assert BN % 8 == 0 and BN <= 64, "BN is whole 8-column tiles staged by NT * 32 <= 256 threads"
    comptime NT = BN // 8
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var rc = frag8(Int(lane_id()))
    var r = rc[0]
    var cb = rc[1]
    var z = Int(block_idx.x)
    var m0 = Int(block_idx.y) * BM
    var n0 = Int(block_idx.z) * BN
    var Mr = Int(mrows)
    var Nc = Int(columns)
    var KB = Int(kb_)
    var k_end = (z + 1) * KB
    var As = stack_allocation[DType.float16, address_space=AddressSpace.SHARED](row_major[BM * BK]())   # As[m][k]
    var Bs = stack_allocation[DType.float16, address_space=AddressSpace.SHARED](row_major[BK * BN]())   # Bs[k][n]
    var acc = InlineArray[SIMD[DType.float32, 2], NT](fill=SIMD[DType.float32, 2](0))

    for kc in range(ceildiv(KB, BK)):
        var k_base = z * KB + kc * BK
        # A: thread -> slot k_base + tid // 8, rows (tid % 8) 8 .. + 8: 8 contiguous bytes (row offsets are 4-aligned)
        var ka = tid // 8
        var ma = (tid % 8) * 8
        var va = SIMD[DType.uint8, 8](0)
        if k_base + ka < k_end:
            if m0 + ma + 8 <= Mr:
                va = base.unsafe_load[width=8, alignment=4](Int(a) + (k_base + ka) * Mr + m0 + ma)
            elif m0 + ma < Mr:
                for j in range(Mr - m0 - ma):
                    va[j] = base[unsafe_offset=Int(a) + (k_base + ka) * Mr + m0 + ma + j]
        var fa = va.cast[DType.float16]()
        comptime for j in range(8):
            As.ptr.unsafe_store((ma + j) * BK + ka, fa[j])
        # B: thread < NT * 32 -> column tid // 4, slots (tid % 4) 8 .. + 8 of the chunk: 8 contiguous bytes
        if tid < NT * 8 * 4:
            var nb = n0 + tid // 4
            var kb = (tid % 4) * 8
            var vb = SIMD[DType.uint8, 8](0)
            if nb < Nc:
                if k_base + kb + 8 <= k_end:
                    vb = base.unsafe_load[width=8](Int(b) + nb * Int(n_total) + k_base + kb)   # k_base is any multiple of K: no alignment
                elif k_base + kb < k_end:
                    for j in range(k_end - k_base - kb):
                        vb[j] = base[unsafe_offset=Int(b) + nb * Int(n_total) + k_base + kb + j]
            var fb = vb.cast[DType.float16]()
            comptime for j in range(8):
                Bs.ptr.unsafe_store((kb + j) * BN + nb - n0, fb[j])
        barrier()
        comptime for ks in range(BK // 8):
            var af = As.ptr.unsafe_load[width=2]((sg * 8 + r) * BK + ks * 8 + cb)
            comptime for nt in range(NT):
                var bf = Bs.ptr.unsafe_load[width=2]((ks * 8 + r) * BN + nt * 8 + cb)
                acc[nt] = mma8(af, bf, acc[nt])
        barrier()
        if (kc + 1) % 32 == 0:
            comptime for nt in range(NT):
                acc[nt] = fp_reduce(acc[nt])

    var Mp = m0 + sg * 8 + r
    if Mp < Mr:
        var row = Int(c) + ((z * Nc) * Int(points) + Mp // E_BYTES) * E_BYTES + Mp % E_BYTES
        comptime for nt in range(NT):
            var v = fp_canonical(acc[nt])
            comptime for el in range(2):
                var n = n0 + nt * 8 + cb + el
                if n < Nc:
                    base[unsafe_offset=row + n * Int(points) * E_BYTES] = v[el]


def open[p: Params](ctx: DeviceContext, arena: Arena,
                    w: Int, rows: Int, stored: Int, columns: Int, K: Int, partial: Int, dst: Int, row_columns: Int) raises:
    """dst[r, c] = <w[r], stored(c)> over K slots: w (slot, rows, e), stored (column, K) bytes, dst (rows, column, e)
    at row stride `row_columns` E values. One GEMM, rows (r, lane) from w, split-K (`gemm_splits`): batch s reduces
    slots [s Ks, (s + 1) Ks) into `partial` (s, c, r, e), then one sum."""
    comptime e = p.e
    comptime assert e == E_BYTES, "k_sum_splits walks E_BYTES-byte E values"
    var splits = gemm_splits(rows, columns, K)
    var Ks = K // splits
    comptime if APPLE8:
        # 60 columns on 56-column blocks ran a second block for 4 columns: 11.2 -> 7.0 ms alone at BN = 64
        if ceildiv(columns, 56) * 56 <= ceildiv(columns, 64) * 64:
            ctx.enqueue_function[k_open_apple8[56]](arena.buf, Int64(w), Int64(stored), Int64(partial), Int32(rows * e), Int32(columns),
                                                    Int32(rows), Int32(K), Int32(Ks),
                                                    grid_dim=(splits, ceildiv(rows * e, OPEN8_BM), ceildiv(columns, 56)), block_dim=OPEN8_THREADS)
        else:
            ctx.enqueue_function[k_open_apple8[64]](arena.buf, Int64(w), Int64(stored), Int64(partial), Int32(rows * e), Int32(columns),
                                                    Int32(rows), Int32(K), Int32(Ks),
                                                    grid_dim=(splits, ceildiv(rows * e, OPEN8_BM), ceildiv(columns, 64)), block_dim=OPEN8_THREADS)
    else:
        launch_gemm_f2[BACKEND, BACKEND.tile, Bytes[kfast_=True], 1](ctx, arena, strided(
            a=w, sa_m=2, sa_k=rows * e, sa_z=Ks * rows * e, b=stored, sb_k=1, sb_hi=K, sb_lo=0, sb_z=Ks,
            c=partial, sc_m=2, sc_hi=rows * e, sc_lo=0, sc_z=columns * rows * e), rows * (e // 2), columns, Ks, batch=splits)
    var total = rows * columns * e
    ctx.enqueue_function[k_sum_splits](arena.buf, Buf[1](partial), Int32(splits), Int32(columns), Int32(rows), Buf[1](dst),
                                       Int32(row_columns * e), Int32(total),
                                       grid_dim=ceildiv(total, BACKEND.block), block_dim=BACKEND.block)


def k_fold[acc: Bool](base: Base, beta: Buf[E_BYTES], stored: Buf[1], columns: Int32, n: Int32, y: Buf[E_BYTES]):
    """y[slot] (+)= sum_c beta_c stored(c)[slot], one thread per slot with the e / 2 F2 lanes of E in fp32
    registers: stored is one F byte per (column, slot), centered to |v| <= 63, so a term is below 8 K
    and 1,024 columns stay exact; beta_c is a uniform load per column."""
    var slot = Int(global_idx.x)
    if slot >= Int(n):
        return
    var re = VH(0)
    var im = VH(0)
    comptime if acc:
        var d = e_planes(y.load(base, slot))
        re = to_f32(d[0])
        im = to_f32(d[1])
    for c in range(Int(columns)):
        var b = e_planes(beta.load(base, c))
        var v = VH(fp_center(stored.load(base, c * Int(n) + slot)))
        re = to_f32(b[0]).fma(v, re)
        im = to_f32(b[1]).fma(v, im)
    y.store(base, slot, e_merge(fp_canonical(re), fp_canonical(im)))


def fold[p: Params, acc: Bool](ctx: DeviceContext, arena: Arena,
                               beta: Int, stored: Int, columns: Int, y: Int) raises:
    """y (slot, e) += sum_c beta_c stored(c) over one tree (beta at the tree's first column): one
    launch per tree, every tree after the first accumulating."""
    comptime N = p.N()
    if columns > 1024:
        raise Error("fold: the fp32 lanes hold 1,024 columns")
    comptime kf = k_fold[acc]
    ctx.enqueue_function[kf](arena.buf, Buf[E_BYTES](beta), Buf[1](stored), Int32(columns), Int32(N), Buf[E_BYTES](y),
                                 grid_dim=ceildiv(N, BACKEND.block), block_dim=BACKEND.block)


def k_expand_beta(base: Base, beta: Buf[E_BYTES], columns_w: Int32, columns: Int32, dst: Buf[E_BYTES]):
    """beta per stored column from beta per opened column: a witness column's own; coordinate column t of an
    E-valued column gets beta_v b_t, so the fold of the stored columns is the fold of the opened ones."""
    var c = Int(global_idx.x)
    if c >= Int(columns):
        return
    if c < Int(columns_w):
        dst.store(base, c, beta.load(base, c))
        return
    var i = c - Int(columns_w)
    var b = E(0)
    b[i % E_BYTES] = 1
    dst.store(base, c, fp_canonical(fp_ext_mul[E_LEVEL](to_f32(beta.load(base, Int(columns_w) + i // E_BYTES)), to_f32(b))))


def k_compact_openings(base: Base, full: Buf[E_BYTES], P: Int32, columns: Int32, columns_w: Int32, opened: Int32, dst: Buf[E_BYTES]):
    """dst[p, v] from the per-stored-column openings full[p, c]: a witness column's value, or
    sum_t b_t full[p, col0 + t] for an E-valued column, its one opened value."""
    var i = Int(global_idx.x)
    if i >= Int(P) * Int(opened):
        return
    var pt = i // Int(opened)
    var v = i % Int(opened)
    if v < Int(columns_w):
        dst.store(base, i, full.load(base, pt * Int(columns) + v))
        return
    var col0 = Int(columns_w) + (v - Int(columns_w)) * E_BYTES
    var acc = EF(0)
    for t in range(E_BYTES):
        var b = E(0)
        b[t] = 1
        acc += fp_ext_mul[E_LEVEL](to_f32(b), to_f32(full.load(base, pt * Int(columns) + col0 + t)))
        if t % 4 == 3:
            acc = fp_reduce(acc)
    dst.store(base, i, fp_canonical(acc))


def expand_beta(ctx: DeviceContext, arena: Arena, beta: Int, columns_w: Int, columns: Int, dst: Int) raises:
    ctx.enqueue_function[k_expand_beta](arena.buf, Buf[E_BYTES](beta), Int32(columns_w), Int32(columns), Buf[E_BYTES](dst),
                                        grid_dim=ceildiv(columns, BACKEND.block), block_dim=BACKEND.block)


def compact_openings(ctx: DeviceContext, arena: Arena, full: Int, P: Int, columns: Int, columns_w: Int, opened: Int, dst: Int) raises:
    ctx.enqueue_function[k_compact_openings](arena.buf, Buf[E_BYTES](full), Int32(P), Int32(columns), Int32(columns_w), Int32(opened),
                                             Buf[E_BYTES](dst), grid_dim=ceildiv(P * opened, BACKEND.block), block_dim=BACKEND.block)
