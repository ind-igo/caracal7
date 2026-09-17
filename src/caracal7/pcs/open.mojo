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

`open` is one GEMM on backend.gemm_f2 with the e / 2 F2 lanes of every point as its rows, C[(point, lane),
column], split-K over OPEN_SPLITS slot chunks; `fold` is k_fold, one thread per slot with the e / 2 F2 lanes of E
in fp32 registers (the lane GEMM at M = 8 paid for its shared-memory staging, like the residual).
"""

from std.math import ceildiv
from max.gpu.host import DeviceContext

from caracal7.core.field import F2, E, f_add, f_sub, f_mul, f_pow, ext_mul, ext_pow, ext_inv0, ext_embed, ext_one, E_LEVEL, E_BYTES, EF, VH, e_planes, e_merge, ef_planes, ef_merge, to_f32
from caracal7.core.field import fp_reduce, fp_center, fp_canonical, fp_ext_mul
from caracal7.relations.ir import FIX_ONE, FIX_E
from caracal7.core.params import Params
from caracal7.core.tables import TableLayout, Domains
from caracal7.pcs.encode import slot_target
from caracal7.core.backend import BACKEND, APPLE8, Bytes, launch_gemm_f2, strided, frag8, mma8
from caracal7.core.bytes import Base, Buf, u16
from caracal7.core.arena import Arena
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


def k_build_queries[p: Params](base: Base, points: Int32, ftab: Buf[E_BYTES], w_z: Buf[E_BYTES]):
    """`slot_weight` on the device, one thread per (point, slot pair): the pair (2k, 2k + 1) holds the two
    coordinates of one c_x(r) (`slot_target`), so with u = Za Zb and v = Pa Pb from the point's factor
    table the weights are u + v and i (u - v); a fixed pair holds coordinate 0 of x1 = 0 and x1 = H1, two
    Za Zb. Two E products per pair where one per slot was the floor before (2026-09-17: 8.8 -> 4.8 ms at
    the chain grid, byte-identical). Lane bound: u + v below 2.8 M, within fp_canonical's range."""
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


def build_queries[p: Params](ctx: DeviceContext, arena: Arena,
                             z: Int, shifts: Int, points: Int, tab: TableLayout, d: Domains, w_tab: Int, w_z: Int) raises:
    """w_z (slot, P, e) for the P opening points derived from z: the per-point tables (P, table_len, e) into
    `w_tab`, the factor tables (P, factor_len, e) after them, then the slots."""
    comptime kt = k_point_tables[p]
    comptime kf = k_factor_tables[p]
    comptime k = k_build_queries[p]
    var ftab = w_tab + points * table_len[p]() * p.e
    ctx.enqueue_function[kt](arena.buf, Buf[E_BYTES](z), Buf[1](shifts), Int32(points), Buf[2](tab.base + tab.g1p), Buf[2](tab.base + tab.g2p),
                             d.rho1, d.rho2, Buf[E_BYTES](w_tab),
                             grid_dim=ceildiv(points * table_len[p](), BACKEND.block), block_dim=BACKEND.block)
    ctx.enqueue_function[kf](arena.buf, Int32(points), d.rho1, d.rho2, Buf[E_BYTES](w_tab), Buf[E_BYTES](ftab),
                             grid_dim=ceildiv(points * factor_len[p](), BACKEND.block), block_dim=BACKEND.block)
    ctx.enqueue_function[k](arena.buf, Int32(points), Buf[E_BYTES](ftab), Buf[E_BYTES](w_z),
                            grid_dim=ceildiv(points * p.N() // 2, BACKEND.block), block_dim=BACKEND.block)


comptime OPEN_SPLITS = 1024  # K chunks of the opening GEMM: batch = splits, so the grid is not just the column and row blocks


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
    var i = gid % E_BYTES
    var c = (gid // E_BYTES) % Int(columns)
    var pt = gid // (E_BYTES * Int(columns))
    var acc: UInt32 = 0
    for s in range(Int(splits)):
        acc += UInt32(src.load(base, ((s * Int(columns) + c) * Int(points) + pt) * E_BYTES + i))
    dst.store(base, pt * Int(dst_stride) + c * E_BYTES + i, UInt8(acc % 127))


comptime OPEN8_BN = 56        # 7 column tiles per block: the trees have at most 56 columns
comptime OPEN8_BK = 32        # slots per staging chunk
comptime OPEN8_BM = 64        # rows per block: 8 simdgroups, one 8-row tile each
comptime OPEN8_THREADS = 256


def k_open_apple8(base: Base, a: Int64, b: Int64, c: Int64, mrows: Int32, columns: Int32, points: Int32, n_total: Int32, kb_: Int32):
    """`open` on the 8x8 fp16 simdgroup op (backend.mma8): C[z, n, M'] = sum over the kb_ slots of split z of
    A'[M', k] B[k, n] mod 127. A'[M', k] = base[a + k mrows + M'] is w_z with its (point, lane, plane) bytes as rows,
    so an A tile is contiguous; B[k, n] = base[b + n n_total + k] the stored column; C lands in the partials
    (s, c, p, e) at c + ((z columns + n) points + M' // e) e + M' % e. A block covers 56 columns (grid z); each simdgroup owns 8 rows x 56 columns
    (7 accumulator pairs); a chunk stages 64 x 32 A and 32 x 56 B bytes as fp16 in threadgroup memory. Sums of
    up to 1056 products are exact in fp32: the accumulators reduce to the centered residue every 32 chunks."""
    comptime BM = OPEN8_BM
    comptime BN = OPEN8_BN
    comptime BK = OPEN8_BK
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
        # B: thread < 224 -> column tid // 4, slots (tid % 4) 8 .. + 8 of the chunk: 8 contiguous bytes
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
                    w_z: Int, points: Int, stored: Int, columns: Int, partial: Int, dst: Int, row_columns: Int) raises:
    """dst[p, c] = <w_z[p], stored(c)> for one tree; rows of the openings buffer hold `row_columns`.
    One GEMM, rows (point, lane) from w_z (slot, P, e), split-K: batch s reduces slots [s K, (s + 1) K)
    into `partial` (s, c, p, e), then one sum."""
    comptime N = p.N()
    comptime e = p.e
    comptime assert e == E_BYTES, "k_sum_splits walks E_BYTES-byte E values"
    var splits = open_splits[p]()
    var K = N // splits
    comptime if APPLE8:
        ctx.enqueue_function[k_open_apple8](arena.buf, Int64(w_z), Int64(stored), Int64(partial), Int32(points * e), Int32(columns),
                                            Int32(points), Int32(N), Int32(K),
                                            grid_dim=(splits, ceildiv(points * e, OPEN8_BM), ceildiv(columns, OPEN8_BN)), block_dim=OPEN8_THREADS)
    else:
        launch_gemm_f2[BACKEND, BACKEND.tile, Bytes[kfast_=True], 1](ctx, arena, strided(
            a=w_z, sa_m=2, sa_k=points * e, sa_z=K * points * e, b=stored, sb_k=1, sb_hi=N, sb_lo=0, sb_z=K,
            c=partial, sc_m=2, sc_hi=points * e, sc_lo=0, sc_z=columns * points * e), points * (e // 2), columns, K, batch=splits)
    var total = points * columns * e
    ctx.enqueue_function[k_sum_splits](arena.buf, Buf[1](partial), Int32(splits), Int32(columns), Int32(points), Buf[1](dst),
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
