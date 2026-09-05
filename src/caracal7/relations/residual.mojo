"""Residual grid stages (design section 4, spec 8 and 10.2, statement-layer 5) on the GEMM skeleton.

Grid G = G1 x G2, G_l = <g_l> of order 2 h_l; point j is g_l^j, even j is H_l, odd j the coset.
Buffers (bytes; slowest ... fastest):
    lde       (column, j2, j1, 2)      F2 values of every witness column on G
    ltmp      (column, k2, j1, 2)      after the axis-1 forward DFT
    families  (entry, ENTRY)           the compiled family list, kappa folded in after alpha
    residual  (j2, j1, e)              R = sum_j alpha^j R_j on G
    quotient  five E-valued scratch tables, QUOTIENT_ELEMS x e bytes (see `quotient`)
    trace_q   (3 e columns, x2, x1)    A, B, Q2 coordinate columns as values on H: witness-shaped
Every stage is a launch of backend.gemm_f2 ("shapes are GEMMs", design section 8):
    lde        axis 1: C[line, j1] = sum_k coeff[line, k] g1^(j1 k); axis 2 per column likewise
    residual   C[slot, point] = sum_entry kappa[entry][slot] * X[entry][point]; X is gathered by the
               Family loader as mult(point) * c_a(shift_a point) * c_b(shift_b point)
    quotient   q1m over G1 -> Q1 on the coset; qinv1, ginv2 -> the A, B coefficients;
               q2m, qinv2 -> the Q2 coefficients; forward DFTs to their values on H"""

from std.math import ceildiv
from max.gpu.host import DeviceContext

from caracal7.core.field import F2, E, f_add, f_mul, f_sub, ext_mul, ext_pow, ext_embed, ext_one
from caracal7.core.params import Params
from caracal7.core.tables import TableLayout
from caracal7.core.backend import BACKEND, LANE_TILE, Operands, Loader, Strided, launch_gemm_f2, strided
from caracal7.relations.ir import ENTRY, NONE, NO_BASIS, POINT
from caracal7.core.bytes import Base, Buf, u16
from caracal7.core.arena import Arena
from std.gpu import global_idx


def quotient_elems[p: Params]() -> Int:
    """E elements of quotient scratch: Q1 on the coset (h1 x G2), its axis-1 transform, the A, B
    coefficients (G2 x h1), the Q2 coefficients, the Q2 axis-1 transform, the axis-1 values of
    A, B, Q2, their values on H: 14 N."""
    return 14 * p.N()


# ---- kernels ----

def k_fold_alpha(base: Base, families: Buf[1], count: Int32, alpha: Buf[16], chals: Buf[16]):
    """kappa = coef * alpha^family * chal * b_t, one thread per entry (see `kappa_of`)."""
    var gid = Int(global_idx.x)
    if gid >= Int(count):
        return
    var ent = families.at(gid * ENTRY)
    var a = alpha.load(base, 0)
    var fam = u16(base, ent + 30)
    var kappa = f_mul(ext_pow[4](a, fam), E(base[unsafe_offset=ent + 29]))
    var chal = Int(base[unsafe_offset=ent + 32])
    if chal != 0:
        kappa = ext_mul[4](kappa, chals.load(base, chal - 1))
    for i in range(2):
        var basis = Int(base[unsafe_offset=ent + 33 + i])
        if basis != NO_BASIS:
            var b = E(0)
            b[basis] = 1
            kappa = ext_mul[4](kappa, b)
    base.unsafe_store[width=16](ent, kappa)


@always_inline
def _read[p: Params](base: Base, lde: Buf[2], at: Int, j1: Int, j2: Int) -> F2:
    """c(shift point) for the read descriptor (col, dj1, dj2) at `at`; shifts are below the domain size."""
    comptime G1 = 2 * p.h1()
    comptime G2 = 2 * p.h2()
    var col = u16(base, at)
    var a = j1 + u16(base, at + 2)
    if a >= G1:
        a -= G1
    var b = j2 + u16(base, at + 4)
    if b >= G2:
        b -= G2
    return lde.load(base, (col * G2 + b) * G1 + a)


struct Family[p: Params](Loader):
    """B[entry, point] of the residual GEMM, gathered from the LDE: D = G1, so n_hi = j2, n_lo = j1.
    aux0 = lde, aux1 = gate1, aux2 = gate2."""

    @staticmethod
    def load(base: Base, o: Operands, k: Int, n_hi: Int, n_lo: Int, z: Int) -> F2:
        var ent = Int(o.b) + k * ENTRY
        var v = _read[Self.p](base, Buf[2](Int(o.aux0)), ent + 16, n_lo, n_hi)
        if u16(base, ent + 22) != NONE:
            v = ext_mul[1](v, _read[Self.p](base, Buf[2](Int(o.aux0)), ent + 22, n_lo, n_hi))
        var mult = base[unsafe_offset=ent + 28]
        if mult == 1:
            v = ext_mul[1](v, base.unsafe_load[width=2](Int(o.aux1) + n_lo * 2))
        elif mult == 2:
            v = ext_mul[1](v, base.unsafe_load[width=2](Int(o.aux2) + n_hi * 2))
        return v


def k_values_to_trace[p: Params](base: Base, vals: Buf[1], trace: Buf[1], groups: Int32):
    """trace[q * e + tau, x] = coordinate tau of V_q(x) for x in H, q < groups, vals (groups, x, e):
    E-valued columns (A, B, Q2; the accumulators) are ordinary F-valued coordinate columns from here
    on (see docs/decisions.md)."""
    comptime N = p.N()
    comptime e = p.e
    var gid = Int(global_idx.x)
    if gid >= Int(groups) * e * N:
        return
    var c = gid // N
    var x = gid % N
    base[unsafe_offset=trace.at(gid)] = base[unsafe_offset=vals.at(((c // e) * N + x) * e + c % e)]


# ---- host orchestration ----

def lde[p: Params](ctx: DeviceContext, arena: Arena,
                   coeff: Int, columns: Int, tab: TableLayout, ltmp: Int, dst: Int) raises:
    """coeff (column, k2, k1, 2) -> dst (column, j2, j1, 2): forward DFT per axis onto G (spec 10.2).
    The twist by g^i is inside the tables g_l^(j k), so there is no separate pass."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime G1 = 2 * h1
    comptime G2 = 2 * h2
    # axis 1: rows are the (column, k2) lines, B[k1, j1] = g1^(j1 k1) read from the (j, k) table
    var o1 = strided(a=coeff, sa_m=h1 * 2, sa_k=2, b=tab.base + tab.wfwd1, sb_k=2, sb_hi=h1 * 2, sb_lo=0,
                     c=ltmp, sc_m=G1 * 2, sc_hi=2, sc_lo=0)
    launch_gemm_f2[BACKEND, BACKEND.tile, Strided, 1](ctx, arena, o1, columns * h2, G1, h1)
    # axis 2: per column, C[j2, j1] = sum_k2 g2^(j2 k2) ltmp[k2, j1]
    var o2 = strided(a=tab.base + tab.wfwd2, sa_m=2 * h2 * 2, sa_k=2, b=ltmp, sb_k=G1 * 2, sb_hi=2, sb_lo=0,
                     c=dst, sc_m=G1 * 2, sc_hi=2, sc_lo=0, sb_z=h2 * G1 * 2, sc_z=G2 * G1 * 2)
    launch_gemm_f2[BACKEND, BACKEND.tile, Strided, 1](ctx, arena, o2, G2, G1, h2, batch=columns)


def residual[p: Params](ctx: DeviceContext, arena: Arena,
                        lde_buf: Int, families: Int, count: Int, tab: TableLayout, alpha: Int, chals: Int, dst: Int) raises:
    """dst (j2, j1, e) = sum_entry kappa_entry X_entry(point): the fused pass of statement-layer 5 as
    one GEMM, A = the kappa table (8 F2 lanes x entries), B gathered from the LDE."""
    comptime G1 = 2 * p.h1()
    comptime G2 = 2 * p.h2()
    ctx.enqueue_function[k_fold_alpha](arena.buf, Buf[1](families), Int32(count), Buf[16](alpha), Buf[16](chals),
                                       grid_dim=ceildiv(count, 64), block_dim=64)
    var o = strided(a=families, sa_m=2, sa_k=ENTRY, b=families, sb_k=0, sb_hi=0, sb_lo=0,
                    c=dst, sc_m=2, sc_hi=G1 * p.e, sc_lo=p.e)
    o.aux0 = Int64(lde_buf)
    o.aux1 = Int64(tab.base + tab.gate1)
    o.aux2 = Int64(tab.base + tab.gate2)
    # ponytail: D = G1 is not a power of two, so the point split costs an integer division per gathered
    # element; a (j2, j1) 2D launch removes it when the residual shows up in the profile.
    launch_gemm_f2[BACKEND, LANE_TILE, Family[p], G1](ctx, arena, o, p.e // 2, G1 * G2, count)


def quotient[p: Params](ctx: DeviceContext, arena: Arena,
                        residual_buf: Int, tab: TableLayout, scratch: Int, trace_q: Int) raises:
    """residual -> A, B, Q2 coefficients -> their values on H -> 3 e coordinate columns as the
    trace of the quotient tree, which the level-1 encoder then treats like any witness column.
    E-valued operands are the D = 8 lane view of the skeleton."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime G1 = 2 * h1
    comptime G2 = 2 * h2
    comptime N = p.N()
    comptime e = p.e
    comptime T = BACKEND.tile
    var R = residual_buf
    var q1c = scratch                       # (t, j2, e)   Q1 on g1 H1 x G2
    var t1 = q1c + h1 * G2 * e              # (k1, j2, e)
    var q1coef = t1 + h1 * G2 * e           # (k2, k1, e)  rows k2 < h2: A; rows k2 >= h2: B
    var q2coef = q1coef + G2 * h1 * e       # (k2, k1, e)  right after B: A, B, Q2 are N e apart
    var t2 = q2coef + h2 * h1 * e           # (k1, t2, e)
    var v1 = t2 + h1 * h2 * e               # (3, x1, k2, e)
    var vals = v1 + 3 * N * e               # (3, x2, x1, e)
    # 1. Q1 on the coset from R over all of G1: q1m (t, j1)
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, arena, strided(
        a=tab.base + tab.q1m, sa_m=G1 * 2, sa_k=2, b=R, sb_k=e, sb_hi=G1 * e, sb_lo=2,
        c=q1c, sc_m=G2 * e, sc_hi=e, sc_lo=2), h1, G2 * 8, G1)
    # 2. axis 1: coset values t -> coefficients k1
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, arena, strided(
        a=tab.base + tab.qinv1, sa_m=h1 * 2, sa_k=2, b=q1c, sb_k=G2 * e, sb_hi=e, sb_lo=2,
        c=t1, sc_m=G2 * e, sc_hi=e, sc_lo=2), h1, G2 * 8, h1)
    # 3. axis 2: G2 values j2 -> coefficients k2 in [0, 2 h2): A + X2^h2 B
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, arena, strided(
        a=tab.base + tab.ginv2, sa_m=G2 * 2, sa_k=2, b=t1, sb_k=e, sb_hi=G2 * e, sb_lo=2,
        c=q1coef, sc_m=h1 * e, sc_hi=e, sc_lo=2), G2, h1 * 8, G2)
    # 4. Q2 = S1 / (-2) on H1 x g2 H2: axis 1 from the even j1 of the odd j2 rows, q2m = 63 winv1
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, arena, strided(
        a=tab.base + tab.q2m, sa_m=h1 * 2, sa_k=2, b=R + G1 * e, sb_k=2 * e, sb_hi=2 * G1 * e, sb_lo=2,
        c=t2, sc_m=h2 * e, sc_hi=e, sc_lo=2), h1, h2 * 8, h1)
    # 5. axis 2: coset values t2 -> coefficients k2
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, arena, strided(
        a=tab.base + tab.qinv2, sa_m=h2 * 2, sa_k=2, b=t2, sb_k=e, sb_hi=h2 * e, sb_lo=2,
        c=q2coef, sc_m=h1 * e, sc_hi=e, sc_lo=2), h2, h1 * 8, h2)
    # 6, 7. values on H of A, B, Q2 (batch of three): the even rows of g_l^(j k) are omega_l^(x k)
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, arena, strided(
        a=tab.base + tab.wfwd1, sa_m=2 * h1 * 2, sa_k=2, b=q1coef, sb_k=e, sb_hi=h1 * e, sb_lo=2,
        c=v1, sc_m=h2 * e, sc_hi=e, sc_lo=2, sb_z=N * e, sc_z=N * e), h1, h2 * 8, h1, batch=3)
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, arena, strided(
        a=tab.base + tab.wfwd2, sa_m=4 * h2 * 2, sa_k=2, b=v1, sb_k=e, sb_hi=h2 * e, sb_lo=2,
        c=vals, sc_m=h1 * e, sc_hi=e, sc_lo=2, sb_z=N * e, sc_z=N * e), h2, h1 * 8, h2, batch=3)
    # 8. coordinate columns as the quotient tree's trace
    comptime k8 = k_values_to_trace[p]
    ctx.enqueue_function[k8](arena.buf, Buf[1](vals), Buf[1](trace_q), Int32(3), grid_dim=ceildiv(3 * e * N, BACKEND.block), block_dim=BACKEND.block)
