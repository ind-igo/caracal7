"""Residual grid stages (design section 4, spec 8 and 10.2, statement-layer 5) on the GEMM skeleton.

Grid G = G1 x G2, G_l = <g_l> of order 2 h_l; point j is g_l^j, even j is H_l, odd j the coset.
Buffers (bytes; slowest ... fastest):
    lde       (column, j2, j1, 2)      F2 values on G: witness, Z coordinate, then public columns (one index space)
    ltmp      (column, k2, j1, 2)      after the axis-1 forward DFT
    families  (entry, ENTRY)           the compiled family list, kappa folded in after alpha
    residual  (j2, j1, e)              R = sum_j alpha^j R_j on G
    quotient  five E-valued scratch tables, QUOTIENT_ELEMS x e bytes (see `quotient`)
    trace_q   (3 e columns, x2, x1)    A, B, Q2 coordinate columns as values on H: witness-shaped
Every stage is a launch of backend.gemm_f2 ("shapes are GEMMs", design section 8):
    lde        axis 1: C[line, j1] = sum_k coeff[line, k] g1^(j1 k); axis 2 per column likewise
    residual   the one stage that is not a GEMM launch: k_residual, one thread per point, gathers
               X[entry][point] = mult(point) * c_a(shift_a point) * c_b(shift_b point) and accumulates
               the 8 kappa lanes in registers (the GEMM skeleton's shared-memory staging cost more than
               the gather at M = 8). The 2 e basis entries of a Horner transition are not in the table it
               walks: it reads the e coordinate columns as one E value R(point) and adds
               alpha^f gate (R(omega1 x) - scale R)
    quotient   q1m over G1 -> Q1 on the coset; qinv1, ginv2 -> the A, B coefficients;
               q2m, qinv2 -> the Q2 coefficients; forward DFTs to their values on H"""

from std.math import ceildiv
from max.gpu.host import DeviceContext

from caracal7.core.field import F2, E, f_add, f_mul, f_sub, ext_mul, ext_pow, fp_center, fp_reduce, fp_canonical, fp_mul_f2
from caracal7.core.params import Params
from caracal7.core.tables import TableLayout
from caracal7.core.backend import BACKEND, Strided, launch_gemm_f2, strided
from caracal7.relations.ir import ENTRY, NONE, NO_BASIS, ACC, KIND_HORNER
from caracal7.core.bytes import Base, Buf, u16
from caracal7.core.arena import Arena
from caracal7.core.dft import dft_axis
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
    var ent = families.offset(gid * ENTRY)
    var a = alpha.load(base, 0)
    var fam = u16(base, ent.at(30))
    var kappa = f_mul(ext_pow[4](a, fam), E(ent.load(base, 29)))
    var chal = Int(ent.load(base, 32))
    if chal != 0:
        kappa = ext_mul[4](kappa, chals.load(base, chal - 1))
    for i in range(2):
        var basis = Int(ent.load(base, 33 + i))
        if basis != NO_BASIS:
            var b = E(0)
            b[basis] = 1
            kappa = ext_mul[4](kappa, b)
    Buf[16](ent.at(0)).store(base, 0, kappa)


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


@always_inline
def _z_read[p: Params](base: Base, lde: Buf[2], col: Int, j1: Int, j2: Int) -> E:
    """R(point) = sum_t b_t R_t(point) from the e coordinate columns at col: with R_t = u_t + i v_t in F2,
    coordinate 2l is u_2l - v_(2l + 1) and 2l + 1 is v_2l + u_(2l + 1) (i^2 = -1, i = b_1)."""
    comptime G1 = 2 * p.h1()
    comptime G2 = 2 * p.h2()
    var r = E(0)
    comptime for l in range(8):
        var a = lde.load(base, ((col + 2 * l) * G2 + j2) * G1 + j1)
        var b = lde.load(base, ((col + 2 * l + 1) * G2 + j2) * G1 + j1)
        r[2 * l] = f_sub(a[0], b[1])
        r[2 * l + 1] = f_add(a[1], b[0])
    return r


def k_residual[p: Params](base: Base, lde: Buf[2], fam: Buf[1], count: Int32, gate1: Buf[2], gate2: Buf[2],
                          families: Buf[1], accs: Buf[1], n_accs: Int32, dst: Buf[16]):
    """dst[point] = sum over entries of kappa X(point) + the Horner transitions, one thread per point
    with the 8 kappa lanes accumulated in registers: X = mult(point) c_a(shift_a point) c_b(shift_b point)
    is gathered once and multiplies the E kappa lane by lane (F2 times F2 per lane on fp32 lanes, 128
    terms between reductions like gemm_f2). Threads walk the odd rows, then the odd columns of the even rows; R is
    zero on H x H for a satisfied statement, so the last quadrant's threads write zero. A Horner
    accumulator adds gate1(j1) (kappa_A R(omega1 x) + kappa_B R(x)) with kappa_A = alpha^f and
    kappa_B = -alpha^f scale, the folded kappas of its basis-0 entries (ir.Families.horner puts the
    2 e basis entries just before the ingest range), R read from the e coordinate columns as one E."""
    comptime G1 = 2 * p.h1()
    comptime G2 = 2 * p.h2()
    comptime Q1 = G1 * p.h2()
    comptime Q2 = p.h1() * p.h2()
    comptime assert BACKEND.max_terms & (BACKEND.max_terms - 1) == 0, "the reduction cadence masks k"
    var gid = Int(global_idx.x)
    if gid >= G1 * G2:
        return
    var j1: Int
    var j2: Int
    if gid < Q1:
        j2 = 2 * (gid // G1) + 1
        j1 = gid % G1
    elif gid < Q1 + Q2:
        j2 = 2 * ((gid - Q1) // p.h1())
        j1 = 2 * ((gid - Q1) % p.h1()) + 1
    else:
        j2 = 2 * ((gid - Q1 - Q2) // p.h1())
        j1 = 2 * ((gid - Q1 - Q2) % p.h1())
        dst.store(base, j2 * G1 + j1, E(0))
        return
    var g1 = gate1.load(base, j1)
    var g1f = fp_center(g1)
    var g2f = fp_center(gate2.load(base, j2))
    var re = SIMD[DType.float32, 8](0)
    var im = SIMD[DType.float32, 8](0)
    # fp32 lanes: |c| <= 126, times a second read <= 31.7 K, times a centered gate <= 4 M, reduced to
    # |v| <= 190; a term kappa v is below 48 K, 128 of them below 6.2 M, exact in fp32
    for k in range(Int(count)):
        var ent = fam.at(k * ENTRY)
        var v = _read[p](base, lde, ent + 16, j1, j2).cast[DType.float32]()
        if u16(base, ent + 22) != NONE:
            v = fp_mul_f2(v, _read[p](base, lde, ent + 22, j1, j2).cast[DType.float32]())
        var mult = fam.load(base, k * ENTRY + 28)
        if mult == 1:
            v = fp_mul_f2(v, g1f)
        elif mult == 2:
            v = fp_mul_f2(v, g2f)
        v = fp_reduce(v)
        var kap = Buf[16](ent).load(base, 0).deinterleave()
        var kre = kap[0].cast[DType.float32]()
        var kim = kap[1].cast[DType.float32]()
        var v0 = SIMD[DType.float32, 8](v[0])
        var v1 = SIMD[DType.float32, 8](v[1])
        re = kim.fma(-v1, kre.fma(v0, re))
        im = kim.fma(v0, kre.fma(v1, im))
        if (k & (BACKEND.max_terms - 1)) == BACKEND.max_terms - 1:
            re = fp_reduce(re)
            im = fp_reduce(im)
    var acc = fp_canonical(re).interleave(fp_canonical(im))
    var jn = j1 + 2
    if jn >= G1:
        jn -= G1
    for k in range(Int(n_accs)):
        var d = accs.offset(k * ACC)
        if Int(d.load(base, 38)) != KIND_HORNER:
            continue
        var col = u16(base, d.at(0))
        var first = u16(base, d.at(2))
        var ka = Buf[16](families.at((first - 32) * ENTRY)).load(base, 0)
        var kb = Buf[16](families.at((first - 31) * ENTRY)).load(base, 0)
        var v = f_add(ext_mul[4](ka, _z_read[p](base, lde, col, jn, j2)), ext_mul[4](kb, _z_read[p](base, lde, col, j1, j2)))
        comptime for l in range(8):                      # times the gate, an F2 scalar, lane by lane
            var w = ext_mul[1](F2(v[2 * l], v[2 * l + 1]), g1)
            v[2 * l] = w[0]
            v[2 * l + 1] = w[1]
        acc = f_add(acc, v)
    dst.store(base, j2 * G1 + j1, acc)


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
    trace.store(base, gid, vals.load(base, ((c // e) * N + x) * e + c % e))


# ---- host orchestration ----

def lde[p: Params](ctx: DeviceContext, arena: Arena,
                   coeff: Int, columns: Int, tab: TableLayout, ltmp: Int, dst: Int) raises:
    """coeff (column, k2, k1, 2) -> dst (column, j2, j1, 2): forward DFT per axis onto G (spec 10.2),
    three radix stages each (dft.mojo). The twist by g^i is inside the tables, so there is no separate
    pass. ltmp holds G2 rows of G1 per column: axis 1 writes its first half and scratches in the
    second, axis 2 reads the first half and scratches over both."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime G1 = 2 * h1
    var half = columns * h2 * G1 * 2
    dft_axis[p, True, 1](ctx, arena, coeff, ltmp, ltmp + half, 1, columns * h2, tab.base + tab.fwd1)
    dft_axis[p, True, 2](ctx, arena, ltmp, dst, ltmp, G1, columns, tab.base + tab.fwd2)


def residual[p: Params](ctx: DeviceContext, arena: Arena,
                        lde_buf: Int, families: Int, count: Int, tab: TableLayout, alpha: Int, chals: Int, dst: Int,
                        families_g: Int, count_g: Int, accs: Int, n_accs: Int) raises:
    """dst (j2, j1, e) = sum_entry kappa_entry X_entry(point): the fused pass of statement-layer 5,
    k_residual over `families_g` (the table without the Horner basis entries; may be `families` itself)
    plus the Horner transitions from `families` (all entries, kappa folded) and the `accs` descriptors."""
    comptime G1 = 2 * p.h1()
    comptime G2 = 2 * p.h2()
    ctx.enqueue_function[k_fold_alpha](arena.buf, Buf[1](families), Int32(count), Buf[16](alpha), Buf[16](chals),
                                       grid_dim=ceildiv(count, 64), block_dim=64)
    if families_g != families:
        ctx.enqueue_function[k_fold_alpha](arena.buf, Buf[1](families_g), Int32(count_g), Buf[16](alpha), Buf[16](chals),
                                           grid_dim=ceildiv(count_g, 64), block_dim=64)
    comptime kr = k_residual[p]
    ctx.enqueue_function[kr](arena.buf, Buf[2](lde_buf), Buf[1](families_g), Int32(count_g),
                             Buf[2](tab.base + tab.gate1), Buf[2](tab.base + tab.gate2),
                             Buf[1](families), Buf[1](accs), Int32(n_accs), Buf[16](dst),
                             grid_dim=ceildiv(G1 * G2, BACKEND.block), block_dim=BACKEND.block)


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
