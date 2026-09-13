"""Residual grid stages (design section 4, spec 8 and 10.2, statement-layer 5) on the GEMM skeleton.

Grid G = G1 x G2, G_l = <g_l> of order 2 h_l; point j is g_l^j, even j is H_l, odd j the coset.
Buffers (bytes; slowest ... fastest):
    lde       (column, j2, j1, 2)      F2 values on G: witness, Z coordinate, then public columns (one index space)
    ltmp      (column, k2, j1, 2)      after the axis-1 forward DFT
    families  (entry, ENTRY)           the compiled family list, kappa folded in after alpha
    residual  (j2, j1, e)              R = sum_j alpha^j R_j on G
    quotient  five E-valued scratch tables, QUOTIENT_ELEMS x e bytes (see `quotient`)
    trace_q   (3 e columns, x2, x1)    A, B, Q2 coordinate columns as values on H: witness-shaped
The coset steps are launches of backend.gemm_f2 ("shapes are GEMMs", design section 8), the full-length
transforms radix stages:
    lde        dft_axis per axis (three radix stages, dft.mojo): coefficients k -> the points of G
    residual   the one stage that is not a GEMM launch: k_residual, one thread per column position and
               two rows, gathers X[entry][point] = mult(point) * c_a(shift_a point) * c_b(shift_b point)
               and accumulates the e / 2 kappa lanes per row in registers (the GEMM skeleton's shared-memory
               staging cost more than the gather at M = e / 2). The 2 e basis entries of a Horner transition
               are not in the table it walks: k_horner, one thread per point after it, reads the e
               coordinate columns as one E value R(point) and adds alpha^f gate (R(omega1 x) - scale R)
    quotient   q1m over G1 -> Q1 on the coset; qinv1 (GEMM), then dft_axis over G2 -> the A, B
               coefficients; q2m, the qinv2p plan -> the Q2 coefficients; dft_axis per axis to their values on H
               (the three dense axis-2 GEMMs at K = 2 h2 and h2 were 40 of the stage's 57 ms)"""

from std.math import ceildiv
from max.gpu.host import DeviceContext

from caracal7.core.field import F2, E, V2, f_add, f_mul, f_sub, ext_mul, ext_pow, fp_center, fp_reduce, fp_canonical, fp_mul_f2, fp_ext_mul, E_LEVEL, E_BYTES, VH, e_planes, e_merge, ef_planes, ef_merge, to_f32, E_DFT_V
from caracal7.core.params import Params
from caracal7.core.tables import TableLayout
from caracal7.core.backend import BACKEND, Strided, launch_gemm_f2, strided
from caracal7.relations.ir import ENTRY, ENT_A, ENT_B, ENT_MULT, ENT_COEF, ENT_FAMILY, ENT_CHAL, ENT_BASIS, NONE, NO_BASIS, ACC, KIND_HORNER, HORNER_TRANSITIONS, acc_kind
from caracal7.core.bytes import Base, Buf, u16, get_u16
from caracal7.core.arena import Arena
from caracal7.core.dft import dft_axis, DftPlan
from std.gpu import global_idx



def quotient_elems[p: Params]() -> Int:
    """E elements of quotient scratch: Q1 on the coset (h1 x G2), its axis-1 transform, the A, B
    coefficients (G2 x h1), the Q2 coefficients, the Q2 axis-1 transform, the axis-1 values of
    A, B, Q2, their values on H: 14 N."""
    return 14 * p.N()


# ---- kernels ----

def k_fold_alpha(base: Base, families: Buf[1], count: Int32, alpha: Buf[E_BYTES], chals: Buf[E_BYTES]):
    """kappa = coef * alpha^family * chal * b_t, one thread per entry (see `kappa_of`)."""
    var gid = Int(global_idx.x)
    if gid >= Int(count):
        return
    var ent = families.offset(gid * ENTRY)
    var a = alpha.load(base, 0)
    var fam = u16(base, ent.at(ENT_FAMILY))
    var kappa = f_mul(ext_pow[E_LEVEL](a, fam), E(ent.load(base, ENT_COEF)))
    var chal = Int(ent.load(base, ENT_CHAL))
    if chal != 0:
        kappa = ext_mul[E_LEVEL](kappa, chals.load(base, chal - 1))
    for i in range(2):
        var basis = Int(ent.load(base, ENT_BASIS + i))
        if basis != NO_BASIS:
            var b = E(0)
            b[basis] = 1
            kappa = ext_mul[E_LEVEL](kappa, b)
    Buf[E_BYTES](ent.at(0)).store(base, 0, kappa)


def k_merge_kappa(base: Base, families: Buf[1], merge: Buf[1], idx_off: Int32, count_g: Int32, families_g: Buf[1]):
    """kappa_g[u] = sum of the folded kappas of the entries `merge` lists for row u, one thread per row."""
    var u = Int(global_idx.x)
    if u >= Int(count_g):
        return
    var start = u16(base, merge.at(4 * u))
    var n = u16(base, merge.at(4 * u + 2))
    var kappa = E(0)
    for t in range(start, start + n):
        var i = u16(base, merge.at(Int(idx_off) + 2 * t))
        kappa = f_add(kappa, Buf[E_BYTES](families.at(i * ENTRY)).load(base, 0))
    Buf[E_BYTES](families_g.at(u * ENTRY)).store(base, 0, kappa)


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
    comptime for l in range(E_BYTES // 2):
        var a = lde.load(base, ((col + 2 * l) * G2 + j2) * G1 + j1)
        var b = lde.load(base, ((col + 2 * l + 1) * G2 + j2) * G1 + j1)
        r[2 * l] = f_sub(a[0], b[1])
        r[2 * l + 1] = f_add(a[1], b[0])
    return r


def k_residual[p: Params](base: Base, lde: Buf[2], fam: Buf[1], count: Int32, gate1: Buf[2], gate2: Buf[2], dst: Buf[E_BYTES]):
    """dst[point] = sum over entries of kappa X(point), one thread per column position and V rows
    (the rows 2 t apart share the entry descriptors and kappa, and have V loads in flight) with the e / 2
    kappa lanes per row accumulated in registers: X = mult(point) c_a(shift_a point) c_b(shift_b point)
    is gathered once and multiplies the E kappa lane by lane (F2 times F2 per lane on fp32 lanes, 128
    terms between reductions like gemm_f2). Threads walk the odd rows, then the odd columns of the even
    rows; R is zero on H x H for a satisfied statement, so the last quadrant's threads write zero. The
    Horner transitions are k_horner's, launched after this one."""
    comptime G1 = 2 * p.h1()
    comptime G2 = 2 * p.h2()
    comptime V = 2 if p.h2() % 2 == 0 else 1
    comptime Q1 = G1 * p.h2()
    comptime Q2 = p.h1() * p.h2()
    comptime R1 = Q1 // V
    comptime R2 = Q2 // V
    comptime assert BACKEND.max_terms & (BACKEND.max_terms - 1) == 0, "the reduction cadence masks k"
    var gid = Int(global_idx.x)
    if gid >= R1 + R2 + Q2:
        return
    var j1: Int
    var j2: Int
    if gid < R1:
        j2 = 2 * V * (gid // G1) + 1
        j1 = gid % G1
    elif gid < R1 + R2:
        var g = gid - R1
        j2 = 2 * V * (g // p.h1())
        j1 = 2 * (g % p.h1()) + 1
    else:
        var g = gid - R1 - R2
        j2 = 2 * (g // p.h1())
        j1 = 2 * (g % p.h1())
        dst.store(base, j2 * G1 + j1, E(0))
        return
    var g1f = fp_center(gate1.load(base, j1))
    var g2f = InlineArray[V2, V](fill=V2(0))
    var re = InlineArray[VH, V](fill=VH(0))
    var im = InlineArray[VH, V](fill=VH(0))
    comptime for t in range(V):
        g2f[t] = fp_center(gate2.load(base, j2 + 2 * t))
    # fp32 lanes: |c| <= 126, times a second read <= 31.7 K, times a centered gate <= 4 M, reduced to
    # |v| <= 190; a term kappa v is below 48 K, 128 of them below 6.2 M, exact in fp32
    for k in range(Int(count)):
        var ent = fam.at(k * ENTRY)
        var second = u16(base, ent + ENT_B) != NONE
        var mult = fam.load(base, k * ENTRY + ENT_MULT)
        var kap = e_planes(Buf[E_BYTES](ent).load(base, 0))
        var kre = to_f32(kap[0])
        var kim = to_f32(kap[1])
        comptime for t in range(V):
            var v = to_f32(_read[p](base, lde, ent + ENT_A, j1, j2 + 2 * t))
            if second:
                v = fp_mul_f2(v, to_f32(_read[p](base, lde, ent + ENT_B, j1, j2 + 2 * t)))
            if mult == 1:
                v = fp_mul_f2(v, g1f)
            elif mult == 2:
                v = fp_mul_f2(v, g2f[t])
            v = fp_reduce(v)
            var v0 = VH(v[0])
            var v1 = VH(v[1])
            re[t] = kim.fma(-v1, kre.fma(v0, re[t]))
            im[t] = kim.fma(v0, kre.fma(v1, im[t]))
        if (k & (BACKEND.max_terms - 1)) == BACKEND.max_terms - 1:
            comptime for t in range(V):
                re[t] = fp_reduce(re[t])
                im[t] = fp_reduce(im[t])
    comptime for t in range(V):
        dst.store(base, (j2 + 2 * t) * G1 + j1, e_merge(fp_canonical(re[t]), fp_canonical(im[t])))


def k_horner[p: Params](base: Base, lde: Buf[2], gate1: Buf[2], families: Buf[1], accs: Buf[1], n_accs: Int32, dst: Buf[E_BYTES]):
    """dst[point] += gate1(j1) (kappa_A R(omega1 x) + kappa_B R(x)) for every Horner accumulator, one
    thread per point of the odd rows and the odd columns of the even rows (the rest is zero), with
    kappa_A = alpha^f and kappa_B = -alpha^f scale, the folded kappas of its basis-0 entries
    (ir.Families.horner puts the 2 e basis entries just before the ingest range), R read from the e
    coordinate columns as one E. fp32 lanes: the two E products of canonical values unreduced (below
    2.1 M each), the sum reduced to |x| <= 190, times the centered gate on lane pairs (below 24 K),
    the running total and dst below 1 M, canonicalized at the store."""
    comptime G1 = 2 * p.h1()
    comptime G2 = 2 * p.h2()
    comptime Q1 = G1 * p.h2()
    comptime Q2 = p.h1() * p.h2()
    var gid = Int(global_idx.x)
    if gid >= Q1 + Q2:
        return
    var j1: Int
    var j2: Int
    if gid < Q1:
        j2 = 2 * (gid // G1) + 1
        j1 = gid % G1
    else:
        j2 = 2 * ((gid - Q1) // p.h1())
        j1 = 2 * ((gid - Q1) % p.h1()) + 1
    var g = fp_center(gate1.load(base, j1))
    var acc = to_f32(dst.load(base, j2 * G1 + j1))
    var jn = j1 + 2
    if jn >= G1:
        jn -= G1
    for k in range(Int(n_accs)):
        var d = accs.offset(k * ACC)
        if Int(d.load(base, 38)) != KIND_HORNER:
            continue
        var col = u16(base, d.at(0))
        var first = u16(base, d.at(2))
        var ka = to_f32(Buf[E_BYTES](families.at((first - HORNER_TRANSITIONS) * ENTRY)).load(base, 0))
        var kb = to_f32(Buf[E_BYTES](families.at((first - HORNER_TRANSITIONS + 1) * ENTRY)).load(base, 0))
        var v = fp_reduce(fp_ext_mul[E_LEVEL](ka, to_f32(_z_read[p](base, lde, col, jn, j2)))
                          + fp_ext_mul[E_LEVEL](kb, to_f32(_z_read[p](base, lde, col, j1, j2))))
        var h = ef_planes(v)                            # times the gate, an F2 scalar, lane by lane
        acc += ef_merge(h[0] * g[0] - h[1] * g[1], h[0] * g[1] + h[1] * g[0])
    dst.store(base, j2 * G1 + j1, fp_canonical(acc))


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

def _put_u16(mut l: List[UInt8], at: Int, v: Int):
    l[at] = UInt8(v & 255)
    l[at + 1] = UInt8(v >> 8)


def merge_tables(families: List[UInt8], accs: List[UInt8], entries: Int) raises -> Tuple[List[UInt8], List[UInt8], Int]:
    """The residual's entry table and its merge index from the family table: one `families_g` row per
    distinct (reads, gate) descriptor (bytes 16 to 28 of an entry), and `merge` as (entry, 2) u16 (start,
    count) then u16 indices of the entries each row sums. Every entry takes part except the HORNER_TRANSITIONS
    linear entries a Horner descriptor emits just before its ingest range (ir.Families.horner; Shape checks
    they are there): those belong to the Horner scan, not to the residual. The ingest entries are kept. Entries that share the reads share X(point), so sum kappa_i X = (sum kappa_i) X.
    Returns (families_g, merge, rows of families_g); families_g is padded to `entries` rows."""
    if entries > 65535:
        raise Error("the merge table indexes entries as u16")
    var keep = List[Bool](length=entries, fill=True)
    for k in range(len(accs) // ACC):
        if acc_kind(accs, k) == KIND_HORNER:
            var first = get_u16(accs, k * ACC + 2)
            for i in range(first - HORNER_TRANSITIONS, first):
                keep[i] = False
    var fg = List[UInt8](length=entries * ENTRY, fill=0)
    var mg = List[UInt8](length=entries * 6, fill=0)
    var row_of = Dict[String, Int]()
    var members = List[List[Int]]()
    var rows = 0
    for i in range(entries):
        if not keep[i]:
            continue
        var key = String("")
        for j in range(ENT_A, ENT_COEF):
            key += String(Int(families[i * ENTRY + j])) + ","
        var u = row_of.get(key, -1)
        if u < 0:
            u = rows
            row_of[key] = u
            members.append(List[Int]())
            for j in range(ENTRY):
                fg[u * ENTRY + j] = families[i * ENTRY + j]
            rows += 1
        members[u].append(i)
    var idx_off = entries * 4
    var at = 0
    for u in range(rows):
        _put_u16(mg, u * 4, at)
        _put_u16(mg, u * 4 + 2, len(members[u]))
        for i in members[u]:
            _put_u16(mg, idx_off + 2 * at, i)
            at += 1
    return (fg^, mg^, rows)


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
    dft_axis[DftPlan(G1, h1)](ctx, arena, coeff, ltmp, ltmp + half, 1, columns * h2, tab.base + tab.fwd1)
    dft_axis[DftPlan(2 * h2, h2), 4](ctx, arena, ltmp, dst, ltmp, G1, columns, tab.base + tab.fwd2)


def residual[p: Params](ctx: DeviceContext, arena: Arena,
                        lde_buf: Int, families: Int, count: Int, tab: TableLayout, alpha: Int, chals: Int, dst: Int,
                        families_g: Int, count_g: Int, accs: Int, n_accs: Int, merge: Int = -1) raises:
    """dst (j2, j1, e) = sum_entry kappa_entry X_entry(point): the fused pass of statement-layer 5,
    k_residual over `families_g` (one row per distinct descriptor when `merge` lists the entries each
    row sums, else the table without the Horner basis entries, or `families` itself) plus the Horner
    transitions from `families` (all entries, kappa folded) and the `accs` descriptors."""
    comptime G1 = 2 * p.h1()
    comptime G2 = 2 * p.h2()
    ctx.enqueue_function[k_fold_alpha](arena.buf, Buf[1](families), Int32(count), Buf[E_BYTES](alpha), Buf[E_BYTES](chals),
                                       grid_dim=ceildiv(count, 64), block_dim=64)
    if merge >= 0 and count_g > 0:
        ctx.enqueue_function[k_merge_kappa](arena.buf, Buf[1](families), Buf[1](merge), Int32(count * 4), Int32(count_g), Buf[1](families_g),
                                            grid_dim=ceildiv(count_g, 64), block_dim=64)
    elif families_g != families:
        ctx.enqueue_function[k_fold_alpha](arena.buf, Buf[1](families_g), Int32(count_g), Buf[E_BYTES](alpha), Buf[E_BYTES](chals),
                                           grid_dim=ceildiv(count_g, 64), block_dim=64)
    comptime V = 2 if p.h2() % 2 == 0 else 1
    comptime T = (G1 * p.h2()) // V + (p.h1() * p.h2()) // V + p.h1() * p.h2()
    comptime kr = k_residual[p]
    ctx.enqueue_function[kr](arena.buf, Buf[2](lde_buf), Buf[1](families_g), Int32(count_g),
                             Buf[2](tab.base + tab.gate1), Buf[2](tab.base + tab.gate2), Buf[E_BYTES](dst),
                             grid_dim=ceildiv(T, BACKEND.block), block_dim=BACKEND.block)
    if n_accs > 0:
        comptime kh = k_horner[p]
        ctx.enqueue_function[kh](arena.buf, Buf[2](lde_buf), Buf[2](tab.base + tab.gate1),
                                 Buf[1](families), Buf[1](accs), Int32(n_accs), Buf[E_BYTES](dst),
                                 grid_dim=ceildiv(G1 * p.h2() + p.h1() * p.h2(), BACKEND.block), block_dim=BACKEND.block)


def quotient[p: Params](ctx: DeviceContext, arena: Arena,
                        residual_buf: Int, tab: TableLayout, scratch: Int, trace_q: Int) raises:
    """residual -> A, B, Q2 coefficients -> their values on H -> 3 e coordinate columns as the
    trace of the quotient tree, which the level-1 encoder then treats like any witness column.
    E-valued operands are the D = e / 2 lane view of the skeleton."""
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
    var v1 = t2 + h1 * h2 * e               # (3, k2, x1, e)
    var vals = v1 + 3 * N * e               # (3, x2, x1, e)
    # 1. Q1 on the coset from R over all of G1: q1m (t, j1)
    launch_gemm_f2[BACKEND, T, Strided, E_BYTES // 2](ctx, arena, strided(
        a=tab.base + tab.q1m, sa_m=G1 * 2, sa_k=2, b=R, sb_k=e, sb_hi=G1 * e, sb_lo=2,
        c=q1c, sc_m=G2 * e, sc_hi=e, sc_lo=2), h1, G2 * (e // 2), G1)
    # ponytail: step 2 stays a dense h1 x h1 GEMM (2 KB, 32 wide); put it on a plan like step 5 if it shows up
    # 2. axis 1: coset values t -> coefficients k1
    launch_gemm_f2[BACKEND, T, Strided, E_BYTES // 2](ctx, arena, strided(
        a=tab.base + tab.qinv1, sa_m=h1 * 2, sa_k=2, b=q1c, sb_k=G2 * e, sb_hi=e, sb_lo=2,
        c=t1, sc_m=G2 * e, sc_hi=e, sc_lo=2), h1, G2 * (e // 2), h1)
    # 3. axis 2: G2 values j2 -> coefficients k2 in [0, 2 h2): A + X2^h2 B, the h1 lines of E rows as 8 F2
    #    lanes, written transposed as (k2, k1, e); t1 is the scratch (dead after stage 3)
    dft_axis[DftPlan(G2, G2), E_DFT_V](ctx, arena, t1, q1coef, t1, e // 2, h1, tab.base + tab.ginv2p, dst_line=e, dst_j=h1 * e)
    # 4. Q2 = S1 / (-2) on H1 x g2 H2: axis 1 from the even j1 of the odd j2 rows, q2m = 63 winv1
    launch_gemm_f2[BACKEND, T, Strided, E_BYTES // 2](ctx, arena, strided(
        a=tab.base + tab.q2m, sa_m=h1 * 2, sa_k=2, b=R + G1 * e, sb_k=2 * e, sb_hi=2 * G1 * e, sb_lo=2,
        c=t2, sc_m=h2 * e, sc_hi=e, sc_lo=2), h1, h2 * (e // 2), h1)
    # 5. axis 2: coset values t2 -> coefficients k2, the output twist g2^-k inside the plan's last stage;
    #    a row of t2 is e / 2 F2 lanes, written transposed as (k2, k1, e); t1 (dead) is the scratch
    dft_axis[DftPlan(h2, h2), E_DFT_V](ctx, arena, t2, q2coef, t1, e // 2, h1, tab.base + tab.qinv2p, dst_line=e, dst_j=h1 * e)
    # 6, 7. values on H of A, B, Q2: axis 1 over the 3 h2 coefficient rows -> v1 (3, k2, x1, e), then
    #    axis 2 with a row of x1 as 8 h1 lanes -> vals (3, x2, x1, e). Scratch: vals, then q1c and t1
    #    (dead); the coefficients stay intact for the tests
    dft_axis[DftPlan(h1, h1), E_DFT_V](ctx, arena, q1coef, v1, vals, e // 2, 3 * h2, tab.base + tab.hfwd1)
    dft_axis[DftPlan(h2, h2), E_DFT_V](ctx, arena, v1, vals, q1c, (e // 2) * h1, 3, tab.base + tab.hfwd2)
    # 8. coordinate columns as the quotient tree's trace
    comptime k8 = k_values_to_trace[p]
    ctx.enqueue_function[k8](arena.buf, Buf[1](vals), Buf[1](trace_q), Int32(3), grid_dim=ceildiv(3 * e * N, BACKEND.block), block_dim=BACKEND.block)
