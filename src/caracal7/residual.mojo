"""Residual grid stages (design section 4, spec 8 and 10.2, statement-layer 5) on the GEMM skeleton.

Grid G = G1 x G2, G_l = <g_l> of order 2 h_l; point j is g_l^j, even j is H_l, odd j the coset.
Buffers (bytes; slowest ... fastest):
    lde       (column, j2, j1, 2)      F2 values of every witness column on G
    ltmp      (column, k2, j1, 2)      after the axis-1 forward DFT
    families  (entry, ENTRY)           the compiled family list, kappa folded in after alpha
    residual  (j2, j1, e)              R = sum_j alpha^j R_j on G
    quotient  five E-valued scratch tables, QUOTIENT_ELEMS x e bytes (see `quotient`)
    stored    (3 e columns, slot)      A, B, Q2 coordinate columns in plain slots (spec 9.2)

Family entry (ENTRY = 32 bytes): kappa E [0, 16); col_a u16, dj1_a, dj2_a [16, 20); col_b u16,
dj1_b, dj2_b [20, 24), col_b = NONE for a linear entry; mult [24] (0 none, 1 the gate (X1 - e1),
2 the gate (X2 - e2)); coef F [25]; family u16 [26, 28). Shifts are offsets on G: a read at
(omega1^k x1, x2) is dj1 = 2k. kappa = coef * alpha^family, one entry per (family, read).
ponytail: collapsing shared reads into one kappa (statement-layer 5) is the compiler's job when a
real family list exists; the kernel does not care.

Every stage is a launch of backend.gemm_f2 ("shapes are GEMMs", design section 8):
    lde        axis 1: C[line, j1] = sum_k coeff[line, k] g1^(j1 k); axis 2 per column likewise
    residual   C[slot, point] = sum_entry kappa[entry][slot] * X[entry][point]; X is gathered by the
               Family loader as mult(point) * c_a(shift_a point) * c_b(shift_b point)
    quotient   q1m over G1 -> Q1 on the coset; qinv1, ginv2 -> the A, B coefficients;
               q2m, qinv2 -> the Q2 coefficients; then the length-m DFT on the odd digit into slots
"""

from std.math import ceildiv
from max.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim

from caracal7.field import F2, E, f_add, f_mul, f_sub, ext_mul, ext_pow, ext_embed
from caracal7.params import Params
from caracal7.tables import TableLayout
from caracal7.backend import BACKEND, Tile, Operands, Loader, Strided, launch_gemm_f2, strided

comptime ENTRY = 32
comptime NONE = 65535
comptime BLOCK = 256
comptime RES_TILE = Tile(BM=8, BN=128, BK=16, TM=8, TN=4)   # M = 8 F2 lanes of E; one SIMD group per block


def quotient_elems[p: Params]() -> Int:
    """E elements of quotient scratch: Q1 on the coset (h1 x G2), its axis-1 transform, the A, B
    coefficients (G2 x h1), the Q2 axis-1 transform (h1 x h2), the Q2 coefficients: 8 N."""
    return 8 * p.N()


# ---- host side of the family list ----

struct Families:
    """Builder for the entry table. Shifts are given on H (k in omega^k) and stored doubled."""
    var bytes: List[UInt8]
    var count: Int

    def __init__(out self):
        self.bytes = List[UInt8]()
        self.count = 0

    def add(mut self, family: Int, coef: Int, col_a: Int, k1_a: Int = 0, k2_a: Int = 0,
            col_b: Int = -1, k1_b: Int = 0, k2_b: Int = 0, mult: Int = 0):
        var e = List[UInt8](length=ENTRY, fill=0)
        e[16] = UInt8(col_a & 255)
        e[17] = UInt8(col_a >> 8)
        e[18] = UInt8(2 * k1_a)
        e[19] = UInt8(2 * k2_a)
        var cb = NONE if col_b < 0 else col_b
        e[20] = UInt8(cb & 255)
        e[21] = UInt8(cb >> 8)
        e[22] = UInt8(2 * k1_b)
        e[23] = UInt8(2 * k2_b)
        e[24] = UInt8(mult)
        e[25] = UInt8(coef % 127)
        e[26] = UInt8(family & 255)
        e[27] = UInt8(family >> 8)
        self.bytes.extend(e^)
        self.count += 1


@fieldwise_init
struct Entry(TrivialRegisterPassable):
    var col_a: Int
    var dj1_a: Int
    var dj2_a: Int
    var col_b: Int      # NONE for a linear entry
    var dj1_b: Int
    var dj2_b: Int
    var mult: Int
    var coef: Int
    var family: Int


def entry(fam: List[UInt8], k: Int) -> Entry:
    var o = k * ENTRY
    return Entry(col_a=Int(fam[o + 16]) | Int(fam[o + 17]) << 8, dj1_a=Int(fam[o + 18]), dj2_a=Int(fam[o + 19]),
                 col_b=Int(fam[o + 20]) | Int(fam[o + 21]) << 8, dj1_b=Int(fam[o + 22]), dj2_b=Int(fam[o + 23]),
                 mult=Int(fam[o + 24]), coef=Int(fam[o + 25]), family=Int(fam[o + 26]) | Int(fam[o + 27]) << 8)


def residual_at(fam: List[UInt8], alpha: E, z1: E, z2: E, e1: F2, e2: F2, reads: List[E]) -> E:
    """R(z) from opened values: reads[2k], reads[2k + 1] are c_a and c_b of entry k at their shifted
    points. The verifier's step 5 and the tests share this."""
    var acc = E(0)
    var g1 = f_sub(z1, ext_embed[4](e1))
    var g2 = f_sub(z2, ext_embed[4](e2))
    for k in range(len(fam) // ENTRY):
        var en = entry(fam, k)
        var v = reads[2 * k]
        if en.col_b != NONE:
            v = ext_mul[4](v, reads[2 * k + 1])
        if en.mult == 1:
            v = ext_mul[4](v, g1)
        elif en.mult == 2:
            v = ext_mul[4](v, g2)
        var kappa = f_mul(ext_pow[4](alpha, en.family), E(UInt8(en.coef)))
        acc = f_add(acc, ext_mul[4](kappa, v))
    return acc


def synthetic_families() -> Families:
    """Milestone 1: six families over eight columns, satisfied by `synthetic_trace`. They cover a
    linear entry, a quadratic entry, both gates, a within-chain shift, a cyclic shift, and an
    axis-2 shift."""
    var f = Families()
    f.add(0, 1, 2)                                   # c2 - c0 c1
    f.add(0, 126, 0, col_b=1)
    f.add(1, 1, 3)                                   # c3 - c0 - c1
    f.add(1, 126, 0)
    f.add(1, 126, 1)
    f.add(2, 1, 4, k1_a=1, mult=1)                   # (X1 - e1) (c4(next) - c0)
    f.add(2, 126, 0, mult=1)
    f.add(3, 1, 5, col_b=5)                          # c5^2 - c5
    f.add(3, 126, 5)
    f.add(4, 1, 6)                                   # c6 - c0(omega1^3 x1): cyclic within the chain
    f.add(4, 126, 0, k1_a=3)
    f.add(5, 1, 7, k2_a=1, mult=2)                   # (X2 - e2) (c7(x1, omega2 x2) - c0)
    f.add(5, 126, 0, mult=2)
    return f^


comptime SYNTHETIC_COLUMNS = 8


def synthetic_trace[p: Params](seed: Int) -> List[UInt8]:
    """Eight columns (column, x2, x1) satisfying `synthetic_families`."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime N = p.N()
    var t = List[UInt8](length=SYNTHETIC_COLUMNS * N, fill=0)
    var s = seed
    for i in range(N):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        t[i] = UInt8((s >> 8) % 127)
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        t[N + i] = UInt8((s >> 8) % 127)
        t[5 * N + i] = UInt8((s >> 20) & 1)
    for x2 in range(h2):
        for x1 in range(h1):
            var i = x2 * h1 + x1
            var c0 = SIMD[DType.uint8, 1](t[i])
            var c1 = SIMD[DType.uint8, 1](t[N + i])
            t[2 * N + i] = f_mul(c0, c1)[0]
            t[3 * N + i] = f_add(c0, c1)[0]
            t[4 * N + i] = t[x2 * h1 + (x1 + h1 - 1) % h1] if x1 > 0 else UInt8((i * 7) % 127)     # c4(omega1 x1) = c0(x1)
            t[6 * N + i] = t[x2 * h1 + (x1 + 3) % h1]                                              # c6 = c0(omega1^3 x1)
            t[7 * N + i] = t[(x2 - 1) * h1 + x1] if x2 > 0 else UInt8((i * 11) % 127)              # c7(omega2 x2) = c0(x2)
    return t^


# ---- kernels ----

@always_inline
def _gid() -> Int:
    return Int(block_idx.x * block_dim.x + thread_idx.x)


def k_fold_alpha(base: Pointer[UInt8, MutAnyOrigin], families: Int64, count: Int32, alpha: Int64):
    """kappa = coef * alpha^family, one thread per entry."""
    var gid = _gid()
    if gid >= Int(count):
        return
    var ent = Int(families) + gid * ENTRY
    var a = base.unsafe_load[width=16](Int(alpha))
    var fam = Int(base[unsafe_offset=ent + 26]) | Int(base[unsafe_offset=ent + 27]) << 8
    var kappa = f_mul(ext_pow[4](a, fam), E(base[unsafe_offset=ent + 25]))
    base.unsafe_store[width=16](ent, kappa)


@always_inline
def _read[p: Params](base: Pointer[UInt8, MutAnyOrigin], lde_buf: Int, at: Int, j1: Int, j2: Int) -> F2:
    """c(shift point) for the read descriptor (col u16, dj1, dj2) at `at`."""
    comptime G1 = 2 * p.h1()
    comptime G2 = 2 * p.h2()
    var col = Int(base[unsafe_offset=at]) | Int(base[unsafe_offset=at + 1]) << 8
    var a = j1 + Int(base[unsafe_offset=at + 2])
    if a >= G1:
        a -= G1
    var b = j2 + Int(base[unsafe_offset=at + 3])
    if b >= G2:
        b -= G2
    return base.unsafe_load[width=2](lde_buf + ((col * G2 + b) * G1 + a) * 2)


struct Family[p: Params](Loader):
    """B[entry, point] of the residual GEMM, gathered from the LDE: D = G1, so n_hi = j2, n_lo = j1.
    aux0 = lde, aux1 = gate1, aux2 = gate2."""

    @staticmethod
    def load(base: Pointer[UInt8, MutAnyOrigin], o: Operands, k: Int, n_hi: Int, n_lo: Int, z: Int) -> F2:
        var ent = Int(o.b) + k * ENTRY
        var v = _read[Self.p](base, Int(o.aux0), ent + 16, n_lo, n_hi)
        var col_b = Int(base[unsafe_offset=ent + 20]) | Int(base[unsafe_offset=ent + 21]) << 8
        if col_b != NONE:
            v = ext_mul[1](v, _read[Self.p](base, Int(o.aux0), ent + 20, n_lo, n_hi))
        var mult = base[unsafe_offset=ent + 24]
        if mult == 1:
            v = ext_mul[1](v, base.unsafe_load[width=2](Int(o.aux1) + n_lo * 2))
        elif mult == 2:
            v = ext_mul[1](v, base.unsafe_load[width=2](Int(o.aux2) + n_hi * 2))
        return v


def k_to_stored_e[p: Params](base: Pointer[UInt8, MutAnyOrigin], q1coef: Int64, q2coef: Int64, stored: Int64,
                             rho1: Int64, rho2: Int64):
    """stored[q * e + tau, x1 + 2^a1 (x2 + 2^a2 r)] = coordinate tau of c_x(r) for Q in (A, B, Q2):
    c_x(r) = sum_y coef[x2 + 2^a2 y2, x1 + 2^a1 y1] rho1^(y1 r1) rho2^(y2 r2), the plain slots of 9.2."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime N = p.N()
    comptime e = p.e
    var gid = _gid()
    if gid >= 3 * e * N:
        return
    var c = gid // N
    var slot = gid % N
    var q = c // e
    var tau = c % e
    var src = Int(q1coef) if q == 0 else (Int(q1coef) + h2 * h1 * e if q == 1 else Int(q2coef))
    var x1 = slot % (1 << p.a1)
    var x2 = (slot >> p.a1) % (1 << p.a2)
    var r = slot >> (p.a1 + p.a2)
    var r1 = r % p.m1
    var r2 = r // p.m1
    var acc = SIMD[DType.uint8, 1](0)
    for y2 in range(p.m2):
        var s2 = base[unsafe_offset=Int(rho2) + (y2 * r2) % p.m2]
        for y1 in range(p.m1):
            var s1 = base[unsafe_offset=Int(rho1) + (y1 * r1) % p.m1]
            var k1 = x1 + (1 << p.a1) * y1
            var k2 = x2 + (1 << p.a2) * y2
            var v = base[unsafe_offset=src + (k2 * h1 + k1) * e + tau]
            acc = f_add(acc, f_mul(SIMD[DType.uint8, 1](v), f_mul(SIMD[DType.uint8, 1](s1), SIMD[DType.uint8, 1](s2))))
    base[unsafe_offset=Int(stored) + gid] = acc[0]


# ---- host orchestration ----

def lde[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
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
    launch_gemm_f2[BACKEND, BACKEND.tile, Strided, 1](ctx, base, o1, columns * h2, G1, h1)
    # axis 2: per column, C[j2, j1] = sum_k2 g2^(j2 k2) ltmp[k2, j1]
    var o2 = strided(a=tab.base + tab.wfwd2, sa_m=h2 * 2, sa_k=2, b=ltmp, sb_k=G1 * 2, sb_hi=2, sb_lo=0,
                     c=dst, sc_m=G1 * 2, sc_hi=2, sc_lo=0, sb_z=h2 * G1 * 2, sc_z=G2 * G1 * 2)
    launch_gemm_f2[BACKEND, BACKEND.tile, Strided, 1](ctx, base, o2, G2, G1, h2, batch=columns)


def residual[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                        lde_buf: Int, families: Int, count: Int, tab: TableLayout, alpha: Int, dst: Int) raises:
    """dst (j2, j1, e) = sum_entry kappa_entry X_entry(point): the fused pass of statement-layer 5 as
    one GEMM, A = the kappa table (8 F2 lanes x entries), B gathered from the LDE."""
    comptime G1 = 2 * p.h1()
    comptime G2 = 2 * p.h2()
    ctx.enqueue_function[k_fold_alpha](base, Int64(families), Int32(count), Int64(alpha),
                                       grid_dim=ceildiv(count, 64), block_dim=64)
    var o = strided(a=families, sa_m=2, sa_k=ENTRY, b=families, sb_k=0, sb_hi=0, sb_lo=0,
                    c=dst, sc_m=2, sc_hi=G1 * p.e, sc_lo=p.e)
    o.aux0 = Int64(lde_buf)
    o.aux1 = Int64(tab.base + tab.gate1)
    o.aux2 = Int64(tab.base + tab.gate2)
    # ponytail: D = G1 is not a power of two, so the point split costs an integer division per gathered
    # element; a (j2, j1) 2D launch removes it when the residual shows up in the profile.
    launch_gemm_f2[BACKEND, RES_TILE, Family[p], G1](ctx, base, o, p.e // 2, G1 * G2, count)


def quotient[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                        residual_buf: Int, tab: TableLayout, scratch: Int, stored_q: Int) raises:
    """residual -> A, B, Q2 coefficients -> 3 e coordinate columns in plain slots, written straight
    into the quotient tree's `stored`. E-valued operands are the D = 8 lane view of the skeleton."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime G1 = 2 * h1
    comptime G2 = 2 * h2
    comptime e = p.e
    comptime T = BACKEND.tile
    var R = residual_buf
    var q1c = scratch                       # (t, j2, e)   Q1 on g1 H1 x G2
    var t1 = q1c + h1 * G2 * e              # (k1, j2, e)
    var q1coef = t1 + h1 * G2 * e           # (k2, k1, e)  rows k2 < h2: A; rows k2 >= h2: B
    var t2 = q1coef + G2 * h1 * e           # (k1, t2, e)
    var q2coef = t2 + h1 * h2 * e           # (k2, k1, e)
    # 1. Q1 on the coset from R over all of G1: q1m (t, j1)
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, base, strided(
        a=tab.base + tab.q1m, sa_m=G1 * 2, sa_k=2, b=R, sb_k=e, sb_hi=G1 * e, sb_lo=2,
        c=q1c, sc_m=G2 * e, sc_hi=e, sc_lo=2), h1, G2 * 8, G1)
    # 2. axis 1: coset values t -> coefficients k1
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, base, strided(
        a=tab.base + tab.qinv1, sa_m=h1 * 2, sa_k=2, b=q1c, sb_k=G2 * e, sb_hi=e, sb_lo=2,
        c=t1, sc_m=G2 * e, sc_hi=e, sc_lo=2), h1, G2 * 8, h1)
    # 3. axis 2: G2 values j2 -> coefficients k2 in [0, 2 h2): A + X2^h2 B
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, base, strided(
        a=tab.base + tab.ginv2, sa_m=G2 * 2, sa_k=2, b=t1, sb_k=e, sb_hi=G2 * e, sb_lo=2,
        c=q1coef, sc_m=h1 * e, sc_hi=e, sc_lo=2), G2, h1 * 8, G2)
    # 4. Q2 = S1 / (-2) on H1 x g2 H2: axis 1 from the even j1 of the odd j2 rows, q2m = 63 winv1
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, base, strided(
        a=tab.base + tab.q2m, sa_m=h1 * 2, sa_k=2, b=R + G1 * e, sb_k=2 * e, sb_hi=2 * G1 * e, sb_lo=2,
        c=t2, sc_m=h2 * e, sc_hi=e, sc_lo=2), h1, h2 * 8, h1)
    # 5. axis 2: coset values t2 -> coefficients k2
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, base, strided(
        a=tab.base + tab.qinv2, sa_m=h2 * 2, sa_k=2, b=t2, sb_k=e, sb_hi=h2 * e, sb_lo=2,
        c=q2coef, sc_m=h1 * e, sc_hi=e, sc_lo=2), h2, h1 * 8, h2)
    # 6. the odd-digit DFT into plain slots, 3 e columns
    comptime k6 = k_to_stored_e[p]
    ctx.enqueue_function[k6](base, Int64(q1coef), Int64(q2coef), Int64(stored_q),
                             Int64(tab.base + tab.rho1), Int64(tab.base + tab.rho2),
                             grid_dim=ceildiv(3 * e * p.N(), BLOCK), block_dim=BLOCK)
