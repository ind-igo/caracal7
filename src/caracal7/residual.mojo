"""Residual grid stages (design section 4, spec 8 and 10.2, statement-layer 5) on the GEMM skeleton.

Grid G = G1 x G2, G_l = <g_l> of order 2 h_l; point j is g_l^j, even j is H_l, odd j the coset.
Buffers (bytes; slowest ... fastest):
    lde       (column, j2, j1, 2)      F2 values of every witness column on G
    ltmp      (column, k2, j1, 2)      after the axis-1 forward DFT
    families  (entry, ENTRY)           the compiled family list, kappa folded in after alpha
    residual  (j2, j1, e)              R = sum_j alpha^j R_j on G
    quotient  five E-valued scratch tables, QUOTIENT_ELEMS x e bytes (see `quotient`)
    trace_q   (3 e columns, x2, x1)    A, B, Q2 coordinate columns as values on H: witness-shaped

Family entry (ENTRY = 48 bytes, every field u16 little-endian unless noted): kappa E [0, 16);
col_a, dj1_a, dj2_a [16, 22); col_b, dj1_b, dj2_b [22, 28), col_b = NONE for a linear entry;
mult u8 [28] (0 none, 1 the gate (X1 - e1), 2 the gate (X2 - e2)); coef F u8 [29]; family [30, 32);
chal u8 [32] (0 none, 1 beta, 2 delta, 3 gamma: a stage-1 challenge factor); basis, basis2 u8 [33, 35)
(t < e: the factor b_t, the unit vector t of E; NO_BASIS none). Shifts are offsets on G in [0, 2 h_l):
a read at (omega1^k x1, x2) is dj1 = 2k. kappa = coef * alpha^family * chal * b_t * b_t2, one entry
per (family, read).
The challenge and basis factors are how an E-valued accumulator enters as e F-valued coordinate
columns (statement-layer 2: a coefficient is a constant, a challenge expression, or a public read).

Opening points (POINT = 4 bytes, (dj1, dj2) u16): a shift of z by g_l^dj, or a fixed coordinate:
FIX_ONE is 1 and FIX_E is e_l = omega_l^(h_l - 1) (spec section 3: (1, z2), (e1, z2), (1, omega2 z2),
(1, 1), (e1, e2)). Fixed points are opening points only; a residual read is always a shift.
ponytail: collapsing shared reads into one kappa (statement-layer 5) is the compiler's job when a
real family list exists; the kernel does not care.

Every stage is a launch of backend.gemm_f2 ("shapes are GEMMs", design section 8):
    lde        axis 1: C[line, j1] = sum_k coeff[line, k] g1^(j1 k); axis 2 per column likewise
    residual   C[slot, point] = sum_entry kappa[entry][slot] * X[entry][point]; X is gathered by the
               Family loader as mult(point) * c_a(shift_a point) * c_b(shift_b point)
    quotient   q1m over G1 -> Q1 on the coset; qinv1, ginv2 -> the A, B coefficients;
               q2m, qinv2 -> the Q2 coefficients; forward DFTs to their values on H
"""

from std.math import ceildiv
from max.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim

from caracal7.field import F2, E, f_add, f_mul, f_sub, ext_mul, ext_pow, ext_embed
from caracal7.params import Params
from caracal7.tables import TableLayout
from caracal7.backend import BACKEND, LANE_TILE, Operands, Loader, Strided, launch_gemm_f2, strided
from caracal7.accumulate import ACC, ACC_W_MAX

comptime ENTRY = 48
comptime NONE = 65535
comptime NO_BASIS = 255
comptime FIX_ONE = 65534    # a point coordinate fixed at 1
comptime FIX_E = 65535      # a point coordinate fixed at e_l
comptime BLOCK = 256


def quotient_elems[p: Params]() -> Int:
    """E elements of quotient scratch: Q1 on the coset (h1 x G2), its axis-1 transform, the A, B
    coefficients (G2 x h1), the Q2 coefficients, the Q2 axis-1 transform, the axis-1 values of
    A, B, Q2, their values on H: 14 N."""
    return 14 * p.N()


comptime POINT = 4      # bytes per opening point: (dj1, dj2) as u16


def shift_points(fam: List[UInt8]) -> List[UInt8]:
    """The opening points as (dj1, dj2) pairs: the seven of spec section 3 in its order (z, the shifted
    point, (1, z2), (e1, z2), (1, omega2 z2), (1, 1), (e1, e2)), then every other distinct read shift of
    the family table in first-seen order."""
    var pts = List[UInt8]()
    for pt in [(0, 0), (2, 0), (FIX_ONE, 0), (FIX_E, 0), (FIX_ONE, 2), (FIX_ONE, FIX_ONE), (FIX_E, FIX_E)]:
        var n = len(pts)
        pts.extend(List[UInt8](length=POINT, fill=0))
        _u16(pts, n, pt[0])
        _u16(pts, n + 2, pt[1])
    for k in range(len(fam) // ENTRY):
        var en = entry(fam, k)
        for side in range(2):
            if side == 1 and en.col_b == NONE:
                continue
            var d1 = en.dj1_a if side == 0 else en.dj1_b
            var d2 = en.dj2_a if side == 0 else en.dj2_b
            if point_index(pts, d1, d2) < 0:
                var n = len(pts)
                pts.extend(List[UInt8](length=POINT, fill=0))
                _u16(pts, n, d1)
                _u16(pts, n + 2, d2)
    return pts^


def point_coord(z: E, dj: Int, g: F2, h: Int) -> E:
    """One coordinate of an opening point: z g^dj, or the fixed 1 / e_l (host side; the kernel reads the
    power table)."""
    if dj == FIX_ONE:
        return ext_embed[4](SIMD[DType.uint8, 1](1))
    if dj == FIX_E:
        return ext_embed[4](ext_pow[1](g, 2 * (h - 1)))
    return ext_mul[4](z, ext_embed[4](ext_pow[1](g, dj)))


def point_index(pts: List[UInt8], dj1: Int, dj2: Int) -> Int:
    for i in range(len(pts) // POINT):
        if _get16(pts, i * POINT) == dj1 and _get16(pts, i * POINT + 2) == dj2:
            return i
    return -1


# ---- host side of the family list ----

struct Families:
    """Builder for the entry table and the accumulator descriptors (accumulate.mojo). Shifts are
    given on H (k in omega^k) and stored doubled."""
    var bytes: List[UInt8]
    var count: Int
    var accs: List[UInt8]

    def __init__(out self):
        self.bytes = List[UInt8]()
        self.count = 0
        self.accs = List[UInt8]()

    def accumulator(mut self, family: Int, z_col: Int, num: List[Int], den: List[Int]) raises:
        """A permutation accumulator (spec 6.2) on witness columns: N = gamma + fp(num), D = gamma +
        fp(den). Its transition (X1 - e1) (Z(next) D - Z N) over the coordinate columns z_col + t is
        e (2 + |num| + |den|) entries: Z = sum_t Z_t b_t and fp(c) = sum_j c_j b_j."""
        if len(num) > ACC_W_MAX or len(den) > ACC_W_MAX or len(num) == 0 or len(den) == 0:
            raise Error("accumulator record width")
        for t in range(16):
            self.add(family, 1, z_col + t, k1_a=1, mult=1, chal=3, basis=t)
            for j in range(len(den)):
                self.add(family, 1, z_col + t, k1_a=1, col_b=den[j], mult=1, basis=t, basis2=j)
            self.add(family, 126, z_col + t, mult=1, chal=3, basis=t)
            for j in range(len(num)):
                self.add(family, 126, z_col + t, col_b=num[j], mult=1, basis=t, basis2=j)
        var a = List[UInt8](length=ACC, fill=0)
        _u16(a, 0, z_col)
        _u16(a, 2, len(num))
        _u16(a, 4, len(den))
        for j in range(len(num)):
            _u16(a, 6 + 2 * j, num[j])
        for j in range(len(den)):
            _u16(a, 22 + 2 * j, den[j])
        self.accs.extend(a^)

    def add(mut self, family: Int, coef: Int, col_a: Int, k1_a: Int = 0, k2_a: Int = 0,
            col_b: Int = -1, k1_b: Int = 0, k2_b: Int = 0, mult: Int = 0, chal: Int = 0, basis: Int = -1, basis2: Int = -1) raises:
        """Shifts k_l are on H, already reduced to [0, h_l). An entry with the axis-2 gate must be
        linear: two columns and (X2 - e2) exceed the bound 2 h2 - 2 of spec section 8 and would alias
        on G. The axis-1 gate admits a quadratic entry (2 h1 - 1): the accumulator transition."""
        if col_b >= 0 and mult == 2:
            raise Error("axis-2 gated entries must be linear (spec 8 degree bound)")
        var e = List[UInt8](length=ENTRY, fill=0)
        for v in [col_a, 2 * k1_a, 2 * k2_a, NONE if col_b < 0 else col_b, 2 * k1_b, 2 * k2_b]:
            if v < 0 or v > 65535:
                raise Error("family entry field out of range")
        _u16(e, 16, col_a)
        _u16(e, 18, 2 * k1_a)
        _u16(e, 20, 2 * k2_a)
        _u16(e, 22, NONE if col_b < 0 else col_b)
        _u16(e, 24, 2 * k1_b)
        _u16(e, 26, 2 * k2_b)
        e[28] = UInt8(mult)
        e[29] = UInt8(coef % 127)
        _u16(e, 30, family)
        e[32] = UInt8(chal)
        e[33] = UInt8(NO_BASIS if basis < 0 else basis)
        e[34] = UInt8(NO_BASIS if basis2 < 0 else basis2)
        self.bytes.extend(e^)
        self.count += 1


def _u16(mut l: List[UInt8], at: Int, v: Int):
    l[at] = UInt8(v & 255)
    l[at + 1] = UInt8(v >> 8)


def _get16(l: List[UInt8], at: Int) -> Int:
    return Int(l[at]) | Int(l[at + 1]) << 8


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
    var chal: Int
    var basis: Int
    var basis2: Int


def entry(fam: List[UInt8], k: Int) -> Entry:
    var o = k * ENTRY
    return Entry(col_a=_get16(fam, o + 16), dj1_a=_get16(fam, o + 18), dj2_a=_get16(fam, o + 20),
                 col_b=_get16(fam, o + 22), dj1_b=_get16(fam, o + 24), dj2_b=_get16(fam, o + 26),
                 mult=Int(fam[o + 28]), coef=Int(fam[o + 29]), family=_get16(fam, o + 30),
                 chal=Int(fam[o + 32]), basis=Int(fam[o + 33]), basis2=Int(fam[o + 34]))


def kappa_of(en: Entry, alpha: E, chals: List[UInt8]) -> E:
    """coef * alpha^family * chal * b_t; chals holds the stage-1 challenges (beta, delta, gamma) as e bytes each."""
    var kappa = f_mul(ext_pow[4](alpha, en.family), E(UInt8(en.coef)))
    if en.chal != 0:
        var c = E(0)
        for t in range(16):
            c[t] = chals[(en.chal - 1) * 16 + t]
        kappa = ext_mul[4](kappa, c)
    for t in [en.basis, en.basis2]:
        if t != NO_BASIS:
            var b = E(0)
            b[t] = 1
            kappa = ext_mul[4](kappa, b)
    return kappa


def residual_at(fam: List[UInt8], alpha: E, chals: List[UInt8], z1: E, z2: E, e1: F2, e2: F2, reads: List[E]) -> E:
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
        acc = f_add(acc, ext_mul[4](kappa_of(en, alpha, chals), v))
    return acc


def synthetic_families(columns_w: Int = 9, with_accumulator: Bool = True) raises -> Families:
    """Eight families over nine columns, satisfied by `synthetic_trace`, plus one permutation
    accumulator (c8 is a permutation of c0 across the grid) whose coordinate columns start the Z
    tree at global index columns_w. The families cover a linear entry, a quadratic entry, both gates,
    a within-chain shift, a cyclic shift, an axis-2 shift, a challenge and basis coefficient, a
    quadratic axis-1 transition, and the accumulator."""
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
    f.add(6, 1, 2, chal=3, basis=3)                  # gamma b_3 (c2 - c0 c1): a challenge-expression coefficient
    f.add(6, 126, 0, col_b=1, chal=3, basis=3)
    f.add(7, 1, 4, k1_a=1, col_b=5, mult=1)          # (X1 - e1) c5 (c4(next) - c0): quadratic with the axis-1 gate
    f.add(7, 126, 0, col_b=5, mult=1)
    if with_accumulator:
        f.accumulator(8, columns_w, [0], [8])              # (X1 - e1) (Z(next) (gamma + c8) - Z (gamma + c0))
    return f^


comptime SYNTHETIC_COLUMNS = 9
comptime SYNTHETIC_PERM = 17                            # c8[i] = c0[(17 i + 5) mod N]: coprime to every grid N


def synthetic_trace[p: Params](seed: Int) -> List[UInt8]:
    """Nine columns (column, x2, x1) satisfying `synthetic_families`."""
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
            t[8 * N + i] = t[(SYNTHETIC_PERM * i + 5) % N]                                             # c8 = c0 permuted
    return t^


# ---- kernels ----

@always_inline
def _gid() -> Int:
    return Int(block_idx.x * block_dim.x + thread_idx.x)


def k_fold_alpha(base: Pointer[UInt8, MutAnyOrigin], families: Int64, count: Int32, alpha: Int64, chals: Int64):
    """kappa = coef * alpha^family * chal * b_t, one thread per entry (see `kappa_of`)."""
    var gid = _gid()
    if gid >= Int(count):
        return
    var ent = Int(families) + gid * ENTRY
    var a = base.unsafe_load[width=16](Int(alpha))
    var fam = _d16(base, ent + 30)
    var kappa = f_mul(ext_pow[4](a, fam), E(base[unsafe_offset=ent + 29]))
    var chal = Int(base[unsafe_offset=ent + 32])
    if chal != 0:
        kappa = ext_mul[4](kappa, base.unsafe_load[width=16](Int(chals) + (chal - 1) * 16))
    for i in range(2):
        var basis = Int(base[unsafe_offset=ent + 33 + i])
        if basis != NO_BASIS:
            var b = E(0)
            b[basis] = 1
            kappa = ext_mul[4](kappa, b)
    base.unsafe_store[width=16](ent, kappa)


@always_inline
def _d16(base: Pointer[UInt8, MutAnyOrigin], at: Int) -> Int:
    return Int(base[unsafe_offset=at]) | Int(base[unsafe_offset=at + 1]) << 8


@always_inline
def _read[p: Params](base: Pointer[UInt8, MutAnyOrigin], lde_buf: Int, at: Int, j1: Int, j2: Int) -> F2:
    """c(shift point) for the read descriptor (col, dj1, dj2) at `at`; shifts are below the domain size."""
    comptime G1 = 2 * p.h1()
    comptime G2 = 2 * p.h2()
    var col = _d16(base, at)
    var a = j1 + _d16(base, at + 2)
    if a >= G1:
        a -= G1
    var b = j2 + _d16(base, at + 4)
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
        if _d16(base, ent + 22) != NONE:
            v = ext_mul[1](v, _read[Self.p](base, Int(o.aux0), ent + 22, n_lo, n_hi))
        var mult = base[unsafe_offset=ent + 28]
        if mult == 1:
            v = ext_mul[1](v, base.unsafe_load[width=2](Int(o.aux1) + n_lo * 2))
        elif mult == 2:
            v = ext_mul[1](v, base.unsafe_load[width=2](Int(o.aux2) + n_hi * 2))
        return v


def k_values_to_trace[p: Params](base: Pointer[UInt8, MutAnyOrigin], vals: Int64, trace: Int64, groups: Int32):
    """trace[q * e + tau, x] = coordinate tau of V_q(x) for x in H, q < groups, vals (groups, x, e):
    E-valued columns (A, B, Q2; the accumulators) are ordinary F-valued coordinate columns from here
    on (see docs/decisions.md)."""
    comptime N = p.N()
    comptime e = p.e
    var gid = _gid()
    if gid >= Int(groups) * e * N:
        return
    var c = gid // N
    var x = gid % N
    base[unsafe_offset=Int(trace) + gid] = base[unsafe_offset=Int(vals) + ((c // e) * N + x) * e + c % e]


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
                        lde_buf: Int, families: Int, count: Int, tab: TableLayout, alpha: Int, chals: Int, dst: Int) raises:
    """dst (j2, j1, e) = sum_entry kappa_entry X_entry(point): the fused pass of statement-layer 5 as
    one GEMM, A = the kappa table (8 F2 lanes x entries), B gathered from the LDE."""
    comptime G1 = 2 * p.h1()
    comptime G2 = 2 * p.h2()
    ctx.enqueue_function[k_fold_alpha](base, Int64(families), Int32(count), Int64(alpha), Int64(chals),
                                       grid_dim=ceildiv(count, 64), block_dim=64)
    var o = strided(a=families, sa_m=2, sa_k=ENTRY, b=families, sb_k=0, sb_hi=0, sb_lo=0,
                    c=dst, sc_m=2, sc_hi=G1 * p.e, sc_lo=p.e)
    o.aux0 = Int64(lde_buf)
    o.aux1 = Int64(tab.base + tab.gate1)
    o.aux2 = Int64(tab.base + tab.gate2)
    # ponytail: D = G1 is not a power of two, so the point split costs an integer division per gathered
    # element; a (j2, j1) 2D launch removes it when the residual shows up in the profile.
    launch_gemm_f2[BACKEND, LANE_TILE, Family[p], G1](ctx, base, o, p.e // 2, G1 * G2, count)


def quotient[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
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
    # 6, 7. values on H of A, B, Q2 (batch of three): the even rows of g_l^(j k) are omega_l^(x k)
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, base, strided(
        a=tab.base + tab.wfwd1, sa_m=2 * h1 * 2, sa_k=2, b=q1coef, sb_k=e, sb_hi=h1 * e, sb_lo=2,
        c=v1, sc_m=h2 * e, sc_hi=e, sc_lo=2, sb_z=N * e, sc_z=N * e), h1, h2 * 8, h1, batch=3)
    launch_gemm_f2[BACKEND, T, Strided, 8](ctx, base, strided(
        a=tab.base + tab.wfwd2, sa_m=2 * h2 * 2, sa_k=2, b=v1, sb_k=e, sb_hi=h2 * e, sb_lo=2,
        c=vals, sc_m=h1 * e, sc_hi=e, sc_lo=2, sb_z=N * e, sc_z=N * e), h2, h1 * 8, h2, batch=3)
    # 8. coordinate columns as the quotient tree's trace
    comptime k8 = k_values_to_trace[p]
    ctx.enqueue_function[k8](base, Int64(vals), Int64(trace_q), Int32(3), grid_dim=ceildiv(3 * e * N, BLOCK), block_dim=BLOCK)
