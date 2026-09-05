"""The Caracal IR (spec 8, statement-layer 2 and 5): the compiled family list, the opening points,
and the host evaluation of a family at a point. What the frontend (milestone 4) emits and the
verifier reads; the kernels in residual.mojo consume the same bytes.


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
"""

from caracal7.field import F2, E, f_add, f_mul, f_sub, ext_mul, ext_pow, ext_embed, ext_one
from caracal7.accumulate import ACC, ACC_W_MAX
from caracal7.bytes import get_u16, set_u16

comptime ENTRY = 48
comptime NONE = 65535
comptime NO_BASIS = 255
comptime FIX_ONE = 65534    # a point coordinate fixed at 1
comptime FIX_E = 65535      # a point coordinate fixed at e_l


comptime POINT = 4      # bytes per opening point: (dj1, dj2) as u16


def shift_points(fam: List[UInt8]) -> List[UInt8]:
    """The opening points as (dj1, dj2) pairs: the seven of spec section 3 in its order (z, the shifted
    point, (1, z2), (e1, z2), (1, omega2 z2), (1, 1), (e1, e2)), then every other distinct read shift of
    the family table in first-seen order."""
    var pts = List[UInt8]()
    for pt in [(0, 0), (2, 0), (FIX_ONE, 0), (FIX_E, 0), (FIX_ONE, 2), (FIX_ONE, FIX_ONE), (FIX_E, FIX_E)]:
        var n = len(pts)
        pts.extend(List[UInt8](length=POINT, fill=0))
        set_u16(pts, n, pt[0])
        set_u16(pts, n + 2, pt[1])
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
                set_u16(pts, n, d1)
                set_u16(pts, n + 2, d2)
    return pts^


def point_coord(z: E, dj: Int, g: F2, h: Int) -> E:
    """One coordinate of an opening point: z g^dj, or the fixed 1 / e_l (host side; the kernel reads the
    power table)."""
    if dj == FIX_ONE:
        return ext_one[4]()
    if dj == FIX_E:
        return ext_embed[4](ext_pow[1](g, 2 * (h - 1)))
    return ext_mul[4](z, ext_embed[4](ext_pow[1](g, dj)))


def point_index(pts: List[UInt8], dj1: Int, dj2: Int) -> Int:
    for i in range(len(pts) // POINT):
        if get_u16(pts, i * POINT) == dj1 and get_u16(pts, i * POINT + 2) == dj2:
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
        set_u16(a, 0, z_col)
        set_u16(a, 2, len(num))
        set_u16(a, 4, len(den))
        for j in range(len(num)):
            set_u16(a, 6 + 2 * j, num[j])
        for j in range(len(den)):
            set_u16(a, 22 + 2 * j, den[j])
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
        set_u16(e, 16, col_a)
        set_u16(e, 18, 2 * k1_a)
        set_u16(e, 20, 2 * k2_a)
        set_u16(e, 22, NONE if col_b < 0 else col_b)
        set_u16(e, 24, 2 * k1_b)
        set_u16(e, 26, 2 * k2_b)
        e[28] = UInt8(mult)
        e[29] = UInt8(coef % 127)
        set_u16(e, 30, family)
        e[32] = UInt8(chal)
        e[33] = UInt8(NO_BASIS if basis < 0 else basis)
        e[34] = UInt8(NO_BASIS if basis2 < 0 else basis2)
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
    var chal: Int
    var basis: Int
    var basis2: Int


def entry(fam: List[UInt8], k: Int) -> Entry:
    var o = k * ENTRY
    return Entry(col_a=get_u16(fam, o + 16), dj1_a=get_u16(fam, o + 18), dj2_a=get_u16(fam, o + 20),
                 col_b=get_u16(fam, o + 22), dj1_b=get_u16(fam, o + 24), dj2_b=get_u16(fam, o + 26),
                 mult=Int(fam[o + 28]), coef=Int(fam[o + 29]), family=get_u16(fam, o + 30),
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
