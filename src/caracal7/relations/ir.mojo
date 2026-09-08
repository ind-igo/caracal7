"""The Caracal IR (spec 8, statement-layer 2 and 5): the compiled family list, the opening points,
and the host evaluation of a family at a point. What the frontend (milestone 4) emits and the
verifier reads; the kernels in residual.mojo consume the same bytes.


Family entry (ENTRY = 48 bytes, every field u16 little-endian unless noted): kappa E [0, 16);
col_a, dj1_a, dj2_a [16, 22); col_b, dj1_b, dj2_b [22, 28), col_b = NONE for a linear entry;
mult u8 [28] (0 none, 1 the gate (X1 - e1), 2 the gate (X2 - e2)); coef F u8 [29]; family [30, 32);
chal u8 [32] (0 none, else stage-1 element chal - 1: beta, delta, gamma sampled, then one per row of the
derivation table, see standard_chals); basis, basis2 u8 [33, 35)
(t < e: the factor b_t, the unit vector t of E; NO_BASIS none). Shifts are offsets on G in [0, 2 h_l):
a read at (omega1^k x1, x2) is dj1 = 2k. kappa = coef * alpha^family * chal * b_t * b_t2, one entry
per (family, read).
The challenge and basis factors are how an E-valued accumulator enters as e F-valued coordinate
columns (statement-layer 2: a coefficient is a constant, a challenge expression, or a public read).

Opening points (POINT = 4 bytes, (dj1, dj2) u16): a shift of z by g_l^dj, or a fixed coordinate:
FIX_ONE is 1 and FIX_E is e_l = omega_l^(h_l - 1). The list is part of the artifact (Shape.point_list);
shift_points is the default list and required_points the set every list must contain. Fixed points are
opening points only; a residual read is always a shift.

Challenge derivation table (CHAL = 3 bytes per row: op u8, a u8, b u8): element SAMPLED + i is
op(element a, element b) with op CHAL_ADD or CHAL_MUL and CHAL_ONE the constant 1 as an operand.
Part of the artifact (Shape.chals); k_derive_chals and derived_chals walk the same rows.

Accumulator descriptors (ACC bytes) are documented in accumulate.mojo; chain-end terms (END bytes,
Shape.ends) in smallgrid.mojo. Both carry the family index whose alpha power weights them.
ponytail: collapsing shared reads into one kappa (statement-layer 5) is the compiler's job when a
real family list exists; the kernel does not care.
"""

from caracal7.core.field import F2, E, f_add, f_mul, f_sub, f_pow, ext_mul, ext_pow, ext_embed, ext_one, ext_inv
from caracal7.core.bytes import get_u16, set_u16, list_e

comptime ENTRY = 48
comptime SAMPLED = 3    # stage-1 elements squeezed from the transcript: beta, delta, gamma
comptime CHAL = 3       # derivation table row: op, a, b
comptime CHAL_ADD = 0
comptime CHAL_MUL = 1
comptime CHAL_ONE = 255 # operand: the constant 1
comptime ACC = 42       # accumulator descriptor bytes (accumulate.mojo); family u16 at [40, 42) for every kind
comptime ACC_W_MAX = 8
comptime KIND_PERM = 0
comptime KIND_LOOKUP = 1
comptime KIND_HORNER = 2    # the spec's {start, ingest, scale, end} record: R(next) = scale R + sum weight read (polynomial-mulmod 5)
# TODO(memory): KIND_MEMORY = 3 when spec 6.4 lands.
comptime END = 10       # chain-end term (smallgrid.mojo): col_a, col_b, family u16; coef, chal, gate u8; pad. A line is a Z block at (e1, X2).
comptime NONE = 65535
comptime NO_BASIS = 255
comptime FIX_ONE = 65534    # a point coordinate fixed at 1
comptime FIX_E = 65535      # a point coordinate fixed at e_l


comptime POINT = 4      # bytes per opening point: (dj1, dj2) as u16
comptime PUB = 2        # public column spec (docs/public-columns.md): m u16; the column is a polynomial in (X1, X2^m), its public data the (h2 / m, h1) values of one period, x2 major
comptime RES = 6        # restriction: column u16, axis-2 coordinate u16 (FIX_ONE or FIX_E), coefficient count u16 (the line has degree < count); opened at (z1, coordinate)


def standard_chals() -> List[UInt8]:
    """The two derived elements the accumulator kernels read by index: 3 = 1 + beta, 4 = (1 + beta) delta."""
    return [CHAL_ADD, 0, CHAL_ONE, CHAL_MUL, 3, 1]


def chal_count(table: Span[UInt8, _]) -> Int:
    return SAMPLED + len(table) // CHAL


def required_points(fam: Span[UInt8, _], restrictions: List[UInt8], accumulators: Bool) -> List[UInt8]:
    """The points every opening list must contain: z, then with accumulators the boundary points the
    verifier reads ((1, z2), (e1, z2), (1, omega2 z2), (e1, e2); (1, 1) of spec section 3 is implied by
    Z(1, z2) = 1 at random z2), then the restriction lines (z1, coordinate), then every distinct read
    shift of the family table in first-seen order."""
    var pts = List[UInt8]()
    var fixed: List[Tuple[Int, Int]] = [(0, 0)]
    if accumulators:
        fixed.extend([(FIX_ONE, 0), (FIX_E, 0), (FIX_ONE, 2), (FIX_E, FIX_E)])
    for pt in fixed:
        var n = len(pts)
        pts.extend(List[UInt8](length=POINT, fill=0))
        set_u16(pts, n, pt[0])
        set_u16(pts, n + 2, pt[1])
    for i in range(len(restrictions) // RES):
        var coord = get_u16(restrictions, i * RES + 2)
        if point_index(pts, 0, coord) < 0:
            var n = len(pts)
            pts.extend(List[UInt8](length=POINT, fill=0))
            set_u16(pts, n, 0)
            set_u16(pts, n + 2, coord)
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


def shift_points(fam: Span[UInt8, _], restrictions: List[UInt8] = List[UInt8](), accumulators: Bool = True) -> List[UInt8]:
    """The default opening list: exactly the required points, nothing gated off."""
    return required_points(fam, restrictions, accumulators)


def point_coord(z: E, dj: Int, g: F2, h: Int) -> E:
    """One coordinate of an opening point: z g^dj, or the fixed 1 / e_l (host side; the kernel reads the
    power table)."""
    if dj == FIX_ONE:
        return ext_one[4]()
    if dj == FIX_E:
        return ext_embed[4](ext_pow[1](g, 2 * (h - 1)))
    return ext_mul[4](z, ext_embed[4](ext_pow[1](g, dj)))


def point_index(pts: Span[UInt8, _], dj1: Int, dj2: Int) -> Int:
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
    var ends: List[UInt8]

    def __init__(out self):
        self.bytes = List[UInt8]()
        self.count = 0
        self.accs = List[UInt8]()
        self.ends = List[UInt8]()

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
        self._descriptor(family, z_col, num, den, KIND_PERM, 0)

    def lookup(mut self, family: Int, z_col: Int, f: List[Int], s: List[Int], table: Int) raises:
        """A lookup accumulator (spec 6.3) of records f against table `table` (Shape.tables), s the sorted
        copy the prover fills (sort.mojo). N = (1 + beta) (delta + fp(f)), D = (1 + beta) delta + fp(s) +
        beta fp(s)(omega1 x1): the transition (X1 - e1) (Z(next) D - Z N) is e (2 + 3 w) entries."""
        if len(f) != len(s) or len(f) == 0 or len(f) > ACC_W_MAX or table < 0 or table > 255:
            raise Error("lookup record width or table id")
        for t in range(16):
            self.add(family, 1, z_col + t, k1_a=1, mult=1, chal=5, basis=t)
            for j in range(len(s)):
                self.add(family, 1, z_col + t, k1_a=1, col_b=s[j], mult=1, basis=t, basis2=j)
                self.add(family, 1, z_col + t, k1_a=1, col_b=s[j], k1_b=1, mult=1, chal=1, basis=t, basis2=j)
            self.add(family, 126, z_col + t, mult=1, chal=5, basis=t)
            for j in range(len(f)):
                self.add(family, 126, z_col + t, col_b=f[j], mult=1, chal=4, basis=t, basis2=j)
        self._descriptor(family, z_col, f, s, KIND_LOOKUP, table)

    def horner(mut self, family: Int, z_col: Int, start: Int, scale: Int, ingest: List[Tuple[Int, Int, Int, Int]]) raises:
        """The second Z kind (polynomial-mulmod 5): R(1, X2) = start, R(omega1 x1, x2) = scale R(x1, x2) + sum_i
        coef_i chal_i c_i(omega1^k1_i x1, x2) with `scale` a stage-1 element index (-1: 1) and `ingest` the
        terms (column, k1, coef, chal index or -1). The transition (X1 - e1) (R(next) - scale R - sum) is
        2 e + |ingest| linear entries; the descriptor names the ingest entries by range so the kernel and the
        residual read one definition."""
        if start < 0 or start > 1 or scale < -1 or scale > 254 or len(ingest) == 0 or len(ingest) > 65535:
            raise Error("horner accumulator: start in {0, 1}, scale an element index, at least one ingest term")
        for t in range(16):
            self.add(family, 1, z_col + t, k1_a=1, mult=1, basis=t)
            self.add(family, 126, z_col + t, mult=1, chal=scale + 1, basis=t)
        var first = self.count
        for it in ingest:
            self.add(family, ((-it[2] % 127) + 127) % 127, it[0], k1_a=it[1], mult=1, chal=it[3] + 1)
        var a = List[UInt8](length=ACC, fill=0)
        set_u16(a, 0, z_col)
        set_u16(a, 2, first)
        set_u16(a, 4, len(ingest))
        a[6] = UInt8(start)
        a[7] = UInt8(scale + 1)
        a[38] = UInt8(KIND_HORNER)
        set_u16(a, 40, family)
        self.accs.extend(a^)

    def chain_end(mut self, family: Int, coef: Int, col_a: Int, col_b: Int, chal: Int, gate: Bool) raises:
        """One term of a chain-end family (spec 7.4, statement-layer 3): coef chal A(e1, X2) [B(e1, X2)] on H2,
        optionally gated by (X2 - e2); A, B are Z blocks by their first coordinate column. Summed into R2 with
        the family's alpha power."""
        if chal < -1 or chal > 254 or col_a < 0 or col_a > 65535 or col_b > 65535:
            raise Error("chain-end term: chal an element index, columns u16")
        var e = List[UInt8](length=END, fill=0)
        set_u16(e, 0, col_a)
        set_u16(e, 2, NONE if col_b < 0 else col_b)
        set_u16(e, 4, family)
        e[6] = UInt8(((coef % 127) + 127) % 127)
        e[7] = UInt8(chal + 1)
        e[8] = UInt8(1 if gate else 0)
        self.ends.extend(e^)

    # TODO(memory): `memory(family, z_col, addr, ts, value)` for spec 6.4: the sort key is (addr, ts), the
    # adjacency families (same address: value carried and timestamp increasing; new address: initial value)
    # are residual entries over the sorted columns, and the descriptor is KIND_MEMORY. Waits for a
    # profile with memory (configuration.md 3).

    def _descriptor(mut self, family: Int, z_col: Int, num: List[Int], den: List[Int], kind: Int, table: Int):
        var a = List[UInt8](length=ACC, fill=0)
        set_u16(a, 40, family)
        set_u16(a, 0, z_col)
        set_u16(a, 2, len(num))
        set_u16(a, 4, len(den))
        for j in range(len(num)):
            set_u16(a, 6 + 2 * j, num[j])
        for j in range(len(den)):
            set_u16(a, 22 + 2 * j, den[j])
        a[38] = UInt8(kind)
        a[39] = UInt8(table)
        self.accs.extend(a^)

    def add(mut self, family: Int, coef: Int, col_a: Int, k1_a: Int = 0, k2_a: Int = 0,
            col_b: Int = -1, k1_b: Int = 0, k2_b: Int = 0, mult: Int = 0, chal: Int = 0, basis: Int = -1, basis2: Int = -1) raises:
        """Shifts k_l are on H, already reduced to [0, h_l). An entry with the axis-2 gate must be
        linear: two columns and (X2 - e2) exceed the bound 2 h2 - 2 of spec section 8 and would alias
        on G. The axis-1 gate admits a quadratic entry (2 h1 - 1): the accumulator transition."""
        if col_b >= 0 and mult == 2:
            raise Error("axis-2 gated entries must be linear (spec 8 degree bound)")
        if chal < 0 or chal > 255:
            raise Error("chal is a u8: 0 or element index + 1")
        if mult < 0 or mult > 2 or basis >= 16 or basis2 >= 16:
            raise Error("mult is 0, 1, or 2; basis is -1 or a coordinate of E")
        var e = List[UInt8](length=ENTRY, fill=0)
        for v in [col_a, 2 * k1_a, 2 * k2_a, NONE if col_b < 0 else col_b, 2 * k1_b, 2 * k2_b, family]:
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


def entry(fam: Span[UInt8, _], k: Int) -> Entry:
    var o = k * ENTRY
    return Entry(col_a=get_u16(fam, o + 16), dj1_a=get_u16(fam, o + 18), dj2_a=get_u16(fam, o + 20),
                 col_b=get_u16(fam, o + 22), dj1_b=get_u16(fam, o + 24), dj2_b=get_u16(fam, o + 26),
                 mult=Int(fam[o + 28]), coef=Int(fam[o + 29]), family=get_u16(fam, o + 30),
                 chal=Int(fam[o + 32]), basis=Int(fam[o + 33]), basis2=Int(fam[o + 34]))


def derived_chals(mut chals: List[UInt8], table: Span[UInt8, _]):
    """Host side of accumulate.k_derive_chals: append one element per table row to the sampled ones."""
    for i in range(len(table) // CHAL):
        var a = ext_one[4]() if Int(table[i * CHAL + 1]) == CHAL_ONE else list_e(chals, Int(table[i * CHAL + 1]))
        var b = ext_one[4]() if Int(table[i * CHAL + 2]) == CHAL_ONE else list_e(chals, Int(table[i * CHAL + 2]))
        var v = f_add(a, b) if Int(table[i * CHAL]) == CHAL_ADD else ext_mul[4](a, b)
        for t in range(16):
            chals.append(v[t])


def lookup_constant(table: Span[UInt8, _], w: Int, chals: Span[UInt8, _]) raises -> E:
    """C_T of spec 6.3 for a (K, w) byte table: prod_j (1 + beta) (delta + fp(t_j)) over
    prod_{j < K - 1} ((1 + beta) delta + fp(t_j) + beta fp(t_{j + 1})). A zero factor is rejected, and so is
    a table without two distinct consecutive entries: with no cross term the identity is met by f = (t, u, ..,
    u) against s = (u, .., u) for any u, so a lookup is only sound when the table has a break."""
    var beta = list_e(chals, 0)
    var delta = list_e(chals, 1)
    var ob = list_e(chals, 3)
    var obd = list_e(chals, 4)
    var k = len(table) // w
    var num = ext_one[4]()
    var den = ext_one[4]()
    var has_break = False
    for j in range(k):
        var fp = E(0)
        for i in range(w):
            fp[i] = table[j * w + i]
        var own = ext_mul[4](ob, f_add(delta, fp))
        if own == E(0):
            raise Error("lookup table constant has a zero factor")
        num = ext_mul[4](num, own)
        if j + 1 < k:
            var fp1 = E(0)
            for i in range(w):
                fp1[i] = table[(j + 1) * w + i]
            var pair = f_add(f_add(obd, fp), ext_mul[4](beta, fp1))
            if pair == E(0):
                raise Error("lookup table constant has a zero factor")
            if fp1 != fp:
                has_break = True
            den = ext_mul[4](den, pair)
    if not has_break:
        raise Error("lookup table needs two distinct entries")
    return ext_mul[4](num, ext_inv[4](den))


def kappa_of(en: Entry, alpha: E, chals: Span[UInt8, _]) -> E:
    """coef * alpha^family * chal * b_t; chals holds the stage-1 elements as e bytes each."""
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


def residual_at(fam: Span[UInt8, _], alpha: E, chals: Span[UInt8, _], z1: E, z2: E, e1: F2, e2: F2, reads: List[E]) -> E:
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


def _lagrange(n: Int, w: F2, z: E) raises -> List[E]:
    """L_i(z) for i < n on the cyclic group <w> of order n: (z^n - 1) / n * w^i / (z - w^i), or the
    indicator of i when z is w^i."""
    var out = List[E](capacity=n)
    var wi = ext_one[4]()
    var we = ext_embed[4](w)
    var n_inv = E(0)
    n_inv[0] = f_pow(SIMD[DType.uint8, 1](n % 127), 125)[0]
    var lead = ext_mul[4](f_sub(ext_pow[4](z, n), ext_one[4]()), n_inv)
    for i in range(n):
        var den = f_sub(z, wi)
        if den.reduce_or() == 0:
            out = List[E](length=n, fill=E(0))
            out[i] = ext_one[4]()
            return out^
        out.append(ext_mul[4](lead, ext_mul[4](wi, ext_inv[4](den))))
        wi = ext_mul[4](wi, we)
    return out^


def eval_values(vals: Span[UInt8, _], off: Int, m: Int, h1: Int, h2: Int, w1: F2, w2: F2, x1: E, x2: E) raises -> E:
    """A public column at a point from its values: `vals` holds one period, (h2 / m, h1) F bytes from `off`,
    and the column is a polynomial in (X1, X2^m), so it is the period's interpolant on <w2^m> at x2^m
    (docs/public-columns.md). Barycentric per axis: h1 + h2 / m inversions, then one F x E product per value."""
    var period = h2 // m
    var l1 = _lagrange(h1, w1, x1)
    var l2 = _lagrange(period, ext_pow[1](w2, m), ext_pow[4](x2, m))
    var acc = E(0)
    for t2 in range(period):
        var row = E(0)
        for t1 in range(h1):
            var v = vals[off + t2 * h1 + t1]
            if v != 0:
                row = f_add(row, f_mul(l1[t1], E(v)))
        acc = f_add(acc, ext_mul[4](row, l2[t2]))
    return acc


def eval_line(coeffs: Span[UInt8, _], off: Int, count: Int, x1: E) -> E:
    """A restriction line (`count` F2 coefficients from `off`, degree < count) at x1, Horner."""
    var acc = E(0)
    for k in range(count - 1, -1, -1):
        acc = f_add(ext_mul[4](acc, x1), ext_embed[4](F2(coeffs[off + 2 * k], coeffs[off + 2 * k + 1])))
    return acc


def value_bytes(publics: Span[UInt8, _], h1: Int, h2: Int) -> Int:
    """Host bytes of every public column's period: sum h1 (h2 / m)."""
    var n = 0
    for i in range(len(publics) // PUB):
        n += h1 * (h2 // get_u16(publics, i * PUB))
    return n


def tile_values(publics: Span[UInt8, _], values: Span[UInt8, _], h1: Int, h2: Int) -> List[UInt8]:
    """The periods as full (column, x2, x1) value tables on H, the layout `idft2` reads."""
    var out = List[UInt8](capacity=(len(publics) // PUB) * h2 * h1)
    var src = 0
    for i in range(len(publics) // PUB):
        var period = h2 // get_u16(publics, i * PUB)
        for x2 in range(h2):
            for t in range(h1):
                out.append(values[src + (x2 % period) * h1 + t])
        src += period * h1
    return out^
