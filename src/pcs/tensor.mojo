"""The tensor form of the tail's queries (spec 9.1, 9.3; paper 6.6): the verifier's side.

Every functional the tail batches is, per odd index r, a sum of products over the binary slot digits:
the evaluation query w_z (sixteen products per point once the slot cases of 9.1 are split; the
per-bit factors depend on r through Par), the level-1 consistency rows g_{s,tau} (the coordinate
functional is a trace, so sum_tau beta_tau coord_tau(V) = sum_j mu_j sigma_j(V) over the four
conjugates: four products per position), and the tail rows pt^row (one per position).
A `Unit` is one product over all M = m1 m2 odd indices at once: a scalar, a per-r1 and a per-r2 scalar
list, and a factor pair per binary digit. The odd index r = r1 + m1 r2 enters every functional as a
product of an r1 part and an r2 part: the query's Par factors are q1^x1 = z1^-x1 rho1^(c1 r1 x1) and
q2^x2 likewise, so a digit of x1 carries the r-free base and a twist rho1^(c1 2^k r1); the geometric
row scalars br^r are br^r1 (br^m1)^r2. Folding an untwisted digit at rho multiplies the scalar by
(1 - rho) f(0) + rho f(1) (the fold weight of tail.mojo), a twisted one multiplies the m1 or m2 scalars
of its axis; the odd digit is never folded, so the clear check sums the units per r against the clear
vector. The verifier never holds a vector of length N, and the unit count does not grow with M.
"""

from std.bit import log2_floor

from core.field import F4, E, f_add, f_sub, f_mul, f_pow, ext_mul, ext_pow, ext_inv0, ext_embed, ext_one, E_LEVEL
from core.params import Params
from pcs.open import host_table
from core.bytes import list_e


struct Unit(Copyable, Movable):
    """scalar * s1[r1] * s2[r2] * prod_d f[2 d + bit_d(index)] * prod_{twisted d with bit 1} rho^(tw_d r) on the
    slots with odd index r = r1 + m1 r2: tw_d > 0 twists by rho1^(tw_d r1), tw_d < 0 by rho2^(-tw_d r2)."""
    var scalar: E
    var s1: List[E]
    var s2: List[E]
    var f: List[E]
    var tw: List[Int]
    var rho1: UInt8
    var rho2: UInt8

    def __init__(out self, scalar: E, var s1: List[E], var s2: List[E], digits: Int, rho1: UInt8 = 1, rho2: UInt8 = 1):
        self.scalar = scalar
        self.s1 = s1^
        self.s2 = s2^
        self.f = List[E](length=2 * digits, fill=ext_one[E_LEVEL]())
        self.tw = List[Int](length=digits, fill=0)
        self.rho1 = rho1
        self.rho2 = rho2

    def geo(mut self, d0: Int, count: Int, base: E, tw: Int = 0):
        """Digits d0 .. d0 + count - 1 carry (1, base^(2^k)), twisted by tw 2^k: the product is
        base^(index) rho^(tw index r)."""
        var pw = base
        var t = tw
        for k in range(count):
            self.f[2 * (d0 + k) + 1] = pw
            self.tw[d0 + k] = t
            pw = ext_mul[E_LEVEL](pw, pw)
            t *= 2

    def delta(mut self, d0: Int, count: Int, bit: Int):
        """Digits d0 .. d0 + count - 1 carry the indicator of `bit`."""
        for k in range(count):
            self.f[2 * (d0 + k) + 1 - bit] = E(0)

    def pair(mut self, d: Int, a: E, b: E):
        self.f[2 * d] = a
        self.f[2 * d + 1] = b

    def _twist(self, d: Int, r: Int) -> E:
        """The twist of digit d at bit 1 for the index r of its axis, broadcast for a lane-wise f_mul."""
        var rho = self.rho1 if self.tw[d] > 0 else self.rho2
        return E(_scal(rho, (abs(self.tw[d]) * r) % 126))

    def fold(mut self, d: Int, rho: E):
        var one = ext_one[E_LEVEL]()
        var a = ext_mul[E_LEVEL](f_sub(one, rho), self.f[2 * d])
        var b = ext_mul[E_LEVEL](rho, self.f[2 * d + 1])
        if self.tw[d] == 0:
            self.scalar = ext_mul[E_LEVEL](self.scalar, f_add(a, b))
        elif self.tw[d] > 0:
            for r in range(len(self.s1)):
                self.s1[r] = ext_mul[E_LEVEL](self.s1[r], f_add(a, f_mul(b, self._twist(d, r))))
        else:
            for r in range(len(self.s2)):
                self.s2[r] = ext_mul[E_LEVEL](self.s2[r], f_add(a, f_mul(b, self._twist(d, r))))

    def at(self, first: Int, idx: Int, count: Int, r1: Int, r2: Int) -> E:
        """The unit at odd index (r1, r2) over digits first .. first + count - 1 at their bits in idx."""
        var w = ext_mul[E_LEVEL](self.scalar, ext_mul[E_LEVEL](self.s1[r1], self.s2[r2]))
        for k in range(count):
            var d = first + k
            var bit = (idx >> k) & 1
            w = ext_mul[E_LEVEL](w, self.f[2 * d + bit])
            if bit == 1 and self.tw[d] != 0:
                w = f_mul(w, self._twist(d, r1 if self.tw[d] > 0 else r2))
        return w


def _neg(a: E) -> E:
    return f_sub(E(0), a)


def _scal(x: UInt8, n: Int) -> UInt8:
    return f_pow(SIMD[DType.uint8, 1](x), n)[0]


def _prod(a: List[E], b: List[E]) -> List[E]:
    var out = List[E](capacity=len(a))
    for k in range(len(a)):
        out.append(ext_mul[E_LEVEL](a[k], b[k]))
    return out^


def query_units[p: Params](z1: E, z2: E, weight: E, rho1: UInt8, rho2: UInt8, mut out: List[Unit]):
    """weight * w_z as units (spec 9.1, the slot cases of `slot_weight`). Digits: t, the a1 - 1 bits of
    x1', the a2 bits of x2 (the top one last). On x1' != 0 the weight is Mon (x) I + Par (x, r) J with
    I = (1, i), J = (1, -i) on t; Par_l(x_l) = K_l q_l^(x_l) for x_l != 0 and 1 at x_l = 0, so Par is a
    geometric product plus deltas, with q_l = z_l^-1 rho_l^(c_l r_l) as a twisted base. The slab x1' = 0
    is subtracted and rebuilt from its four cases. Sixteen units per point."""
    comptime A1 = 1 << p.a1
    comptime A2 = 1 << p.a2
    comptime H1 = 1 << (p.a1 - 1)
    comptime H2 = 1 << (p.a2 - 1)
    comptime D = p.a1 + p.a2
    comptime C1 = 1 << (7 - p.a1)
    comptime C2 = 1 << (7 - p.a2)
    var tab = host_table[p](z1, z2, rho1, rho2)
    var one = ext_one[E_LEVEL]()
    var im = E(0)
    im[1] = 1
    var nim = _neg(im)
    var z1inv = ext_inv0[E_LEVEL](z1)                  # 0 -> 0: every use of z^-1 multiplies a factor that is 0 when z is
    var z2inv = ext_inv0[E_LEVEL](z2)
    var z1a = ext_pow[E_LEVEL](z1, A1)
    var z2a = ext_pow[E_LEVEL](z2, A2)
    var z1h = ext_pow[E_LEVEL](z1, H1)
    var z2h = ext_pow[E_LEVEL](z2, H2)
    var nw = _neg(weight)
    # the r1 part: the table's r1 scalar, K1 = z1^A1 rho1^-r1, and K1 q1^H1 (q1^H1 = z1^-H1 rho1^(64 r1))
    var t1 = List[E](capacity=p.m1)
    var k1 = List[E](capacity=p.m1)
    var e1 = List[E](capacity=p.m1)
    var z1hinv = ext_pow[E_LEVEL](z1inv, H1)
    for r1 in range(p.m1):
        t1.append(list_e(tab, A1 + A2 + r1))
        k1.append(f_mul(z1a, E(_scal(rho1, (125 * r1) % 126))))
        e1.append(f_mul(ext_mul[E_LEVEL](k1[r1], z1hinv), E(_scal(rho1, (64 * r1) % 126))))
    # the r2 part: the table's r2 scalar, K2 = z2^A2 rho2^-r2 and 1 - K2
    var t2 = List[E](capacity=p.m2)
    var k2 = List[E](capacity=p.m2)
    var c2 = List[E](capacity=p.m2)
    for r2 in range(p.m2):
        t2.append(list_e(tab, A1 + A2 + p.m1 + r2))
        k2.append(f_mul(z2a, E(_scal(rho2, (125 * r2) % 126))))
        c2.append(f_sub(one, k2[r2]))
    var t1k1 = _prod(t1, k1)
    var t1e1 = _prod(t1, e1)
    var t2k2 = _prod(t2, k2)
    var t2c2 = _prod(t2, c2)


    # x1' != 0, Par J over the whole space, minus the slab x1' = 0
    var u = Unit(weight, t1k1.copy(), t2k2.copy(), D, rho1, rho2)
    u.geo(1, p.a1 - 1, z1inv, C1)
    u.geo(p.a1, p.a2, z2inv, -C2)
    u.pair(0, one, nim)
    out.append(u^)
    u = Unit(weight, t1k1.copy(), t2c2.copy(), D, rho1, rho2)
    u.geo(1, p.a1 - 1, z1inv, C1)
    u.delta(p.a1, p.a2, 0)
    u.pair(0, one, nim)
    out.append(u^)
    u = Unit(nw, t1k1.copy(), t2k2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.geo(p.a1, p.a2, z2inv, -C2)
    u.pair(0, one, nim)
    out.append(u^)
    # 0 < x2 < H2: Par = K2 q2^x2
    u = Unit(weight, t1.copy(), t2k2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.delta(D - 1, 1, 0)
    u.geo(p.a1, p.a2 - 1, z2inv, -C2)
    u.pair(0, one, nim)
    out.append(u^)
    # x2 > H2: Par = K1 q1^H1 K2 q2^low
    u = Unit(weight, t1e1.copy(), t2k2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.delta(D - 1, 1, 1)
    u.geo(p.a1, p.a2 - 1, z2inv, -C2)
    u.pair(0, one, nim)
    out.append(u^)
    # x1' != 0: Mon I over the whole space, minus the slab x1' = 0
    u = Unit(weight, t1.copy(), t2.copy(), D, rho1, rho2)
    u.geo(1, p.a1 - 1, z1)
    u.geo(p.a1, p.a2, z2)
    u.pair(0, one, im)
    out.append(u^)
    u = Unit(nw, t1.copy(), t2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.geo(p.a1, p.a2, z2)
    u.pair(0, one, im)
    out.append(u^)
    u = Unit(nw, t1k1.copy(), t2c2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2, 0)
    u.pair(0, one, nim)
    out.append(u^)
    # the slab: x2 = 0 and x2 = H2 hold Mon of (t H1, x2)
    u = Unit(weight, t1.copy(), t2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2, 0)
    u.pair(0, one, z1h)
    out.append(u^)
    u = Unit(ext_mul[E_LEVEL](weight, z2h), t1.copy(), t2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2 - 1, 0)
    u.delta(D - 1, 1, 1)
    u.pair(0, one, z1h)
    out.append(u^)
    # 0 < x2 < H2: x = (0, x2); the geometric form minus its value at x2 = 0
    u = Unit(weight, t1.copy(), t2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.delta(D - 1, 1, 0)
    u.geo(p.a1, p.a2 - 1, z2)
    u.pair(0, one, im)
    out.append(u^)
    u = Unit(nw, t1.copy(), t2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2, 0)
    u.pair(0, one, im)
    out.append(u^)
    u = Unit(nw, t1.copy(), t2k2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2, 0)
    u.pair(0, one, nim)
    out.append(u^)
    # x2 > H2: x = (H1, x2 - H2), Mon = z1^H1 z2^low; minus the value at low = 0
    u = Unit(ext_mul[E_LEVEL](weight, z1h), t1.copy(), t2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.delta(D - 1, 1, 1)
    u.geo(p.a1, p.a2 - 1, z2)
    u.pair(0, one, im)
    out.append(u^)
    u = Unit(ext_mul[E_LEVEL](nw, z1h), t1.copy(), t2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2 - 1, 0)
    u.delta(D - 1, 1, 1)
    u.pair(0, one, im)
    out.append(u^)
    u = Unit(nw, t1e1.copy(), t2k2.copy(), D, rho1, rho2)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2 - 1, 0)
    u.delta(D - 1, 1, 1)
    u.pair(0, one, nim)
    out.append(u^)


def f4_frob(w: F4, j: Int) -> F4:
    """The j-th conjugate w^(127^j)."""
    var v = w
    for _ in range(j):
        v = ext_pow[2](v, 127)
    return v


def f4_trace(w: F4) -> UInt8:
    """Tr_{F4/F}(w): the sum of the four conjugates, in F."""
    var acc = F4(0)
    for j in range(4):
        acc = f_add(acc, f4_frob(w, j))
    return acc[0]


def f4_dual() raises -> InlineArray[F4, 4]:
    """lambda_tau with coord_tau(V) = Tr(lambda_tau V) for the coordinate basis b = (1, i, j', i j'):
    lambda_tau = sum_l Tinv[tau, l] b_l over the trace form T[k, l] = Tr(b_k b_l)."""
    var a = InlineArray[UInt8, 32](fill=0)               # 4 rows of (T | I)
    for k in range(4):
        var bk = F4(0)
        bk[k] = 1
        for l in range(4):
            var bl = F4(0)
            bl[l] = 1
            a[k * 8 + l] = f4_trace(ext_mul[2](bk, bl))
        a[k * 8 + 4 + k] = 1
    for col in range(4):
        var piv = col
        while piv < 4 and a[piv * 8 + col] == 0:
            piv += 1
        if piv == 4:
            raise Error("trace form is singular")
        for c in range(8):
            var t = a[col * 8 + c]
            a[col * 8 + c] = a[piv * 8 + c]
            a[piv * 8 + c] = t
        var inv = _scal(a[col * 8 + col], 125)
        for c in range(8):
            a[col * 8 + c] = f_mul(SIMD[DType.uint8, 1](a[col * 8 + c]), SIMD[DType.uint8, 1](inv))[0]
        for row in range(4):
            if row != col and a[row * 8 + col] != 0:
                var m = a[row * 8 + col]
                for c in range(8):
                    var prod = f_mul(SIMD[DType.uint8, 1](m), SIMD[DType.uint8, 1](a[col * 8 + c]))
                    a[row * 8 + c] = f_sub(SIMD[DType.uint8, 1](a[row * 8 + c]), prod)[0]
    var out = InlineArray[F4, 4](fill=F4(0))
    for tau in range(4):
        for l in range(4):
            out[tau][l] = a[tau * 8 + 4 + l]
    return out^


def _geometric(base: E, m1: Int, m2: Int) -> Tuple[List[E], List[E]]:
    """base^r over r = r1 + m1 r2 as (base^r1, base^(m1 r2))."""
    var s1 = List[E](capacity=m1)
    var s2 = List[E](capacity=m2)
    var pr = ext_one[E_LEVEL]()
    for _ in range(m1):
        s1.append(pr)
        pr = ext_mul[E_LEVEL](pr, base)
    var bm = pr                                  # base^m1
    pr = ext_one[E_LEVEL]()
    for _ in range(m2):
        s2.append(pr)
        pr = ext_mul[E_LEVEL](pr, bm)
    return (s1^, s2^)


def consistency_units[p: Params](pt: F4, weights: InlineArray[E, 4], dual: InlineArray[F4, 4], cw: Int, mut out: List[Unit]):
    """sum_tau weights[tau] g_{pt,cw,tau} as units: g_{pt,cw,tau}[slot(i, j)] = coord_tau(b_j pt^i') on the slots
    of codeword cw and 0 elsewhere (tail.mojo), i over (t, x1', x2 >> 2, r), j the two low bits of x2 and
    (i', cw) = split_index(i): per conjugate sigma the product is sigma(pt)^(i' mod 2^LOW) sigma(b_j)
    sigma(pt)^(2^LOW r) times the indicator of cw on the top log2 n_cw binary digits, with weight
    mu = sum_tau weights[tau] sigma(lambda_tau). Packed bit k < a1 is unit digit k, bit a1 + m is digit a1 + 2 + m."""
    comptime D = p.a1 + p.a2
    comptime LOW = D - 2 - log2_floor(p.n_cw())
    var iu = F4(0, 1, 0, 0)
    var ju = F4(0, 0, 1, 0)
    for j in range(4):
        var b = f4_frob(pt, j)
        var mu = E(0)
        for tau in range(4):
            mu = f_add(mu, ext_mul[E_LEVEL](weights[tau], ext_embed[E_LEVEL](f4_frob(dual[tau], j))))
        var be = ext_embed[E_LEVEL](b)
        var g = _geometric(ext_pow[E_LEVEL](be, 1 << LOW), p.m1, p.m2)
        var u = Unit(mu, g[0].copy(), g[1].copy(), D)
        var pw = be
        for k in range(D - 2):
            var d = k if k < p.a1 else k + 2
            if k < LOW:
                u.pair(d, ext_one[E_LEVEL](), pw)
            else:
                u.delta(d, 1, (cw >> (k - LOW)) & 1)
            pw = ext_mul[E_LEVEL](pw, pw)
        u.pair(p.a1, ext_one[E_LEVEL](), ext_embed[E_LEVEL](f4_frob(iu, j)))
        u.pair(p.a1 + 1, ext_one[E_LEVEL](), ext_embed[E_LEVEL](f4_frob(ju, j)))
        out.append(u^)


def row_units(pt: F4, weight: E, first: Int, digits: Int, m1: Int, m2: Int, mut out: List[Unit], cw: Int = 0, n_cw: Int = 1):
    """weight * pt^row' on the rows of codeword cw as units, row over the digits from `first` then r (a tail
    level's rows); the codeword is the top log2 n_cw of those digits (tail.tail_split), an indicator."""
    var be = ext_embed[E_LEVEL](pt)
    var u_digits = digits - first
    var low = u_digits - log2_floor(n_cw)
    var g = _geometric(ext_pow[E_LEVEL](be, 1 << low), m1, m2)
    var u = Unit(weight, g[0].copy(), g[1].copy(), digits)
    u.geo(first, low, be)
    for k in range(low, u_digits):
        u.delta(first + k, 1, (cw >> (k - low)) & 1)
    out.append(u^)


def clear_value(units: List[Unit], y: Span[UInt8, _], first: Int, digits: Int) -> E:
    """<y, sum of units> on the clear vector, index over the digits from `first` then r = r1 + m1 r2. Per
    unit and index the r-free product is taken once and the twists collapse to one power per axis, so the
    inner sums cost one E product and a lane-wise scalar per clear entry."""
    var count = digits - first
    var acc = E(0)
    for i in range(len(units)):
        ref u = units[i]
        var m1 = len(u.s1)
        var m2 = len(u.s2)
        for idx in range(1 << count):
            var w0 = u.scalar
            var e1 = 0
            var e2 = 0
            for k in range(count):
                var d = first + k
                var bit = (idx >> k) & 1
                w0 = ext_mul[E_LEVEL](w0, u.f[2 * d + bit])
                if bit == 1:
                    if u.tw[d] > 0:
                        e1 += u.tw[d]
                    elif u.tw[d] < 0:
                        e2 -= u.tw[d]
            var outer = E(0)
            for r2 in range(m2):
                var inner = E(0)
                for r1 in range(m1):
                    var yv = list_e(y, idx + (1 << count) * (r1 + m1 * r2))
                    if e1 != 0:
                        yv = f_mul(yv, E(_scal(u.rho1, (e1 * r1) % 126)))
                    inner = f_add(inner, ext_mul[E_LEVEL](u.s1[r1], yv))
                if e2 != 0:
                    inner = f_mul(inner, E(_scal(u.rho2, (e2 * r2) % 126)))
                outer = f_add(outer, ext_mul[E_LEVEL](u.s2[r2], inner))
            acc = f_add(acc, ext_mul[E_LEVEL](w0, outer))
    return acc
