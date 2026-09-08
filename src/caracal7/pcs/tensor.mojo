"""The tensor form of the tail's queries (spec 9.1, 9.3; paper 6.6): the verifier's side.

Every functional the tail batches is, per odd index r, a sum of products over the binary slot digits:
the evaluation query w_z (sixteen products per point and r once the slot cases of 9.1 are split; the
per-bit factors depend on r through Par), the level-1 consistency rows g_{s,tau} (the coordinate
functional is a trace, so sum_tau beta_tau coord_tau(V) = sum_j mu_j sigma_j(V) over the four
conjugates: four products per position and r), and the tail rows pt^row (one per position and r).
A `Unit` is one product: its r, a scalar, and a factor pair per binary digit. Folding a digit at rho
multiplies the scalar by (1 - rho) f(0) + rho f(1) (the fold weight of tail.mojo); the odd digit is
never folded, so the clear check sums the units per r against the clear vector. The verifier never
holds a vector of length N.
"""

from caracal7.core.field import F4, E, f_add, f_sub, f_mul, f_pow, ext_mul, ext_pow, ext_inv, ext_embed, ext_one
from caracal7.core.params import Params
from caracal7.pcs.open import host_table
from caracal7.core.bytes import list_e


struct Unit(Copyable, Movable):
    """scalar * prod_d f[2 d + bit_d(index)] on the slots with odd index r."""
    var r: Int
    var scalar: E
    var f: List[E]

    def __init__(out self, r: Int, scalar: E, digits: Int):
        self.r = r
        self.scalar = scalar
        self.f = List[E](length=2 * digits, fill=ext_one[4]())

    def geo(mut self, d0: Int, count: Int, base: E):
        """Digits d0 .. d0 + count - 1 carry (1, base^(2^k)): the product is base^(index)."""
        var pw = base
        for k in range(count):
            self.f[2 * (d0 + k) + 1] = pw
            pw = ext_mul[4](pw, pw)

    def delta(mut self, d0: Int, count: Int, bit: Int):
        """Digits d0 .. d0 + count - 1 carry the indicator of `bit`."""
        for k in range(count):
            self.f[2 * (d0 + k) + 1 - bit] = E(0)

    def pair(mut self, d: Int, a: E, b: E):
        self.f[2 * d] = a
        self.f[2 * d + 1] = b

    def fold(mut self, d: Int, rho: E):
        var one = ext_one[4]()
        self.scalar = ext_mul[4](self.scalar, f_add(ext_mul[4](f_sub(one, rho), self.f[2 * d]), ext_mul[4](rho, self.f[2 * d + 1])))

    def at(self, first: Int, idx: Int, count: Int) -> E:
        """The product over digits first .. first + count - 1 at their bits in idx, times the scalar."""
        var w = self.scalar
        for k in range(count):
            w = ext_mul[4](w, self.f[2 * (first + k) + ((idx >> k) & 1)])
        return w


def _neg(a: E) -> E:
    return f_sub(E(0), a)


def _scal(x: UInt8, n: Int) -> UInt8:
    return f_pow(SIMD[DType.uint8, 1](x), n)[0]


def query_units[p: Params](z1: E, z2: E, weight: E, rho1: UInt8, rho2: UInt8, mut out: List[Unit]) raises:
    """weight * w_z as units (spec 9.1, the slot cases of `slot_weight`). Digits: t, the a1 - 1 bits of
    x1', the a2 bits of x2 (the top one last). On x1' != 0 the weight is Mon (x) I + Par (x, r) J with
    I = (1, i), J = (1, -i) on t; Par_l(x_l) = K_l q_l^(x_l) for x_l != 0 and 1 at x_l = 0, so Par is a
    geometric product plus deltas. The slab x1' = 0 is subtracted and rebuilt from its four cases."""
    comptime A1 = 1 << p.a1
    comptime A2 = 1 << p.a2
    comptime H1 = 1 << (p.a1 - 1)
    comptime H2 = 1 << (p.a2 - 1)
    comptime D = p.a1 + p.a2
    var tab = host_table[p](z1, z2, rho1, rho2)
    var one = ext_one[4]()
    var im = E(0)
    im[1] = 1
    var nim = _neg(im)
    var z1inv = ext_inv[4](z1)
    var z2inv = ext_inv[4](z2)
    var z1a = ext_pow[4](z1, A1)
    var z2a = ext_pow[4](z2, A2)
    var z1h = ext_pow[4](z1, H1)
    var z2h = ext_pow[4](z2, H2)
    for r in range(p.m1 * p.m2):
        var r1 = r % p.m1
        var r2 = r // p.m1
        var s = ext_mul[4](weight, ext_mul[4](list_e(tab, A1 + A2 + r1), list_e(tab, A1 + A2 + p.m1 + r2)))
        var r1v = _scal(rho1, r1)
        var r2v = _scal(rho2, r2)
        var k1 = f_mul(z1a, E(_scal(r1v, 125)))
        var k2 = f_mul(z2a, E(_scal(r2v, 125)))
        var q1 = f_mul(z1inv, E(_scal(r1v, 1 << (7 - p.a1))))
        var q2 = f_mul(z2inv, E(_scal(r2v, 1 << (7 - p.a2))))
        var k1k2 = ext_mul[4](k1, k2)
        var k1c = ext_mul[4](k1, f_sub(one, k2))
        var e7 = ext_mul[4](ext_mul[4](k1, ext_pow[4](q1, H1)), k2)     # Par at x = (H1, low), low != 0
        # x1' != 0: Mon I + Par J over the whole space, minus the same on the slab x1' = 0
        var u = Unit(r, s, D)
        u.geo(1, p.a1 - 1, z1)
        u.geo(p.a1, p.a2, z2)
        u.pair(0, one, im)
        out.append(u^)
        u = Unit(r, ext_mul[4](s, k1k2), D)
        u.geo(1, p.a1 - 1, q1)
        u.geo(p.a1, p.a2, q2)
        u.pair(0, one, nim)
        out.append(u^)
        u = Unit(r, ext_mul[4](s, k1c), D)
        u.geo(1, p.a1 - 1, q1)
        u.delta(p.a1, p.a2, 0)
        u.pair(0, one, nim)
        out.append(u^)
        u = Unit(r, _neg(s), D)
        u.delta(1, p.a1 - 1, 0)
        u.geo(p.a1, p.a2, z2)
        u.pair(0, one, im)
        out.append(u^)
        u = Unit(r, _neg(ext_mul[4](s, k1k2)), D)
        u.delta(1, p.a1 - 1, 0)
        u.geo(p.a1, p.a2, q2)
        u.pair(0, one, nim)
        out.append(u^)
        u = Unit(r, _neg(ext_mul[4](s, k1c)), D)
        u.delta(1, p.a1 - 1, 0)
        u.delta(p.a1, p.a2, 0)
        u.pair(0, one, nim)
        out.append(u^)
        # the slab: x2 = 0 and x2 = H2 hold Mon of (t H1, x2)
        u = Unit(r, s, D)
        u.delta(1, p.a1 - 1, 0)
        u.delta(p.a1, p.a2, 0)
        u.pair(0, one, z1h)
        out.append(u^)
        u = Unit(r, ext_mul[4](s, z2h), D)
        u.delta(1, p.a1 - 1, 0)
        u.delta(p.a1, p.a2 - 1, 0)
        u.delta(D - 1, 1, 1)
        u.pair(0, one, z1h)
        out.append(u^)
        # 0 < x2 < H2: x = (0, x2), Par = K2 q2^x2; the geometric form minus its value at x2 = 0
        u = Unit(r, s, D)
        u.delta(1, p.a1 - 1, 0)
        u.delta(D - 1, 1, 0)
        u.geo(p.a1, p.a2 - 1, z2)
        u.pair(0, one, im)
        out.append(u^)
        u = Unit(r, ext_mul[4](s, k2), D)
        u.delta(1, p.a1 - 1, 0)
        u.delta(D - 1, 1, 0)
        u.geo(p.a1, p.a2 - 1, q2)
        u.pair(0, one, nim)
        out.append(u^)
        u = Unit(r, _neg(s), D)
        u.delta(1, p.a1 - 1, 0)
        u.delta(p.a1, p.a2, 0)
        u.pair(0, one, im)
        out.append(u^)
        u = Unit(r, _neg(ext_mul[4](s, k2)), D)
        u.delta(1, p.a1 - 1, 0)
        u.delta(p.a1, p.a2, 0)
        u.pair(0, one, nim)
        out.append(u^)
        # x2 > H2: x = (H1, x2 - H2), Mon = z1^H1 z2^low, Par = K1 q1^H1 K2 q2^low; minus the value at low = 0
        u = Unit(r, ext_mul[4](s, z1h), D)
        u.delta(1, p.a1 - 1, 0)
        u.delta(D - 1, 1, 1)
        u.geo(p.a1, p.a2 - 1, z2)
        u.pair(0, one, im)
        out.append(u^)
        u = Unit(r, ext_mul[4](s, e7), D)
        u.delta(1, p.a1 - 1, 0)
        u.delta(D - 1, 1, 1)
        u.geo(p.a1, p.a2 - 1, q2)
        u.pair(0, one, nim)
        out.append(u^)
        u = Unit(r, _neg(ext_mul[4](s, z1h)), D)
        u.delta(1, p.a1 - 1, 0)
        u.delta(p.a1, p.a2 - 1, 0)
        u.delta(D - 1, 1, 1)
        u.pair(0, one, im)
        out.append(u^)
        u = Unit(r, _neg(ext_mul[4](s, e7)), D)
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


def consistency_units[p: Params](pt: F4, weights: InlineArray[E, 4], dual: InlineArray[F4, 4], mut out: List[Unit]):
    """sum_tau weights[tau] g_{pt,tau} as units: g_{pt,tau}[slot(i, j)] = coord_tau(b_j pt^i) (tail.mojo),
    i over (t, x1', x2 >> 2, r) and j the two low bits of x2, so per conjugate sigma the product is
    sigma(pt)^(t + 2 x1' + 2^a1 (x2 >> 2)) sigma(b_j) sigma(pt)^(2^(D - 2) r) with weight
    mu = sum_tau weights[tau] sigma(lambda_tau)."""
    comptime D = p.a1 + p.a2
    var iu = F4(0, 1, 0, 0)
    var ju = F4(0, 0, 1, 0)
    for j in range(4):
        var b = f4_frob(pt, j)
        var mu = E(0)
        for tau in range(4):
            mu = f_add(mu, ext_mul[4](weights[tau], ext_embed[4](f4_frob(dual[tau], j))))
        var be = ext_embed[4](b)
        var bhi = ext_pow[4](be, 1 << p.a1)
        var br = ext_pow[4](be, 1 << (D - 2))
        var pr = ext_one[4]()
        for r in range(p.m1 * p.m2):
            var u = Unit(r, ext_mul[4](mu, pr), D)
            u.geo(0, p.a1, be)
            u.pair(p.a1, ext_one[4](), ext_embed[4](f4_frob(iu, j)))
            u.pair(p.a1 + 1, ext_one[4](), ext_embed[4](f4_frob(ju, j)))
            u.geo(p.a1 + 2, p.a2 - 2, bhi)
            out.append(u^)
            pr = ext_mul[4](pr, br)


def row_units(pt: F4, weight: E, first: Int, digits: Int, m: Int, mut out: List[Unit]):
    """weight * pt^row as units, row over the digits from `first` then r (a tail level's rows)."""
    var be = ext_embed[4](pt)
    var u_digits = digits - first
    var br = ext_pow[4](be, 1 << u_digits)
    var pr = ext_one[4]()
    for r in range(m):
        var u = Unit(r, ext_mul[4](weight, pr), digits)
        u.geo(first, u_digits, be)
        out.append(u^)
        pr = ext_mul[4](pr, br)


def clear_value(units: List[Unit], y: Span[UInt8, _], first: Int, digits: Int) -> E:
    """<y, sum of units> on the clear vector, index over the digits from `first` then r."""
    var count = digits - first
    var acc = E(0)
    for i in range(len(units)):
        for idx in range(1 << count):
            acc = f_add(acc, ext_mul[4](units[i].at(first, idx, count), list_e(y, idx + (1 << count) * units[i].r)))
    return acc
