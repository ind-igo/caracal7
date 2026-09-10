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
The consistency and row units of one (position, conjugate) share their factor lists across the M = m1 m2
odd indices, so a `Unit` is a group with a per-r scalar list; of the query units, the eleven kinds per
point whose factors are r-free group too and the five with Par factors (q1, q2 depend on r) stay
single. ponytail: widen the host E product when the client grid (M = 567) is measured.
"""

from caracal7.core.field import F4, E, f_add, f_sub, f_mul, f_pow, ext_mul, ext_pow, ext_inv0, ext_embed, ext_one
from caracal7.core.params import Params
from caracal7.pcs.open import host_table
from caracal7.core.bytes import list_e


struct Unit(Copyable, Movable):
    """scalar * scalars[k] * prod_d f[2 d + bit_d(index)] on the slots with odd index r0 + k: a group of
    units that share their factor lists and differ by the per-r scalar. Folding a digit or weighing a
    level touches the shared scalar only, so a group of M units costs one product, not M."""
    var r0: Int
    var scalar: E
    var scalars: List[E]
    var f: List[E]

    def __init__(out self, r: Int, scalar: E, digits: Int):
        """A single unit at odd index r."""
        self.r0 = r
        self.scalar = scalar
        self.scalars = [ext_one[4]()]
        self.f = List[E](length=2 * digits, fill=ext_one[4]())

    def __init__(out self, var scalars: List[E], digits: Int):
        """A group at odd indices 0 .. len(scalars) - 1 with the given per-r scalars."""
        self.r0 = 0
        self.scalar = ext_one[4]()
        self.scalars = scalars^
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
        """The product over digits first .. first + count - 1 at their bits in idx, times the shared
        scalar (the unit at r0 + k is this times scalars[k])."""
        var w = self.scalar
        for k in range(count):
            w = ext_mul[4](w, self.f[2 * (first + k) + ((idx >> k) & 1)])
        return w


def _neg(a: E) -> E:
    return f_sub(E(0), a)


def _scal(x: UInt8, n: Int) -> UInt8:
    return f_pow(SIMD[DType.uint8, 1](x), n)[0]


def query_units[p: Params](z1: E, z2: E, weight: E, rho1: UInt8, rho2: UInt8, mut out: List[Unit]):
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
    var z1inv = ext_inv0[4](z1)                  # 0 -> 0: every use of z^-1 multiplies a factor that is 0 when z is
    var z2inv = ext_inv0[4](z2)
    var z1a = ext_pow[4](z1, A1)
    var z2a = ext_pow[4](z2, A2)
    var z1h = ext_pow[4](z1, H1)
    var z2h = ext_pow[4](z2, H2)
    comptime M = p.m1 * p.m2
    # per r: the scalar s and the r-dependent Par scalars; the factor bases q1, q2 depend on r too
    var sc = List[E](capacity=M)                # s
    var s_k1k2 = List[E](capacity=M)
    var s_k1c = List[E](capacity=M)
    var s_k2 = List[E](capacity=M)
    var s_e7 = List[E](capacity=M)
    var s_z2h = List[E](capacity=M)
    var s_z1h = List[E](capacity=M)
    var n_s = List[E](capacity=M)               # negatives
    var n_k1c = List[E](capacity=M)
    var n_k2 = List[E](capacity=M)
    var n_z1h = List[E](capacity=M)
    var n_e7 = List[E](capacity=M)
    for r in range(M):
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
        sc.append(s)
        s_k1k2.append(ext_mul[4](s, k1k2))
        s_k1c.append(ext_mul[4](s, k1c))
        s_k2.append(ext_mul[4](s, k2))
        s_e7.append(ext_mul[4](s, e7))
        s_z2h.append(ext_mul[4](s, z2h))
        s_z1h.append(ext_mul[4](s, z1h))
        n_s.append(_neg(s))
        n_k1c.append(_neg(s_k1c[r]))
        n_k2.append(_neg(s_k2[r]))
        n_z1h.append(_neg(s_z1h[r]))
        n_e7.append(_neg(s_e7[r]))
        # the units whose factors carry q1 or q2 stay single per r
        # x1' != 0, Par J over the whole space, minus the slab x1' = 0
        var u = Unit(r, s_k1k2[r], D)
        u.geo(1, p.a1 - 1, q1)
        u.geo(p.a1, p.a2, q2)
        u.pair(0, one, nim)
        out.append(u^)
        u = Unit(r, s_k1c[r], D)
        u.geo(1, p.a1 - 1, q1)
        u.delta(p.a1, p.a2, 0)
        u.pair(0, one, nim)
        out.append(u^)
        u = Unit(r, _neg(s_k1k2[r]), D)
        u.delta(1, p.a1 - 1, 0)
        u.geo(p.a1, p.a2, q2)
        u.pair(0, one, nim)
        out.append(u^)
        # 0 < x2 < H2: Par = K2 q2^x2
        u = Unit(r, s_k2[r], D)
        u.delta(1, p.a1 - 1, 0)
        u.delta(D - 1, 1, 0)
        u.geo(p.a1, p.a2 - 1, q2)
        u.pair(0, one, nim)
        out.append(u^)
        # x2 > H2: Par = K1 q1^H1 K2 q2^low
        u = Unit(r, s_e7[r], D)
        u.delta(1, p.a1 - 1, 0)
        u.delta(D - 1, 1, 1)
        u.geo(p.a1, p.a2 - 1, q2)
        u.pair(0, one, nim)
        out.append(u^)
    # the units whose factors are r-free: one group each, the r dependence in the scalar list
    # x1' != 0: Mon I over the whole space, minus the slab x1' = 0
    var u = Unit(sc.copy(), D)
    u.geo(1, p.a1 - 1, z1)
    u.geo(p.a1, p.a2, z2)
    u.pair(0, one, im)
    out.append(u^)
    u = Unit(n_s.copy(), D)
    u.delta(1, p.a1 - 1, 0)
    u.geo(p.a1, p.a2, z2)
    u.pair(0, one, im)
    out.append(u^)
    u = Unit(n_k1c^, D)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2, 0)
    u.pair(0, one, nim)
    out.append(u^)
    # the slab: x2 = 0 and x2 = H2 hold Mon of (t H1, x2)
    u = Unit(sc.copy(), D)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2, 0)
    u.pair(0, one, z1h)
    out.append(u^)
    u = Unit(s_z2h^, D)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2 - 1, 0)
    u.delta(D - 1, 1, 1)
    u.pair(0, one, z1h)
    out.append(u^)
    # 0 < x2 < H2: x = (0, x2); the geometric form minus its value at x2 = 0
    u = Unit(sc^, D)
    u.delta(1, p.a1 - 1, 0)
    u.delta(D - 1, 1, 0)
    u.geo(p.a1, p.a2 - 1, z2)
    u.pair(0, one, im)
    out.append(u^)
    u = Unit(n_s^, D)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2, 0)
    u.pair(0, one, im)
    out.append(u^)
    u = Unit(n_k2^, D)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2, 0)
    u.pair(0, one, nim)
    out.append(u^)
    # x2 > H2: x = (H1, x2 - H2), Mon = z1^H1 z2^low; minus the value at low = 0
    u = Unit(s_z1h^, D)
    u.delta(1, p.a1 - 1, 0)
    u.delta(D - 1, 1, 1)
    u.geo(p.a1, p.a2 - 1, z2)
    u.pair(0, one, im)
    out.append(u^)
    u = Unit(n_z1h^, D)
    u.delta(1, p.a1 - 1, 0)
    u.delta(p.a1, p.a2 - 1, 0)
    u.delta(D - 1, 1, 1)
    u.pair(0, one, im)
    out.append(u^)
    u = Unit(n_e7^, D)
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
        var scalars = List[E](capacity=p.m1 * p.m2)
        for _ in range(p.m1 * p.m2):
            scalars.append(ext_mul[4](mu, pr))
            pr = ext_mul[4](pr, br)
        var u = Unit(scalars^, D)
        u.geo(0, p.a1, be)
        u.pair(p.a1, ext_one[4](), ext_embed[4](f4_frob(iu, j)))
        u.pair(p.a1 + 1, ext_one[4](), ext_embed[4](f4_frob(ju, j)))
        u.geo(p.a1 + 2, p.a2 - 2, bhi)
        out.append(u^)


def row_units(pt: F4, weight: E, first: Int, digits: Int, m: Int, mut out: List[Unit]):
    """weight * pt^row as units, row over the digits from `first` then r (a tail level's rows)."""
    var be = ext_embed[4](pt)
    var u_digits = digits - first
    var br = ext_pow[4](be, 1 << u_digits)
    var pr = ext_one[4]()
    var scalars = List[E](capacity=m)
    for _ in range(m):
        scalars.append(ext_mul[4](weight, pr))
        pr = ext_mul[4](pr, br)
    var u = Unit(scalars^, digits)
    u.geo(first, u_digits, be)
    out.append(u^)


def clear_value(units: List[Unit], y: Span[UInt8, _], first: Int, digits: Int) -> E:
    """<y, sum of units> on the clear vector, index over the digits from `first` then r."""
    var count = digits - first
    var acc = E(0)
    for i in range(len(units)):
        ref u = units[i]
        for idx in range(1 << count):
            var w = u.at(first, idx, count)
            var s = E(0)                                 # sum_k scalars[k] y[idx, r0 + k], then times w
            for k in range(len(u.scalars)):
                s = f_add(s, ext_mul[4](u.scalars[k], list_e(y, idx + (1 << count) * (u.r0 + k))))
            acc = f_add(acc, ext_mul[4](w, s))
    return acc
