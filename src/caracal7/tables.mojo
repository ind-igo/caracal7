"""Domains and twiddle tables (docs/design.md section 5). Host side, run once at setup.

Generators (recorded in docs/decisions.md):
    gamma2  primitive element of F2*, order 16128 = 2^8 * 63; omega_l = gamma2^(16128 / h_l)
    rho_l   = omega_l^(2^a_l), the order-m_l generator of mu_l (lies in F)
    g       generator of the order-L0 subgroup of F4*: gamma4^((127^4 - 1) / L0)
    gA      = g^M (order 2^b), gB = g^(2^b) (order M), where L0 = 2^b * M, M odd
    gamma4  a primitive element of F4*; coset k of an RS domain is gamma4^k D0

Leaf s of an RS domain with m cosets is the point gamma4^(s // L0) g^(s mod L0), s in [m L0).
"""

from max.gpu.host import DeviceContext, HostBuffer

from caracal7.field import F2, F4, f_add, f_sub, f_pow, f_mul, f_inv, ext_mul, ext_pow, ext_embed
from caracal7.params import Params

comptime F2_ORDER = 16128            # |F2*| = 127^2 - 1
comptime F4_ORDER = 260144640        # |F4*| = 127^4 - 1


def two_adic(n: Int) -> Int:
    var b = 0
    var m = n
    while m % 2 == 0:
        m //= 2
        b += 1
    return b


def modinv(a: Int, m: Int) -> Int:
    """a^-1 mod m by search; m is tiny (<= 9)."""
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return 0


def _is_one[k: Int](a: SIMD[DType.uint8, 1 << k]) -> Bool:
    return a[0] == 1 and (a - SIMD[DType.uint8, 1 << k](a[0]) * ext_embed[k, 1](SIMD[DType.uint8, 1](1))).reduce_or() == 0


def f2_primitive() raises -> F2:
    for lo in range(1, 127):
        for hi in range(1, 127):
            var c = F2(UInt8(lo), UInt8(hi))
            var ok = True
            for p in [2, 3, 7]:
                if _is_one[1](ext_pow[1](c, F2_ORDER // p)):
                    ok = False
            if ok:
                return c
    raise Error("no primitive element of F2 found")


def f4_primitive() raises -> F4:
    """A generator of F4*."""
    for lo in range(1, 127):
        for hi in range(1, 127):
            var c = F4(UInt8(lo), 0, UInt8(hi), 1)
            var ok = True
            for p in [2, 3, 5, 7, 1613]:
                if _is_one[2](ext_pow[2](c, F4_ORDER // p)):
                    ok = False
            if ok:
                return c
    raise Error("no primitive element of F4 found")


def f4_subgroup_generator(order: Int) raises -> F4:
    """An element of exact order `order` (order | 161280) in F4*."""
    for lo in range(1, 127):
        for hi in range(1, 127):
            var c = F4(UInt8(lo), 0, UInt8(hi), 1)
            var g = ext_pow[2](c, F4_ORDER // order)
            var ok = True
            for p in [2, 3, 5, 7]:
                if order % p == 0 and _is_one[2](ext_pow[2](g, order // p)):
                    ok = False
            if ok:
                return g
    raise Error("no generator found")


struct RsDomain(TrivialRegisterPassable):
    """An RS evaluation domain: m cosets gamma4^k D0 of the order-L0 subgroup D0 = <g> (spec 9.1)."""
    var L0: Int
    var m: Int
    var g: F4
    var gk: SIMD[DType.uint8, 16]     # gamma4^k for k < 4, 4 bytes each

    def __init__(out self, L0: Int, m: Int) raises:
        self.L0 = L0
        self.m = m
        self.g = f4_subgroup_generator(L0)
        var gamma = f4_primitive()
        self.gk = SIMD[DType.uint8, 16](0)
        var pw = F4(1, 0, 0, 0)
        for k in range(4):
            for c in range(4):
                self.gk[k * 4 + c] = pw[c]
            pw = ext_mul[2](pw, gamma)

    def rep(self, k: Int) -> F4:
        return F4(self.gk[4 * k], self.gk[4 * k + 1], self.gk[4 * k + 2], self.gk[4 * k + 3])

    def point(self, s: Int) -> F4:
        """The point of leaf s in [m L0)."""
        return ext_mul[2](self.rep(s // self.L0), ext_pow[2](self.g, s % self.L0))


struct Domains(TrivialRegisterPassable):
    var omega1: F2
    var omega2: F2
    var rho1: UInt8
    var rho2: UInt8
    var g: F4
    var g1: F2          # generator of G1 (order 2 h1); omega1 = g1^2
    var g2: F2
    var level1: RsDomain

    def __init__[p: Params](out self) raises:
        var gamma2 = f2_primitive()
        self.g1 = ext_pow[1](gamma2, F2_ORDER // (2 * p.h1()))
        self.g2 = ext_pow[1](gamma2, F2_ORDER // (2 * p.h2()))
        self.omega1 = ext_pow[1](self.g1, 2)
        self.omega2 = ext_pow[1](self.g2, 2)
        var r1 = ext_pow[1](self.omega1, 1 << p.a1)
        var r2 = ext_pow[1](self.omega2, 1 << p.a2)
        if r1[1] != 0 or r2[1] != 0:
            raise Error("rho_l not in F")
        self.rho1 = r1[0]
        self.rho2 = r2[0]
        self.level1 = RsDomain(p.L0, p.m_cosets)
        self.g = self.level1.g


def rs_factors(M: Int) -> Tuple[Int, Int, Int]:
    """The Good-Thomas radices of the odd part M | 315, in stage order: (5 or 1, 7 or 1, 9 or 3 or 1)."""
    var f9 = 9 if M % 9 == 0 else (3 if M % 3 == 0 else 1)
    return (5 if M % 5 == 0 else 1, 7 if M % 7 == 0 else 1, f9)


struct RsTables(TrivialRegisterPassable):
    """Twiddles of one RS domain L = m * 2^b * M for messages of length K (encode.rs_encode):
    ga (2^b, 4) gA^n; wr (r, r, 4) w_r^(t k) per radix; crt / ruri (M, 2) the Good-Thomas index maps
    lin -> i2 and lin -> t2 as u16; twist (m, K, 4) gamma4^(k i), the coset twist (absent at m = 1)."""
    var base: Int
    var ga: Int
    var w5: Int
    var w7: Int
    var w9: Int
    var crt: Int
    var ruri: Int
    var twist: Int
    var bytes: Int

    def __init__(out self, base: Int, L0: Int, m: Int, K: Int):
        var b = two_adic(L0)
        var M = L0 >> b
        var off = 0
        self.base = base
        self.ga = off; off += (1 << b) * 4
        self.w5 = off; off += 5 * 5 * 4
        self.w7 = off; off += 7 * 7 * 4
        self.w9 = off; off += 9 * 9 * 4
        self.crt = off; off += M * 2
        self.ruri = off; off += M * 2
        self.twist = off; off += (m * K * 4 if m > 1 else 0)
        self.bytes = off


def build_rs_tables(ctx: DeviceContext, rs: RsTables, dom: RsDomain, K: Int) raises -> HostBuffer[DType.uint8]:
    """The tables of `rs` for domain `dom`; upload at rs.base."""
    var h = ctx.enqueue_create_host_buffer[DType.uint8](rs.bytes)
    ctx.synchronize()
    _fill_rs(h, 0, rs, dom, K)
    return h^


def _fill_rs(h: HostBuffer[DType.uint8], at: Int, rs: RsTables, dom: RsDomain, K: Int) raises:
    var b = two_adic(dom.L0)
    var M = dom.L0 >> b
    var f5: Int
    var f7: Int
    var f9: Int
    f5, f7, f9 = rs_factors(M)
    if f5 * f7 * f9 != M:
        raise Error("odd part of L0 must divide 315")
    var gA = ext_pow[2](dom.g, M)
    var gB = ext_pow[2](dom.g, 1 << b)
    for n in range(1 << b):
        _put(h, at + rs.ga + n * 4, ext_pow[2](gA, n))
    for pair in [(f5, rs.w5), (f7, rs.w7), (f9, rs.w9)]:
        var r = pair[0]
        if r > 1:
            var w = ext_pow[2](gB, M // r)
            for tt in range(r):
                for k in range(r):
                    _put(h, at + pair[1] + (tt * r + k) * 4, ext_pow[2](w, (tt * k) % r))
    for lin in range(M):
        var d5 = lin // (f7 * f9)
        var d7 = (lin // f9) % f7
        var d9 = lin % f9
        var i2 = 0
        var t2 = 0
        for pair in [(f5, d5), (f7, d7), (f9, d9)]:
            var f = pair[0]
            if f > 1:
                var q = M // f
                i2 = (i2 + pair[1] * q * modinv(q % f, f)) % M
                t2 = (t2 + q * pair[1]) % M
        h[at + rs.crt + lin * 2] = UInt8(i2 & 255)
        h[at + rs.crt + lin * 2 + 1] = UInt8(i2 >> 8)
        h[at + rs.ruri + lin * 2] = UInt8(t2 & 255)
        h[at + rs.ruri + lin * 2 + 1] = UInt8(t2 >> 8)
    if dom.m > 1:
        for k in range(dom.m):
            var rep = dom.rep(k)
            var pw = F4(1, 0, 0, 0)
            for i in range(K):
                _put(h, at + rs.twist + (k * K + i) * 4, pw)
                pw = ext_mul[2](pw, rep)


struct TableLayout(TrivialRegisterPassable):
    """Byte offsets of every table inside the arena, relative to `base`."""
    var base: Int
    var winv1: Int      # (h1, h1, 2)   F2: h1^-1 * omega1^(-t k), row k, col t
    var winv2: Int      # (h2, h2, 2)
    var rho1: Int       # (m1)          F: rho1^y
    var rho2: Int       # (m2)
    var rs: RsTables    # level-1 RS domain, absolute offsets
    # residual grid G_l = <g_l>, point j = g_l^j; even j is H_l, odd j the coset (spec 8, 10.2)
    var g1p: Int        # (2 h1, 2)     g1^j
    var g2p: Int        # (2 h2, 2)
    var wfwd1: Int      # (2 h1, h1, 2) g1^(j k): coefficient k -> point j
    var wfwd2: Int      # (2 h2, h2, 2)
    var q1m: Int        # (h1, 2 h1, 2) Q1 on the coset from R on G1: 63 at j = 2t + 1, 64 * (1/h1) sum_k g1^((2t + 1 - 2s) k) at j = 2s
    var q2m: Int        # (h1, h1, 2)   63 * winv1: Q2 = S1 / (-2) on H1, coefficients from values
    var qinv1: Int      # (h1, h1, 2)   coset values t -> coefficient k: g1^-k h1^-1 omega1^(-t k)
    var ginv2: Int      # (2 h2, 2 h2, 2) G2 values j -> coefficient k: (2 h2)^-1 g2^(-j k)
    var qinv2: Int      # (h2, h2, 2)   coset values t -> coefficient k: g2^-k h2^-1 omega2^(-t k)
    var gate1: Int      # (2 h1, 2)     g1^j - e1, the chain gate (X1 - e1) on G1; e1 = omega1^-1
    var gate2: Int      # (2 h2, 2)     g2^j - e2
    var bytes: Int

    def __init__[p: Params](out self, base: Int):
        var off = 0
        self.base = base
        self.winv1 = off; off += p.h1() * p.h1() * 2
        self.winv2 = off; off += p.h2() * p.h2() * 2
        self.rho1 = off; off += p.m1
        self.rho2 = off; off += p.m2
        off = (off + 3) & ~3                       # the F4 and u16 tables want 4-byte alignment
        self.rs = RsTables(base + off, p.L0, p.m_cosets, p.N() // 4); off += self.rs.bytes
        self.g1p = off; off += 2 * p.h1() * 2
        self.g2p = off; off += 2 * p.h2() * 2
        self.wfwd1 = off; off += 2 * p.h1() * p.h1() * 2
        self.wfwd2 = off; off += 2 * p.h2() * p.h2() * 2
        self.q1m = off; off += p.h1() * 2 * p.h1() * 2
        self.q2m = off; off += p.h1() * p.h1() * 2
        self.qinv1 = off; off += p.h1() * p.h1() * 2
        self.ginv2 = off; off += 2 * p.h2() * 2 * p.h2() * 2
        self.qinv2 = off; off += p.h2() * p.h2() * 2
        self.gate1 = off; off += 2 * p.h1() * 2
        self.gate2 = off; off += 2 * p.h2() * 2
        self.bytes = off


def _get(h: HostBuffer[DType.uint8], off: Int) -> F2:
    return F2(h[off], h[off + 1])


def _put[w: SIMDLength](h: HostBuffer[DType.uint8], off: Int, v: SIMD[DType.uint8, w]):
    comptime for c in range(w):
        h[off + c] = v[c]


def build_tables[p: Params](ctx: DeviceContext, t: TableLayout, d: Domains) raises -> HostBuffer[DType.uint8]:
    """Fill a host buffer with every table; the caller uploads it to the arena at `t.base`."""
    var h = ctx.enqueue_create_host_buffer[DType.uint8](t.bytes)
    ctx.synchronize()

    var inv_h1 = f_inv(UInt8(p.h1() % 127))
    var inv_h2 = f_inv(UInt8(p.h2() % 127))
    var w1_inv = ext_pow[1](d.omega1, p.h1() - 1)      # omega1^-1
    var w2_inv = ext_pow[1](d.omega2, p.h2() - 1)
    for k in range(p.h1()):
        for tt in range(p.h1()):
            _put(h, t.winv1 + (k * p.h1() + tt) * 2, f_mul(ext_pow[1](w1_inv, (tt * k) % p.h1()), F2(inv_h1)))
    for k in range(p.h2()):
        for tt in range(p.h2()):
            _put(h, t.winv2 + (k * p.h2() + tt) * 2, f_mul(ext_pow[1](w2_inv, (tt * k) % p.h2()), F2(inv_h2)))
    for y in range(p.m1):
        h[t.rho1 + y] = f_pow(SIMD[DType.uint8, 1](d.rho1), y)[0]
    for y in range(p.m2):
        h[t.rho2 + y] = f_pow(SIMD[DType.uint8, 1](d.rho2), y)[0]

    _fill_rs(h, t.rs.base - t.base, t.rs, d.level1, p.N() // 4)

    _residual_tables[p](h, t, d)
    return h^


def _residual_tables[p: Params](h: HostBuffer[DType.uint8], t: TableLayout, d: Domains) raises:
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    var g1_inv = ext_pow[1](d.g1, 2 * h1 - 1)
    var g2_inv = ext_pow[1](d.g2, 2 * h2 - 1)
    var inv_h1 = F2(f_inv(UInt8(h1 % 127)))
    var inv_h2 = F2(f_inv(UInt8(h2 % 127)))
    var inv_2h2 = F2(f_inv(UInt8((2 * h2) % 127)))
    for j in range(2 * h1):
        _put(h, t.g1p + j * 2, ext_pow[1](d.g1, j))
        for k in range(h1):
            _put(h, t.wfwd1 + (j * h1 + k) * 2, ext_pow[1](d.g1, (j * k) % (2 * h1)))
    for j in range(2 * h2):
        _put(h, t.g2p + j * 2, ext_pow[1](d.g2, j))
        for k in range(h2):
            _put(h, t.wfwd2 + (j * h2 + k) * 2, ext_pow[1](d.g2, (j * k) % (2 * h2)))
        for k in range(2 * h2):
            _put(h, t.ginv2 + (j * 2 * h2 + k) * 2, f_mul(ext_pow[1](g2_inv, (j * k) % (2 * h2)), inv_2h2))
    var e1 = ext_pow[1](d.omega1, h1 - 1)
    var e2 = ext_pow[1](d.omega2, h2 - 1)
    for j in range(2 * h1):
        _put(h, t.gate1 + j * 2, f_sub(ext_pow[1](d.g1, j), e1))
    for j in range(2 * h2):
        _put(h, t.gate2 + j * 2, f_sub(ext_pow[1](d.g2, j), e2))
    for i in range(h1 * 2 * h1 * 2):
        h[t.q1m + i] = 0
    for tt in range(h1):
        # Q1(g1^(2t+1)) = (R - S1) / (-2), S1 the axis-1 interpolant of R on H1: one row over all of G1
        _put(h, t.q1m + (tt * 2 * h1 + 2 * tt + 1) * 2, F2(63, 0))
        for s in range(h1):
            var acc = F2(0)
            for k in range(h1):
                acc = f_add(acc, ext_pow[1](d.g1, ((2 * (tt - s) + 1) * k) % (2 * h1)))
            _put(h, t.q1m + (tt * 2 * h1 + 2 * s) * 2, f_mul(f_mul(acc, inv_h1), F2(64)))
        for k in range(h1):
            var wi = _get(h, t.winv1 + (k * h1 + tt) * 2)
            _put(h, t.q2m + (k * h1 + tt) * 2, f_mul(wi, F2(63)))
            var w = f_mul(ext_pow[1](g1_inv, k), inv_h1)                       # g1^-k / h1
            _put(h, t.qinv1 + (k * h1 + tt) * 2, ext_mul[1](w, ext_pow[1](g1_inv, (2 * tt * k) % (2 * h1))))
    for tt in range(h2):
        for k in range(h2):
            var w = f_mul(ext_pow[1](g2_inv, k), inv_h2)
            _put(h, t.qinv2 + (k * h2 + tt) * 2, ext_mul[1](w, ext_pow[1](g2_inv, (2 * tt * k) % (2 * h2))))
