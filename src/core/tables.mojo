"""Domains and twiddle tables (docs/design.md section 5). Host side, run once at setup.

Generators:
    gamma2  primitive element of F2*, order 16128 = 2^8 * 63; omega_l = gamma2^(16128 / h_l)
    rho_l   = omega_l^(2^a_l), the order-m_l generator of mu_l (lies in F)
    g       generator of the order-L0 subgroup of F4*: gamma4^((127^4 - 1) / L0)
    gA      = g^M (order 2^b), gB = g^(2^b) (order M), where L0 = 2^b * M, M odd
    gamma4  a primitive element of F4*; coset k of an RS domain is gamma4^k D0

Leaf s of an RS domain with m cosets is the point gamma4^(s // L0) g^(s mod L0), s in [m L0).
"""

from max.gpu.host import DeviceContext, HostBuffer

from core.field import F2, F4, f_add, f_sub, f_pow, f_mul, f_inv, ext_mul, ext_pow, ext_embed
from core.params import Params
from core.dft import DftPlan
from core.backend import frag8

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


@always_inline
def node_id(kappa: F4, s: Int, x: F2) -> F4:
    """The wiring id of slot s at the chain point x of H2: kappa^s x in F4 (accumulate.mojo "Wiring"). kappa
    generates F4*, so the cosets kappa^s H2 are distinct for s below F4_ORDER / h2: the slot count is not a
    constraint of the grid (it was, with ids in F2*: six cosets at h2 = 2688)."""
    return ext_mul[2](ext_pow[2](kappa, s), ext_embed[2](x))


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
    for n in range(0, 1 << b, 1 << max(b - 6, 0)):      # the 2-adic stages read gA^(2^(b - 6) x): in F2
        if h[at + rs.ga + n * 4 + 2] != 0 or h[at + rs.ga + n * 4 + 3] != 0:
            raise Error("2-adic stage twiddle outside F2")
    for pair in [(f5, rs.w5), (f7, rs.w7), (f9, rs.w9)]:
        var r = pair[0]
        if r > 1:
            var w = ext_pow[2](gB, M // r)
            if r != 5 and (w[1] != 0 or w[2] != 0 or w[3] != 0):      # orders 3, 7, 9 divide 126: the stages read one byte
                raise Error("odd radix twiddle outside F")
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
    var rho1: Int       # (m1)          F: rho1^y
    var rho2: Int       # (m2)
    var rho1t: Int      # (m1, m1, 2)   F2: the radix table of the to_stored digit pass (encode.mojo): rho1^(r y), or the compact odd block at m1 <= 9
    var rho2t: Int      # DftPlan(m2, m2) stage tables of rho2^(r y), the chain digit's pass (split past 9)
    var rs: RsTables    # level-1 RS domain, absolute offsets
    # residual grid G_l = <g_l>, point j = g_l^j; even j is H_l, odd j the coset (spec 8, 10.2)
    var g1p: Int        # (2 h1, 2)     g1^j
    var g2p: Int        # (2 h2, 2)
    var q1m: Int        # (h1, 2 h1, 2) the axis-1 coefficients k of Q1 from R on G1: qinv1 (coset values t -> coefficients,
                        #               g1^-k h1^-1 omega1^(-t k)) composed with the coset map (63 at j = 2t + 1,
                        #               64 * (1/h1) sum_k g1^((2t + 1 - 2s) k) at j = 2s); one GEMM instead of two
    var q2m: Int        # (h1, h1, 2)   63 * winv1: Q2 = S1 / (-2) on H1, coefficients from values
    var gate1: Int      # (2 h1, 2)     g1^j - e1, the chain gate (X1 - e1) on G1; e1 = omega1^-1
    var gate2: Int      # (2 h2, 2)     g2^j - e2
    var c2p: Int        # (2 h2, 2)     the coset points c_t = gamma2 g2^t of the small grid (smallgrid.mojo)
    var fwd1e: Int      # DftPlan(h1, h1) stage tables of the axis-1 forward DFT onto H1 (residual.lde)
    var fwd1o: Int      # the same onto the coset g1 H1, the twist g1^k inside
    var inv1: Int       # DftPlan(h1, h1) stage tables of the axis-1 inverse DFT, h1^-1 folded in
    var fwd2e: Int      # DftPlan(h2, h2) coefficients -> H2 on axis 2 (residual.lde)
    var fwd2o: Int      # the same onto the coset g2 H2, the twist g2^k inside
    var inv2: Int
    var ginv2p: Int     # DftPlan(2 h2, 2 h2) of the G2 inverse (residual.quotient step 3), (2 h2)^-1 folded in
    var qinv2p: Int     # DftPlan(h2, h2) of quotient step 5: coset values t -> coefficient k, g2^-k h2^-1 omega2^(-t k)
    var gfwd2p: Int     # DftPlan(2 h2, 2 h2) coefficients -> values on G2 (smallgrid.mojo)
    var cfwd2p: Int     # DftPlan(2 h2, h2) coefficient k -> value at the small grid's coset point c_t = gamma2 g2^t: gamma2^k g2^(t k)
    var cinv2p: Int     # DftPlan(2 h2, 2 h2) coset values t -> coefficient k: (2 h2)^-1 gamma2^-k g2^(-t k)
    var bytes: Int

    def __init__[p: Params](out self, base: Int):
        var off = 0
        self.base = base
        self.winv1 = off; off += p.h1() * p.h1() * 2
        self.rho1 = off; off += p.m1
        self.rho2 = off; off += p.m2
        self.rho1t = off; off += p.m1 * p.m1 * 2
        self.rho2t = off; off += DftPlan(p.m2, p.m2).bytes()
        off = (off + 3) & ~3                       # the F4 and u16 tables want 4-byte alignment
        self.rs = RsTables(base + off, p.L0, p.m_cosets, p.K()); off += self.rs.bytes
        self.g1p = off; off += 2 * p.h1() * 2
        self.g2p = off; off += 2 * p.h2() * 2
        self.q1m = off; off += p.h1() * 2 * p.h1() * 2
        self.q2m = off; off += p.h1() * p.h1() * 2
        self.gate1 = off; off += 2 * p.h1() * 2
        self.gate2 = off; off += 2 * p.h2() * 2
        self.c2p = off; off += 2 * p.h2() * 2
        self.fwd1e = off; off += DftPlan(p.h1(), p.h1()).bytes()
        self.fwd1o = off; off += DftPlan(p.h1(), p.h1()).bytes()
        self.inv1 = off; off += DftPlan(p.h1(), p.h1()).bytes()
        self.fwd2e = off; off += DftPlan(p.h2(), p.h2()).bytes()
        self.fwd2o = off; off += DftPlan(p.h2(), p.h2()).bytes()
        self.inv2 = off; off += DftPlan(p.h2(), p.h2()).bytes()
        self.ginv2p = off; off += DftPlan(2 * p.h2(), 2 * p.h2()).bytes()
        self.qinv2p = off; off += DftPlan(p.h2(), p.h2()).bytes()
        self.gfwd2p = off; off += DftPlan(2 * p.h2(), 2 * p.h2()).bytes()
        self.cfwd2p = off; off += DftPlan(2 * p.h2(), p.h2()).bytes()
        self.cinv2p = off; off += DftPlan(2 * p.h2(), 2 * p.h2()).bytes()
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
    for y in range(p.m1):
        h[t.rho1 + y] = f_pow(SIMD[DType.uint8, 1](d.rho1), y)[0]
    for y in range(p.m2):
        h[t.rho2 + y] = f_pow(SIMD[DType.uint8, 1](d.rho2), y)[0]
    if p.m1 > 1 and p.m1 <= 9:                        # the compact odd block of dft.k_radix_odd: no twists, w = rho1
        for y in range(p.m1):
            _put(h, t.rho1t + y * 2, F2(1, 0))
            _put(h, t.rho1t + (p.m1 + y) * 2, F2(1, 0))
            _put(h, t.rho1t + (2 * p.m1 + y) * 2, F2(f_pow(SIMD[DType.uint8, 1](d.rho1), y)[0], 0))
    else:                                            # dense (m1, m1), the lane kernel k_radix
        for r in range(p.m1):
            for y in range(p.m1):
                _put(h, t.rho1t + (r * p.m1 + y) * 2, F2(f_pow(SIMD[DType.uint8, 1](d.rho1), (r * y) % p.m1)[0], 0))
    _dft_tables(h, t.rho2t, DftPlan(p.m2, p.m2), F2(d.rho2, 0), 1)

    _fill_rs(h, t.rs.base - t.base, t.rs, d.level1, p.K())

    _residual_tables[p](h, t, d)
    _dft_tables(h, t.fwd1e, DftPlan(p.h1(), p.h1()), d.omega1, 1)
    _dft_tables(h, t.fwd1o, DftPlan(p.h1(), p.h1()), d.omega1, 1, twist_in=d.g1)
    _dft_tables(h, t.inv1, DftPlan(p.h1(), p.h1()), w1_inv, inv_h1)
    _dft_tables(h, t.fwd2e, DftPlan(p.h2(), p.h2()), d.omega2, 1)
    _dft_tables(h, t.fwd2o, DftPlan(p.h2(), p.h2()), d.omega2, 1, twist_in=d.g2)
    _dft_tables(h, t.inv2, DftPlan(p.h2(), p.h2()), w2_inv, inv_h2)
    _dft_tables(h, t.ginv2p, DftPlan(2 * p.h2(), 2 * p.h2()), ext_pow[1](d.g2, 2 * p.h2() - 1), f_inv(UInt8((2 * p.h2()) % 127)))
    return h^


def _dft_tables(h: HostBuffer[DType.uint8], at: Int, plan: DftPlan, root: F2, scale: UInt8,
                twist_in: F2 = F2(1, 0), twist_out: F2 = F2(1, 0)) raises:
    """The stage tables of dft.mojo for `root` of order plan.n, `scale` folded into stage 3. At n1 = 1 the
    stage-1 table is all ones (twist_out folded into stage 2), so dft_axis may skip that stage.
    The transform twist_out^j sum_k root^(j k) twist_in^k x[k] (a coset evaluation or its inverse) splits
    over the digits: twist_in^(n1 n2 k3) into T3, twist_in^(n1 k2) into T2, twist_in^k1 twist_out^j into T1,
    or with the odd split twist_in^(na kb) into Tb and twist_in^ka twist_out^j into Ta."""
    var n = plan.n
    var n1 = plan.n1
    var n2 = plan.n2
    var n3 = plan.n3
    var na = plan.na
    var nb = plan.nb
    for j3 in range(n3):
        for k3 in range(plan.k3):
            var w = f_mul(ext_pow[1](root, (n1 * n2 * j3 * k3) % n), F2(scale))
            _put(h, at + plan.t3() + (j3 * plan.k3 + k3) * 2, ext_mul[1](w, ext_pow[1](twist_in, n1 * n2 * k3)))
        for j2 in range(n2):
            for k2 in range(n2):
                var w = ext_pow[1](root, (n1 * (j3 + n3 * j2) * k2) % n)
                if n1 == 1:                                  # stage 1 would only apply twist_out^j: fold it here
                    w = ext_mul[1](w, ext_pow[1](twist_out, j3 + n3 * j2))
                _put(h, at + plan.t2() + ((j3 * n2 + j2) * n2 + k2) * 2, ext_mul[1](w, ext_pow[1](twist_in, n1 * k2)))
    # the odd stages (nb, then na when split) read compact blocks in the r x r slots (dft.k_radix_odd):
    # [ct: r column twists][rt: r row twists][w: the r powers of the order-r root], so that
    # T[j][k] = rt(j) w^(j k) ct(k) with ct(kb) = root^(na jj kb) twist_in^(na kb), w = root^(na n2 n3)
    if (nb > 1 and (nb > 9 or nb == 5)) or na > 9 or na == 5:
        raise Error("odd radix outside 3, 7, 9: the compact block needs a root in F (dft.k_radix_odd)")
    if plan.has_dense():
        # the dense (k_in, n) table of dft.k_dft8, every twist folded, in fragment order: 8x8 fragment
        # (kq, jq) is 32 lanes of 4 bytes, lane (fr, fc) = frag8 holding T[kq 8 + fr, jq 8 + fc + (0, 1)],
        # so a simdgroup's fragment load is one contiguous 128-byte line
        for kq in range(plan.k_in // 8):
            for jq in range(n // 8):
                for lane in range(32):
                    var rc = frag8(lane)
                    var k = kq * 8 + rc[0]
                    for el in range(2):
                        var j = jq * 8 + rc[1] + el
                        var w = f_mul(ext_pow[1](root, (j * k) % n), F2(scale))
                        w = ext_mul[1](w, ext_mul[1](ext_pow[1](twist_in, k), ext_pow[1](twist_out, j)))
                        _put(h, at + plan.dense() + ((kq * (n // 8) + jq) * 32 + lane) * 4 + el * 2, w)
    for jj in range(n2 * n3):
        var blk = at + plan.t1() + jj * nb * nb * 2
        if nb == 1:                                      # n1 = 1: the stage is skipped, the slot unused
            _put(h, blk, F2(1, 0))
            continue
        var wb = ext_pow[1](root, na * n2 * n3)
        for kb in range(nb):
            _put(h, blk + kb * 2, ext_mul[1](ext_pow[1](root, (na * jj * kb) % n), ext_pow[1](twist_in, na * kb)))
        for jb in range(nb):
            _put(h, blk + (nb + jb) * 2, ext_pow[1](twist_out, jj + n2 * n3 * jb) if na == 1 and n1 > 1 else F2(1, 0))
        for m in range(nb):
            _put(h, blk + (2 * nb + m) * 2, ext_pow[1](wb, m))
        if na > 1:
            var wa = ext_pow[1](root, n2 * n3 * nb)
            for jb in range(nb):
                var blka = at + plan.ta() + (jb + nb * jj) * na * na * 2
                var jp = jj + n2 * n3 * jb
                for ka in range(na):
                    _put(h, blka + ka * 2, ext_mul[1](ext_pow[1](root, (jp * ka) % n), ext_pow[1](twist_in, ka)))
                for ja in range(na):
                    _put(h, blka + (na + ja) * 2, ext_pow[1](twist_out, jj + n2 * n3 * (jb + nb * ja)))
                for m in range(na):
                    _put(h, blka + (2 * na + m) * 2, ext_pow[1](wa, m))

def _residual_tables[p: Params](h: HostBuffer[DType.uint8], t: TableLayout, d: Domains) raises:
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    var g1_inv = ext_pow[1](d.g1, 2 * h1 - 1)
    var g2_inv = ext_pow[1](d.g2, 2 * h2 - 1)
    var inv_h1 = F2(f_inv(UInt8(h1 % 127)))
    for j in range(2 * h1):
        _put(h, t.g1p + j * 2, ext_pow[1](d.g1, j))
    for j in range(2 * h2):
        _put(h, t.g2p + j * 2, ext_pow[1](d.g2, j))
    var e1 = ext_pow[1](d.omega1, h1 - 1)
    var e2 = ext_pow[1](d.omega2, h2 - 1)
    for j in range(2 * h1):
        _put(h, t.gate1 + j * 2, f_sub(ext_pow[1](d.g1, j), e1))
    for j in range(2 * h2):
        _put(h, t.gate2 + j * 2, f_sub(ext_pow[1](d.g2, j), e2))
    # Q1(g1^(2t+1)) = (R - S1) / (-2), S1 the axis-1 interpolant of R on H1: one row over all of G1.
    # The interpolant weight sum_k g1^((2 (t - s) + 1) k) depends only on (t - s) mod h1: one row of h1 sums.
    # The stored table is the composition with the coset inverse DFT qinv1 (row k, col t), so the stage
    # goes from R on G1 to the axis-1 coefficients of Q1 in one GEMM.
    var q1row = List[F2](capacity=h1)
    for diff in range(h1):
        var acc = F2(0)
        for k in range(h1):
            acc = f_add(acc, ext_pow[1](d.g1, ((2 * diff + 1) * k) % (2 * h1)))
        q1row.append(f_mul(f_mul(acc, inv_h1), F2(64)))
    # q1m[k][j] = sum_t qinv1[k][t] cos[t][j] with qinv1[k][t] = g1^-k / h1 * omega1^(-t k) and cos[t][2t+1] = 63,
    # cos[t][2s] = q1row[(t - s) mod h1]: the even columns are a circular convolution, omega1^(-s k) qhat[k]
    # with qhat[k] = sum_d omega1^(-d k) q1row[d]; O(h1^2), not the dense triple product
    var qhat = List[F2](capacity=h1)
    for k in range(h1):
        var acc = F2(0)
        for dd in range(h1):
            acc = f_add(acc, ext_mul[1](ext_pow[1](g1_inv, (2 * dd * k) % (2 * h1)), q1row[dd]))
        qhat.append(acc)
    for k in range(h1):
        var wk = f_mul(ext_pow[1](g1_inv, k), inv_h1)                          # g1^-k / h1
        for s in range(h1):
            var om = ext_mul[1](wk, ext_pow[1](g1_inv, (2 * s * k) % (2 * h1)))  # g1^-k / h1 * omega1^(-s k)
            _put(h, t.q1m + (k * 2 * h1 + 2 * s) * 2, ext_mul[1](om, qhat[k]))
            _put(h, t.q1m + (k * 2 * h1 + 2 * s + 1) * 2, f_mul(om, F2(63)))
        for tt in range(h1):
            var wi = _get(h, t.winv1 + (k * h1 + tt) * 2)
            _put(h, t.q2m + (k * h1 + tt) * 2, f_mul(wi, F2(63)))
    # the small grid's coset gamma2 G2 (gamma2 is in no proper subgroup, so c^h2 - 1 vanishes nowhere on it).
    # At 2 h2 = F2_ORDER, G2 is all of F2* and there is no coset: its tables stay unset and the prover
    # refuses a statement with accumulators (prover.mojo)
    var gam = f2_primitive()
    _dft_tables(h, t.qinv2p, DftPlan(h2, h2), ext_pow[1](d.omega2, h2 - 1), f_inv(UInt8(h2 % 127)), twist_out=g2_inv)
    _dft_tables(h, t.gfwd2p, DftPlan(2 * h2, 2 * h2), d.g2, 1)
    if _is_one[1](ext_pow[1](gam, 2 * h2)):
        return
    var gam_inv = ext_pow[1](gam, F2_ORDER - 1)
    for tt in range(2 * h2):
        _put(h, t.c2p + tt * 2, ext_mul[1](gam, ext_pow[1](d.g2, tt)))
    # the coset plans: before 2026-09-11 these were dense h2^2 tables, 2 GB and 255 s at h2 = 8064
    _dft_tables(h, t.cfwd2p, DftPlan(2 * h2, h2), d.g2, 1, twist_in=gam)
    _dft_tables(h, t.cinv2p, DftPlan(2 * h2, 2 * h2), g2_inv, f_inv(UInt8((2 * h2) % 127)), twist_out=gam_inv)
