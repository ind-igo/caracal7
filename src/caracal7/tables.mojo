"""Domains and twiddle tables (docs/design.md section 5). Host side, run once at setup.

Generators (recorded in docs/decisions.md):
    gamma2  primitive element of F2*, order 16128 = 2^8 * 63; omega_l = gamma2^(16128 / h_l)
    rho_l   = omega_l^(2^a_l), the order-m_l generator of mu_l (lies in F)
    g       generator of the order-L0 subgroup of F4*: gamma4^((127^4 - 1) / L0)
    gA      = g^M (order 2^b), gB = g^(2^b) (order M), where L0 = 2^b * M, M odd

Leaf s of the level-1 code is the point g^s, s in [L0).
"""

from max.gpu.host import DeviceContext, HostBuffer

from caracal7.field import F2, F4, f_add, f_pow, f_mul, f_inv, ext_mul, ext_pow, ext_embed
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


struct Domains(TrivialRegisterPassable):
    var omega1: F2
    var omega2: F2
    var rho1: UInt8
    var rho2: UInt8
    var g: F4
    var g1: F2          # generator of G1 (order 2 h1); omega1 = g1^2
    var g2: F2

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
        self.g = f4_subgroup_generator(p.L0)


struct TableLayout(TrivialRegisterPassable):
    """Byte offsets of every table inside the arena, relative to `base`."""
    var base: Int
    var winv1: Int      # (h1, h1, 2)   F2: h1^-1 * omega1^(-t k), row k, col t
    var winv2: Int      # (h2, h2, 2)
    var rho1: Int       # (m1)          F: rho1^y
    var rho2: Int       # (m2)
    var ga: Int         # (2^b, 4)      F4: gA^n
    var w5: Int         # (5, 5, 4)     F4: w5^(t k)
    var w7: Int         # (7, 7, 4)
    var w9: Int         # (9, 9, 4)
    var crt: Int        # (315, 2)     lin -> i2 = crt(d5, d7, d9), little-endian u16
    var ruri: Int       # (315, 2)     lin -> t2 = (63 t5 + 45 t7 + 35 t9) mod 315
    # residual grid G_l = <g_l>, point j = g_l^j; even j is H_l, odd j the coset (spec 8, 10.2)
    var g1p: Int        # (2 h1, 2)     g1^j
    var g2p: Int        # (2 h2, 2)
    var wfwd1: Int      # (2 h1, h1, 2) g1^(j k): coefficient k -> point j
    var wfwd2: Int      # (2 h2, h2, 2)
    var s1ext: Int      # (h1, h1, 2)   S1 on the coset from S1 on H1: (1/h1) sum_k g1^((2t + 1 - 2s) k)
    var qinv1: Int      # (h1, h1, 2)   coset values t -> coefficient k: g1^-k h1^-1 omega1^(-t k)
    var ginv2: Int      # (2 h2, 2 h2, 2) G2 values j -> coefficient k: (2 h2)^-1 g2^(-j k)
    var qinv2: Int      # (h2, h2, 2)   coset values t -> coefficient k: g2^-k h2^-1 omega2^(-t k)
    var bytes: Int

    def __init__[p: Params](out self, base: Int):
        comptime b = two_adic(p.L0)
        var off = 0
        self.base = base
        self.winv1 = off; off += p.h1() * p.h1() * 2
        self.winv2 = off; off += p.h2() * p.h2() * 2
        self.rho1 = off; off += p.m1
        self.rho2 = off; off += p.m2
        self.ga = off; off += (1 << b) * 4
        self.w5 = off; off += 5 * 5 * 4
        self.w7 = off; off += 7 * 7 * 4
        self.w9 = off; off += 9 * 9 * 4
        self.crt = off; off += 315 * 2
        self.ruri = off; off += 315 * 2
        self.g1p = off; off += 2 * p.h1() * 2
        self.g2p = off; off += 2 * p.h2() * 2
        self.wfwd1 = off; off += 2 * p.h1() * p.h1() * 2
        self.wfwd2 = off; off += 2 * p.h2() * p.h2() * 2
        self.s1ext = off; off += p.h1() * p.h1() * 2
        self.qinv1 = off; off += p.h1() * p.h1() * 2
        self.ginv2 = off; off += 2 * p.h2() * 2 * p.h2() * 2
        self.qinv2 = off; off += p.h2() * p.h2() * 2
        self.bytes = off


def _put[w: SIMDLength](h: HostBuffer[DType.uint8], off: Int, v: SIMD[DType.uint8, w]):
    comptime for c in range(w):
        h[off + c] = v[c]


def build_tables[p: Params](ctx: DeviceContext, t: TableLayout, d: Domains) raises -> HostBuffer[DType.uint8]:
    """Fill a host buffer with every table; the caller uploads it to the arena at `t.base`."""
    comptime b = two_adic(p.L0)
    comptime M = p.L0 >> b
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

    var gA = ext_pow[2](d.g, M)
    var gB = ext_pow[2](d.g, 1 << b)
    for n in range(1 << b):
        _put(h, t.ga + n * 4, ext_pow[2](gA, n))
    var w5 = ext_pow[2](gB, M // 5)
    var w7 = ext_pow[2](gB, M // 7)
    var w9 = ext_pow[2](gB, M // 9)
    for tt in range(5):
        for k in range(5):
            _put(h, t.w5 + (tt * 5 + k) * 4, ext_pow[2](w5, (tt * k) % 5))
    for tt in range(7):
        for k in range(7):
            _put(h, t.w7 + (tt * 7 + k) * 4, ext_pow[2](w7, (tt * k) % 7))
    for tt in range(9):
        for k in range(9):
            _put(h, t.w9 + (tt * 9 + k) * 4, ext_pow[2](w9, (tt * k) % 9))
    var e5 = modinv(63, 5)
    var e7 = modinv(45, 7)
    var e9 = modinv(35, 9)
    for lin in range(315):
        var d5 = lin // 63
        var d7 = (lin // 9) % 7
        var d9 = lin % 9
        var i2 = (d5 * 63 * e5 + d7 * 45 * e7 + d9 * 35 * e9) % 315
        var t2 = (63 * d5 + 45 * d7 + 35 * d9) % 315
        h[t.crt + lin * 2] = UInt8(i2 & 255)
        h[t.crt + lin * 2 + 1] = UInt8(i2 >> 8)
        h[t.ruri + lin * 2] = UInt8(t2 & 255)
        h[t.ruri + lin * 2 + 1] = UInt8(t2 >> 8)

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
    for tt in range(h1):
        for s in range(h1):
            var acc = F2(0)
            for k in range(h1):
                acc = f_add(acc, ext_pow[1](d.g1, ((2 * (tt - s) + 1) * k) % (2 * h1)))
            _put(h, t.s1ext + (tt * h1 + s) * 2, f_mul(acc, inv_h1))
        for k in range(h1):
            var w = f_mul(ext_pow[1](g1_inv, k), inv_h1)                       # g1^-k / h1
            _put(h, t.qinv1 + (k * h1 + tt) * 2, ext_mul[1](w, ext_pow[1](g1_inv, (2 * tt * k) % (2 * h1))))
    for tt in range(h2):
        for k in range(h2):
            var w = f_mul(ext_pow[1](g2_inv, k), inv_h2)
            _put(h, t.qinv2 + (k * h2 + tt) * 2, ext_mul[1](w, ext_pow[1](g2_inv, (2 * tt * k) % (2 * h2))))
