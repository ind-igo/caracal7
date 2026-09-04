"""Domains and twiddle tables (docs/design.md section 5). Host side, run once at setup.

Generators (recorded in docs/decisions.md):
    gamma2  primitive element of F2*, order 16128 = 2^8 * 63; omega_l = gamma2^(16128 / h_l)
    rho_l   = omega_l^(2^a_l), the order-m_l generator of mu_l (lies in F)
    g       generator of the order-L0 subgroup of F4*: gamma4^((127^4 - 1) / L0)
    gA      = g^M (order 2^b), gB = g^(2^b) (order M), where L0 = 2^b * M, M odd

Leaf s of the level-1 code is the point g^s, s in [L0).
"""

from max.gpu.host import DeviceContext, HostBuffer

from caracal7.field import F2, F4, f_pow, f_mul, f_inv, ext_mul, ext_pow, ext_embed
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

    def __init__[p: Params](out self) raises:
        var gamma2 = f2_primitive()
        self.omega1 = ext_pow[1](gamma2, F2_ORDER // p.h1())
        self.omega2 = ext_pow[1](gamma2, F2_ORDER // p.h2())
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
    return h^
