"""secp256k1's field on four 64-bit limbs for the host: schoolbook products through UInt128, the reduction
by 2^256 = 2^32 + 977 (two folds, one subtraction), Fermat inversion on an addition chain. `Big` stays the type the workloads
speak; `Curve` converts at its field operations."""

from caracal7.workloads.bigint import Big

comptime C: UInt64 = (1 << 32) + 977     # 2^256 mod p
comptime L4 = SIMD[DType.uint64, 4]


@fieldwise_init
struct Fp(Copyable, Movable, ImplicitlyCopyable, Equatable):
    """A field element below p as little-endian limbs."""
    var l: L4

    @staticmethod
    def p() -> L4:
        return L4(0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)

    @staticmethod
    def from_big(b: Big) -> Fp:
        """The magnitude's low 256 bits (callers pass canonical values)."""
        var l = L4(0)
        for i in range(min(len(b.mag), 8)):
            l[i // 2] |= UInt64(b.mag[i]) << UInt64(32 * (i % 2))
        return Fp(l)

    def to_big(self) -> Big:
        var mag = List[UInt32](capacity=8)
        for i in range(8):
            mag.append(UInt32((self.l[i // 2] >> UInt64(32 * (i % 2))) & 0xFFFFFFFF))
        return Big(False, mag^)

    def __eq__(self, other: Fp) -> Bool:
        return self.l == other.l

    def __ne__(self, other: Fp) -> Bool:
        return not (self == other)

    @staticmethod
    def _ge(a: L4, b: L4) -> Bool:
        for i in range(3, -1, -1):
            if a[i] != b[i]:
                return a[i] > b[i]
        return True

    @staticmethod
    def _sub(a: L4, b: L4) -> Tuple[L4, UInt64]:
        """a - b mod 2^256 and the borrow."""
        var r = L4(0)
        var borrow: UInt64 = 0
        for i in range(4):
            var d = a[i] - b[i] - borrow
            borrow = UInt64(1) if (a[i] < b[i]) or (a[i] == b[i] and borrow == 1) else UInt64(0)
            r[i] = d
        return (r, borrow)

    @staticmethod
    def _add(a: L4, b: L4) -> Tuple[L4, UInt64]:
        var r = L4(0)
        var carry: UInt128 = 0
        for i in range(4):
            var w = UInt128(a[i]) + UInt128(b[i]) + carry
            r[i] = w.cast[DType.uint64]()
            carry = w >> 64
        return (r, carry.cast[DType.uint64]())

    def __add__(self, o: Fp) -> Fp:
        var s = Fp._add(self.l, o.l)
        if s[1] != 0 or Fp._ge(s[0], Fp.p()):
            return Fp(Fp._sub(s[0], Fp.p())[0])
        return Fp(s[0])

    def __sub__(self, o: Fp) -> Fp:
        var d = Fp._sub(self.l, o.l)
        if d[1] != 0:
            return Fp(Fp._add(d[0], Fp.p())[0])
        return Fp(d[0])

    def __mul__(self, o: Fp) -> Fp:
        var r = SIMD[DType.uint64, 8](0)
        comptime for i in range(4):
            var carry: UInt128 = 0
            comptime for j in range(4):
                var w = UInt128(self.l[i]) * UInt128(o.l[j]) + UInt128(r[i + j]) + carry
                r[i + j] = w.cast[DType.uint64]()
                carry = w >> 64
            r[i + 4] = carry.cast[DType.uint64]()
        # lo + hi C: five limbs, the fifth below 2^35
        var s = L4(0)
        var carry: UInt128 = 0
        comptime for i in range(4):
            var w = UInt128(r[4 + i]) * UInt128(C) + UInt128(r[i]) + carry
            s[i] = w.cast[DType.uint64]()
            carry = w >> 64
        # the fifth limb folds again: below 2^70 into the low limbs, a carry out only for s near 2^256
        var w = carry * UInt128(C) + UInt128(s[0])
        s[0] = w.cast[DType.uint64]()
        carry = w >> 64
        comptime for i in range(1, 4):
            var w2 = UInt128(s[i]) + carry
            s[i] = w2.cast[DType.uint64]()
            carry = w2 >> 64
        if carry != 0:
            s = Fp._add(s, L4(C, 0, 0, 0))[0]
        if Fp._ge(s, Fp.p()):
            s = Fp._sub(s, Fp.p())[0]
        return Fp(s)

    def sqn(self, n: Int) -> Fp:
        """self^(2^n)."""
        var r = self
        for _ in range(n):
            r = r * r
        return r

    def pow(self, e: L4) -> Fp:
        var r = Fp(L4(1, 0, 0, 0))
        for i in range(255, -1, -1):
            r = r * r
            if (e[i // 64] >> UInt64(i % 64)) & 1 == 1:
                r = r * self
        return r

    def inv(self) -> Fp:
        """a^(p - 2) by the addition chain of libsecp256k1 (255 squarings, 15 products): p - 2 is
        [223 x 1][0][22 x 1][0000 1][011][01], the runs of ones from 2^n - 1 at n = 2, 3, 22, 223.
        Zero maps to zero."""
        var x2 = self.sqn(1) * self
        var x3 = x2.sqn(1) * self
        var x6 = x3.sqn(3) * x3
        var x9 = x6.sqn(3) * x3
        var x11 = x9.sqn(2) * x2
        var x22 = x11.sqn(11) * x11
        var x44 = x22.sqn(22) * x22
        var x88 = x44.sqn(44) * x44
        var x176 = x88.sqn(88) * x88
        var x220 = x176.sqn(44) * x44
        var x223 = x220.sqn(3) * x3
        var t = x223.sqn(23) * x22
        t = t.sqn(5) * self
        t = t.sqn(3) * x2
        return t.sqn(2) * self
