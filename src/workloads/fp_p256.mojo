"""The secp256r1 field on four 64-bit limbs for the host: Montgomery products (R = 2^256; -p^-1 = 1 mod 2^64,
since p = -1 mod 2^64), values held in Montgomery form between `from_big` and `to_big`, Fermat inversion by
square-and-multiply on p - 2. The prime has no small fold, so no `fp_k1.mojo` reduction applies."""

from workloads.bigint import Big
from workloads.ecurve import Field

comptime L4 = SIMD[DType.uint64, 4]
comptime P256 = L4(0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF, 0x0000000000000000, 0xFFFFFFFF00000001)
comptime R2 = L4(0x0000000000000003, 0xFFFFFFFBFFFFFFFF, 0xFFFFFFFFFFFFFFFE, 0x00000004FFFFFFFD)   # R^2 mod p
comptime ONE = L4(0x0000000000000001, 0xFFFFFFFF00000000, 0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFE)  # R mod p
comptime PM2 = L4(0xFFFFFFFFFFFFFFFD, 0x00000000FFFFFFFF, 0x0000000000000000, 0xFFFFFFFF00000001)  # p - 2


@fieldwise_init
struct FpP256(Field):
    """A field element in Montgomery form (a R mod p) as little-endian limbs."""
    var l: L4

    @staticmethod
    def _limbs(b: Big) -> L4:
        """The magnitude's low 256 bits (callers pass canonical values)."""
        var l = L4(0)
        for i in range(min(len(b.mag), 8)):
            l[i // 2] |= UInt64(b.mag[i]) << UInt64(32 * (i % 2))
        return l

    @staticmethod
    def from_big(b: Big) -> FpP256:
        return FpP256(FpP256._mont(FpP256._limbs(b), R2))

    def to_big(self) -> Big:
        var l = FpP256._mont(self.l, L4(1, 0, 0, 0))
        var mag = List[UInt32](capacity=8)
        for i in range(8):
            mag.append(UInt32((l[i // 2] >> UInt64(32 * (i % 2))) & 0xFFFFFFFF))
        return Big(False, mag^)

    def __eq__(self, other: FpP256) -> Bool:
        return self.l == other.l

    def __ne__(self, other: FpP256) -> Bool:
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

    def __add__(self, o: FpP256) -> FpP256:
        var s = FpP256._add(self.l, o.l)
        if s[1] != 0 or FpP256._ge(s[0], P256):
            return FpP256(FpP256._sub(s[0], P256)[0])
        return FpP256(s[0])

    def __sub__(self, o: FpP256) -> FpP256:
        var d = FpP256._sub(self.l, o.l)
        if d[1] != 0:
            return FpP256(FpP256._add(d[0], P256)[0])
        return FpP256(d[0])

    @staticmethod
    def _mont(a: L4, b: L4) -> L4:
        """a b R^-1 mod p for a, b below p (CIOS; the Montgomery factor m is t[0] since -p^-1 = 1)."""
        var t = SIMD[DType.uint64, 8](0)      # t[0..5] live; lanes 6, 7 stay zero
        comptime for i in range(4):
            var carry: UInt128 = 0
            comptime for j in range(4):
                var w = UInt128(a[i]) * UInt128(b[j]) + UInt128(t[j]) + carry
                t[j] = w.cast[DType.uint64]()
                carry = w >> 64
            var w4 = UInt128(t[4]) + carry
            t[4] = w4.cast[DType.uint64]()
            t[5] = (w4 >> 64).cast[DType.uint64]()
            var m = t[0]
            carry = 0
            comptime for j in range(4):
                var w2 = UInt128(m) * UInt128(P256[j]) + UInt128(t[j]) + carry
                t[j] = w2.cast[DType.uint64]()
                carry = w2 >> 64
            var w5 = UInt128(t[4]) + carry
            t[4] = w5.cast[DType.uint64]()
            t[5] += (w5 >> 64).cast[DType.uint64]()
            comptime for j in range(5):
                t[j] = t[j + 1]
            t[5] = 0
        var r = L4(t[0], t[1], t[2], t[3])
        if t[4] != 0 or FpP256._ge(r, P256):
            r = FpP256._sub(r, P256)[0]
        return r

    def __mul__(self, o: FpP256) -> FpP256:
        return FpP256(FpP256._mont(self.l, o.l))

    def inv(self) -> FpP256:
        """a^(p - 2) by square-and-multiply over the bits of p - 2. Zero maps to zero."""
        var r = FpP256(ONE)
        for i in range(255, -1, -1):
            r = r * r
            if (PM2[i // 64] >> UInt64(i % 64)) & 1 == 1:
                r = r * self
        return r
