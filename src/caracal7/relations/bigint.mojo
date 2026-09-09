"""Signed big integers for the host side of the arithmetic relations: 32-bit limbs, schoolbook products,
binary long division. Sizes are a few hundred bits, so nothing here is optimized."""


struct Big(Copyable, Movable, Equatable, Writable):
    """Sign and magnitude; the magnitude is little-endian limbs with no leading zero limb (zero is no limbs)."""
    var neg: Bool
    var mag: List[UInt32]

    def __init__(out self):
        self.neg = False
        self.mag = List[UInt32]()

    def __init__(out self, v: Int):
        self.neg = v < 0
        self.mag = List[UInt32]()
        var m = -v if v < 0 else v
        while m > 0:
            self.mag.append(UInt32(m & 0xFFFFFFFF))
            m >>= 32

    def __init__(out self, neg: Bool, var mag: List[UInt32]):
        self.neg = neg
        self.mag = mag^
        self._norm()

    @staticmethod
    def from_bits(bits: List[Int]) -> Big:
        """Bit 0 first."""
        var mag = List[UInt32](length=(len(bits) + 31) // 32, fill=0)
        for i in range(len(bits)):
            if bits[i] != 0:
                mag[i // 32] |= UInt32(1) << UInt32(i % 32)
        return Big(False, mag^)

    @staticmethod
    def from_bytes(bytes: List[UInt8]) -> Big:
        """Little-endian, non-negative."""
        var mag = List[UInt32](length=(len(bytes) + 3) // 4, fill=0)
        for i in range(len(bytes)):
            mag[i // 4] |= UInt32(bytes[i]) << UInt32(8 * (i % 4))
        return Big(False, mag^)

    @staticmethod
    def from_hex(s: String) raises -> Big:
        """Non-negative, big-endian hex digits."""
        var v = Big()
        for c in s.codepoint_slices():
            var d: Int
            if c >= "0" and c <= "9":
                d = Int(ord(c)) - Int(ord("0"))
            elif c >= "a" and c <= "f":
                d = Int(ord(c)) - Int(ord("a")) + 10
            elif c >= "A" and c <= "F":
                d = Int(ord(c)) - Int(ord("A")) + 10
            else:
                raise Error("hex digit")
            v = v.shl(4) + Big(d)
        return v^

    def _norm(mut self):
        while len(self.mag) > 0 and self.mag[len(self.mag) - 1] == 0:
            _ = self.mag.pop()
        if len(self.mag) == 0:
            self.neg = False

    def is_zero(self) -> Bool:
        return len(self.mag) == 0

    def bit_length(self) -> Int:
        if len(self.mag) == 0:
            return 0
        var top = Int(self.mag[len(self.mag) - 1])
        var n = 0
        while top > 0:
            n += 1
            top >>= 1
        return 32 * (len(self.mag) - 1) + n

    def bit(self, i: Int) -> Int:
        """Bit i of the magnitude."""
        if i // 32 >= len(self.mag):
            return 0
        return Int(self.mag[i // 32] >> UInt32(i % 32)) & 1

    def bits(self, n: Int) raises -> List[Int]:
        """The n low bits, bit 0 first; the value is non-negative and fits."""
        if self.neg or self.bit_length() > n:
            raise Error("the value is negative or exceeds " + String(n) + " bits")
        var v = List[Int](capacity=n)
        for i in range(n):
            v.append(self.bit(i))
        return v^

    def bytes(self, n: Int) raises -> List[UInt8]:
        """n little-endian bytes; non-negative and fits."""
        if self.neg or self.bit_length() > 8 * n:
            raise Error("the value is negative or exceeds " + String(n) + " bytes")
        var v = List[UInt8](length=n, fill=0)
        for i in range(n):
            if i // 4 < len(self.mag):
                v[i] = UInt8((self.mag[i // 4] >> UInt32(8 * (i % 4))) & 255)
        return v^

    def to_int(self) raises -> Int:
        if self.bit_length() > 62:
            raise Error("does not fit an Int")
        var m = 0
        for i in range(len(self.mag) - 1, -1, -1):
            m = (m << 32) | Int(self.mag[i])
        return -m if self.neg else m

    def write_to(self, mut writer: Some[Writer]):
        if self.is_zero():
            writer.write("0")
            return
        if self.neg:
            writer.write("-")
        var digits = List[UInt8]()
        var v = self.abs()
        var ten = Big(10)
        while not v.is_zero():
            try:
                var qr = v.divmod(ten)
                digits.append(UInt8(qr[1].to_int_unchecked()))
                v = qr[0].copy()
            except:
                return
        for i in range(len(digits) - 1, -1, -1):
            writer.write(Int(digits[i]))

    def to_int_unchecked(self) -> Int:
        var m = 0
        for i in range(len(self.mag) - 1, -1, -1):
            m = (m << 32) | Int(self.mag[i])
        return -m if self.neg else m

    def abs(self) -> Big:
        return Big(False, self.mag.copy())

    def __neg__(self) -> Big:
        return Big(not self.neg, self.mag.copy())

    @staticmethod
    def _cmp_mag(a: List[UInt32], b: List[UInt32]) -> Int:
        if len(a) != len(b):
            return 1 if len(a) > len(b) else -1
        for i in range(len(a) - 1, -1, -1):
            if a[i] != b[i]:
                return 1 if a[i] > b[i] else -1
        return 0

    @staticmethod
    def _add_mag(a: List[UInt32], b: List[UInt32]) -> List[UInt32]:
        var n = max(len(a), len(b))
        var v = List[UInt32](capacity=n + 1)
        var carry: UInt64 = 0
        for i in range(n):
            var s = carry + (UInt64(a[i]) if i < len(a) else 0) + (UInt64(b[i]) if i < len(b) else 0)
            v.append(UInt32(s & 0xFFFFFFFF))
            carry = s >> 32
        if carry > 0:
            v.append(UInt32(carry))
        return v^

    @staticmethod
    def _sub_mag(a: List[UInt32], b: List[UInt32]) -> List[UInt32]:
        """a - b for |a| >= |b|."""
        var v = List[UInt32](capacity=len(a))
        var borrow: Int64 = 0
        for i in range(len(a)):
            var d = Int64(a[i]) - (Int64(b[i]) if i < len(b) else 0) - borrow
            if d < 0:
                d += 1 << 32
                borrow = 1
            else:
                borrow = 0
            v.append(UInt32(d))
        return v^

    def cmp(self, other: Big) -> Int:
        """-1, 0, 1."""
        if self.neg != other.neg:
            return -1 if self.neg else 1
        var c = Big._cmp_mag(self.mag, other.mag)
        return -c if self.neg else c

    def __eq__(self, other: Big) -> Bool:
        return self.cmp(other) == 0

    def __ne__(self, other: Big) -> Bool:
        return self.cmp(other) != 0

    def __lt__(self, other: Big) -> Bool:
        return self.cmp(other) < 0

    def __ge__(self, other: Big) -> Bool:
        return self.cmp(other) >= 0

    def __add__(self, other: Big) -> Big:
        if self.neg == other.neg:
            return Big(self.neg, Big._add_mag(self.mag, other.mag))
        if Big._cmp_mag(self.mag, other.mag) >= 0:
            return Big(self.neg, Big._sub_mag(self.mag, other.mag))
        return Big(other.neg, Big._sub_mag(other.mag, self.mag))

    def __sub__(self, other: Big) -> Big:
        return self + (-other)

    def __mul__(self, other: Big) -> Big:
        var v = List[UInt32](length=len(self.mag) + len(other.mag), fill=0)
        for i in range(len(self.mag)):
            var carry: UInt64 = 0
            for j in range(len(other.mag)):
                var s = UInt64(self.mag[i]) * UInt64(other.mag[j]) + UInt64(v[i + j]) + carry
                v[i + j] = UInt32(s & 0xFFFFFFFF)
                carry = s >> 32
            var k = i + len(other.mag)
            while carry > 0:
                var s = UInt64(v[k]) + carry
                v[k] = UInt32(s & 0xFFFFFFFF)
                carry = s >> 32
                k += 1
        return Big(self.neg != other.neg, v^)

    def shl(self, k: Int) -> Big:
        var v = List[UInt32](length=k // 32, fill=0)
        var s = UInt32(k % 32)
        var carry: UInt32 = 0
        for i in range(len(self.mag)):
            v.append((self.mag[i] << s) | carry)
            carry = (self.mag[i] >> (32 - s)) if s > 0 else 0
        if carry > 0:
            v.append(carry)
        return Big(self.neg, v^)

    def shr(self, k: Int) -> Big:
        """The magnitude shifted down (exact division by 2^k for a negative value only when it divides)."""
        var v = List[UInt32]()
        var s = UInt32(k % 32)
        for i in range(k // 32, len(self.mag)):
            var hi = (self.mag[i + 1] << (32 - s)) if s > 0 and i + 1 < len(self.mag) else 0
            v.append((self.mag[i] >> s) | hi)
        return Big(self.neg, v^)

    def low(self, k: Int) -> Big:
        """The low k bits of the magnitude, non-negative."""
        var v = List[UInt32]()
        for i in range(min((k + 31) // 32, len(self.mag))):
            v.append(self.mag[i])
        if k % 32 != 0 and len(v) == (k + 31) // 32:
            v[len(v) - 1] &= (UInt32(1) << UInt32(k % 32)) - 1
        return Big(False, v^)

    def divmod(self, m: Big) raises -> Tuple[Big, Big]:
        """Floor division: self = q m + r with 0 <= r < m, for m > 0."""
        if m.neg or m.is_zero():
            raise Error("division by a non-positive modulus")
        var q = List[UInt32](length=len(self.mag), fill=0)
        var r = Big()
        for i in range(self.bit_length() - 1, -1, -1):
            r = r.shl(1) + Big(self.bit(i))
            if Big._cmp_mag(r.mag, m.mag) >= 0:
                r = Big(False, Big._sub_mag(r.mag, m.mag))
                q[i // 32] |= UInt32(1) << UInt32(i % 32)
        var quo = Big(False, q^)
        if self.neg and not r.is_zero():
            return (-quo - Big(1), m - r)
        if self.neg:
            return (-quo, r^)
        return (quo^, r^)

    def mod(self, m: Big) raises -> Big:
        return self.divmod(m)[1].copy()

    def mulmod(self, other: Big, m: Big) raises -> Big:
        return (self * other).mod(m)

    def pow_mod(self, e: Big, m: Big) raises -> Big:
        var r = Big(1)
        var b = self.mod(m)
        for i in range(e.bit_length() - 1, -1, -1):
            r = r.mulmod(r, m)
            if e.bit(i) == 1:
                r = r.mulmod(b, m)
        return r^

    def inv_mod(self, m: Big) raises -> Big:
        """For prime m."""
        if self.mod(m).is_zero():
            raise Error("zero has no inverse")
        return self.pow_mod(m - Big(2), m)
