"""secp256k1 on the host: field and curve arithmetic on `Big`, the GLV split, the digit recoding of
`docs/ecdsa.md` section 2, the blinding point of section 3, and the plain ECDSA check the verifier runs
before the circuit. Affine points, Fermat inversions, the field reduction by `2^256 = 2^32 + 977`.
ponytail: a few milliseconds per curve operation; Jacobian coordinates or a limb field if the verifier's
curve work ever shows in a profile."""

from caracal7.core.bytes import host_base
from caracal7.core.hash import Blake3
from caracal7.relations.bigint import Big


@fieldwise_init
struct Point(Copyable, Movable, Equatable, Writable):
    """Affine, `inf` the identity (coordinates then zero)."""
    var x: Big
    var y: Big
    var inf: Bool

    @staticmethod
    def identity() -> Point:
        return Point(Big(), Big(), True)

    def __eq__(self, other: Point) -> Bool:
        if self.inf or other.inf:
            return self.inf == other.inf
        return self.x == other.x and self.y == other.y

    def __ne__(self, other: Point) -> Bool:
        return not (self == other)

    def write_to(self, mut writer: Some[Writer]):
        if self.inf:
            writer.write("O")
        else:
            writer.write("(", self.x, ", ", self.y, ")")


struct Curve(Movable):
    """The constants (libsecp256k1's endomorphism and lattice basis) and the operations."""
    var p: Big
    var n: Big
    var g: Point
    var beta: Big
    var lam: Big
    var a1: Big
    var b1: Big
    var a2: Big
    var b2: Big

    def __init__(out self) raises:
        self.p = Big.from_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f")
        self.n = Big.from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141")
        self.g = Point(Big.from_hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"),
                       Big.from_hex("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"), False)
        self.beta = Big.from_hex("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee")
        self.lam = Big.from_hex("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72")
        self.a1 = Big.from_hex("3086d221a7d46bcde86c90e49284eb15")
        self.b1 = -Big.from_hex("e4437ed6010e88286f547fa90abfe4c3")
        self.a2 = Big.from_hex("114ca50f7a8e2f3f657c1108d9d44cfd8")
        self.b2 = Big.from_hex("3086d221a7d46bcde86c90e49284eb15")

    # ---- the field ----

    def red(self, v: Big) -> Big:
        """v mod p for 0 <= v < 2^512, by the fold twice and at most two subtractions."""
        var c = Big(977) + Big(1).shl(32)
        var t = v.low(256) + v.shr(256) * c
        t = t.low(256) + t.shr(256) * c
        while t >= self.p:
            t = t - self.p
        return t^

    def fadd(self, a: Big, b: Big) -> Big:
        var t = a + b
        return t - self.p if t >= self.p else t^

    def fsub(self, a: Big, b: Big) -> Big:
        var t = a - b
        return t + self.p if t.neg else t^

    def fmul(self, a: Big, b: Big) -> Big:
        return self.red(a * b)

    def fpow(self, a: Big, e: Big) -> Big:
        var r = Big(1)
        for i in range(e.bit_length() - 1, -1, -1):
            r = self.fmul(r, r)
            if e.bit(i) == 1:
                r = self.fmul(r, a)
        return r^

    def finv(self, a: Big) -> Big:
        return self.fpow(a, self.p - Big(2))

    def fsqrt(self, a: Big) raises -> Big:
        """p = 3 mod 4: a^((p + 1) / 4), checked."""
        var r = self.fpow(a, (self.p + Big(1)).shr(2))
        if self.fmul(r, r) != a:
            raise Error("not a square")
        return r^

    # ---- the group ----

    def on_curve(self, a: Point) -> Bool:
        if a.inf:
            return True
        if a.x.neg or a.y.neg or a.x >= self.p or a.y >= self.p:
            return False
        var lhs = self.fmul(a.y, a.y)
        var rhs = self.fadd(self.fmul(self.fmul(a.x, a.x), a.x), Big(7))
        return lhs == rhs

    def neg(self, a: Point) -> Point:
        if a.inf or a.y.is_zero():
            return a.copy()
        return Point(a.x.copy(), self.p - a.y, False)

    def double(self, a: Point) -> Point:
        if a.inf or a.y.is_zero():
            return Point.identity()
        var x2 = self.fmul(a.x, a.x)
        var l = self.fmul(self.fadd(self.fadd(x2, x2), x2), self.finv(self.fadd(a.y, a.y)))
        var x3 = self.fsub(self.fsub(self.fmul(l, l), a.x), a.x)
        var y3 = self.fsub(self.fmul(l, self.fsub(a.x, x3)), a.y)
        return Point(x3^, y3^, False)

    def add(self, a: Point, b: Point) -> Point:
        if a.inf:
            return b.copy()
        if b.inf:
            return a.copy()
        if a.x == b.x:
            return self.double(a) if a.y == b.y else Point.identity()
        var l = self.fmul(self.fsub(b.y, a.y), self.finv(self.fsub(b.x, a.x)))
        var x3 = self.fsub(self.fsub(self.fmul(l, l), a.x), b.x)
        var y3 = self.fsub(self.fmul(l, self.fsub(a.x, x3)), a.y)
        return Point(x3^, y3^, False)

    def mul(self, a: Point, k: Big) -> Point:
        """k a for a signed k, double-and-add."""
        var r = Point.identity()
        for i in range(k.bit_length() - 1, -1, -1):
            r = self.double(r)
            if k.bit(i) == 1:
                r = self.add(r, a)
        return self.neg(r) if k.neg else r^

    def phi(self, a: Point) -> Point:
        """lambda a = (beta x, y)."""
        if a.inf:
            return a.copy()
        return Point(self.fmul(self.beta, a.x), a.y.copy(), False)

    # ---- scalars ----

    def _round_div(self, a: Big, b: Big) raises -> Big:
        """round(a / b) for b > 0."""
        return (a.shl(1) + b).divmod(b.shl(1))[0].copy()

    def split(self, k: Big) raises -> Tuple[Big, Big]:
        """k = k1 + k2 lambda mod n with |k1|, |k2| below 2^128 (the lattice basis (a1, b1), (a2, b2))."""
        var c1 = self._round_div(self.b2 * k, self.n)
        var c2 = self._round_div(-self.b1 * k, self.n)
        var k1 = k - c1 * self.a1 - c2 * self.a2
        var k2 = -c1 * self.b1 - c2 * self.b2
        return (k1^, k2^)

    def blinding(self, msg: List[UInt8]) raises -> Point:
        """hash_to_curve by try-and-increment: x = blake3(msg || counter) mod p, y the even root."""
        var m = msg.copy()
        m.append(0)
        var digest = List[UInt8](length=32, fill=0)
        for ctr in range(256):
            m[len(m) - 1] = UInt8(ctr)
            Blake3.leaf(host_base(m), len(m), host_base(digest))
            var x = Big.from_bytes(digest).mod(self.p)
            var rhs = self.fadd(self.fmul(self.fmul(x, x), x), Big(7))
            var y = self.fpow(rhs, (self.p + Big(1)).shr(2))
            if self.fmul(y, y) == rhs:
                if y.bit(0) == 1:
                    y = self.p - y
                return Point(x^, y^, False)
        raise Error("no curve point in 256 tries")

    def verify(self, r: Big, s: Big, e: Big, q: Point) raises -> Bool:
        """Plain ECDSA: the public-input conditions of docs/ecdsa.md section 1, then x(u1 G + u2 Q) = r mod n."""
        if r.neg or s.neg or r.is_zero() or s.is_zero() or r >= self.n or s >= self.n or e.neg or e >= self.n:
            return False
        if q.inf or not self.on_curve(q):
            return False
        var w = s.inv_mod(self.n)
        var pt = self.add(self.mul(self.g, e.mulmod(w, self.n)), self.mul(q, r.mulmod(w, self.n)))
        return not pt.inf and pt.x.mod(self.n) == r


def recode(k: Big, b: Int, digits: Int = 32) raises -> List[Int]:
    """Signed digits of an odd k, |k| < 2^(b digits): the low digits d = (k mod 2^(b + 1)) - 2^b, odd, then
    the remainder as the last digit."""
    if k.bit(0) != 1:
        raise Error("recode takes an odd value")
    var v = List[Int](capacity=digits)
    var rest = k.copy()
    var span = Big(1).shl(b + 1)
    for _ in range(digits - 1):
        var d = rest.mod(span).to_int() - (1 << b)
        v.append(d)
        rest = (rest - Big(d)).shr(b)
    v.append(rest.to_int())
    return v^


def skew(k: Big) -> Tuple[Big, Int]:
    """The odd value recoded for a half k and its window-0 correction c: k = (k - c) + c, c = 1 for even
    k, 2 for odd, so 0 and 1 both recode -1."""
    var c = 2 if k.bit(0) == 1 else 1
    return (k - Big(c), c)
