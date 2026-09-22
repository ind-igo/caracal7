"""Short Weierstrass curves on the host and the ECDSA circuit walk shared by the curve variants
(`ecdsa_k1.mojo`, `ecdsa_p256.mojo`): affine points on `Big`, a `Curve` over a limb field `F`, the digit
recoding of `docs/ecdsa.md` section 2, the blinding point of section 3, the plain ECDSA check the verifier
runs before the circuit, and the `Walk` that emits the group operations of section 4 as mulmod ops.
ponytail: `Curve` converts to the limb field at every field operation; Jacobian coordinates if the verifier's
curve work ever shows in a profile."""

from core.bytes import host_base
from core.hash import Blake3
from workloads.bigint import Big
from workloads.mulmod import Op, PUB, NIL, MOD_P, MOD_N, VALUE, mul, add, sub, eq, canon, guard, hint


trait Field(Copyable, Movable, ImplicitlyCopyable, Equatable, Deinitable):
    """A prime field on limbs: canonical values in and out through `Big`."""

    @staticmethod
    def from_big(b: Big) -> Self:
        ...

    def to_big(self) -> Big:
        ...

    def __add__(self, o: Self) -> Self:
        ...

    def __sub__(self, o: Self) -> Self:
        ...

    def __mul__(self, o: Self) -> Self:
        ...

    def inv(self) -> Self:
        """Zero maps to zero."""
        ...


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


@fieldwise_init
struct Curve[F: Field](Movable):
    """y^2 = x^3 + a x + b over F_p, prime order n, cofactor 1, generator g; `a` canonical (p - 3 for a = -3)."""
    var p: Big
    var n: Big
    var a: Big
    var b: Big
    var g: Point

    # ---- the field ----

    def fadd(self, a: Big, b: Big) -> Big:
        var t = a + b
        return t - self.p if t >= self.p else t^

    def fsub(self, a: Big, b: Big) -> Big:
        var t = a - b
        return t + self.p if t.neg else t^

    def fmul(self, a: Big, b: Big) -> Big:
        return (Self.F.from_big(a) * Self.F.from_big(b)).to_big()

    def fpow(self, a: Big, e: Big) -> Big:
        var x = Self.F.from_big(a)
        var r = Self.F.from_big(Big(1))
        for i in range(e.bit_length() - 1, -1, -1):
            r = r * r
            if e.bit(i) == 1:
                r = r * x
        return r.to_big()

    def finv(self, a: Big) -> Big:
        return Self.F.from_big(a).inv().to_big()

    def rhs(self, x: Big) -> Big:
        """x^3 + a x + b."""
        return self.fadd(self.fadd(self.fmul(self.fmul(x, x), x), self.fmul(self.a, x)), self.b)

    # ---- the group ----

    def on_curve(self, a: Point) -> Bool:
        if a.inf:
            return True
        if a.x.neg or a.y.neg or a.x >= self.p or a.y >= self.p:
            return False
        return self.fmul(a.y, a.y) == self.rhs(a.x)

    def neg(self, a: Point) -> Point:
        if a.inf or a.y.is_zero():
            return a.copy()
        return Point(a.x.copy(), self.p - a.y, False)

    def double(self, a: Point) -> Point:
        if a.inf or a.y.is_zero():
            return Point.identity()
        var x2 = self.fmul(a.x, a.x)
        var num = self.fadd(self.fadd(self.fadd(x2, x2), x2), self.a)
        var l = self.fmul(num, self.finv(self.fadd(a.y, a.y)))
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

    def blinding(self, msg: List[UInt8]) raises -> Point:
        """hash_to_curve by try-and-increment: x = blake3(msg || counter) mod p, y the even root (p = 3 mod 4)."""
        var m = msg.copy()
        m.append(0)
        var digest = List[UInt8](length=32, fill=0)
        for ctr in range(256):
            m[len(m) - 1] = UInt8(ctr)
            Blake3.leaf(host_base(m), len(m), host_base(digest))
            var x = Big.from_bytes(digest).mod(self.p)
            var rhs = self.rhs(x)
            var y = self.fpow(rhs, (self.p + Big(1)).shr(2))
            if self.fmul(y, y) == rhs:
                if y.bit(0) == 1:
                    y = self.p - y
                return Point(x^, y^, False)
        raise Error("no curve point in 256 tries")

    def valid_inputs(self, r: Big, s: Big, e: Big, q: Point) -> Bool:
        """The public-input conditions of docs/ecdsa.md section 1: r, s in (0, n), e below n, Q a finite
        point of the curve with canonical coordinates."""
        if r.neg or s.neg or r.is_zero() or s.is_zero() or r >= self.n or s >= self.n or e.neg or e >= self.n:
            return False
        return not q.inf and self.on_curve(q)

    def verify(self, r: Big, s: Big, e: Big, q: Point) raises -> Bool:
        """Plain ECDSA: the input conditions, then x(u1 G + u2 Q) = r mod n."""
        if not self.valid_inputs(r, s, e, q):
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
    """The odd value recoded for a scalar k and its window-0 correction c: k = (k - c) + c, c = 1 for even
    k, 2 for odd, so 0 and 1 both recode -1."""
    var c = 2 if k.bit(0) == 1 else 1
    return (k - Big(c), c)


# ---- the circuit ----

comptime QW = 6             # window bits
comptime INPUT = 5 * 32     # r, s, e, x_Q, y_Q, 32 little-endian bytes each


@fieldwise_init
struct Ref(Copyable, Movable):
    """An op operand: `r` the reference (`PUB`, an op index, a hint) and, for a public one, its value."""
    var r: Int
    var v: Big


struct Walk[F: Field]:
    """One pass over the schedule of docs/ecdsa.md section 2, emitting the ops and, when `live`, the public
    factor values in factor order (each op's public operands x, y, z, then a public s) and the slope hints.
    The circuit is fixed: which public point a step adds changes per signature, the ops do not."""
    var c: Curve[Self.F]
    var live: Bool
    var ops: List[Op]
    var inputs: List[List[UInt8]]
    var hints: List[Big]
    var acc: Point
    var ax: Ref
    var ay: Ref

    def __init__(out self, var c: Curve[Self.F], live: Bool):
        self.c = c^
        self.live = live
        self.ops = List[Op]()
        self.inputs = List[List[UInt8]]()
        self.hints = List[Big]()
        self.acc = Point.identity()
        self.ax = Ref(PUB, Big())
        self.ay = Ref(PUB, Big())

    def pub(self, v: Big) -> Ref:
        return Ref(PUB, v.copy())

    def hint(mut self, v: Big) -> Ref:
        self.hints.append(v.copy())
        return Ref(hint(len(self.hints) - 1), Big())

    def emit(mut self, op: Op, refs: List[Ref]) raises -> Ref:
        """Appends the op; the public values of `refs` (its operands and constant, in role order) in factor
        order. Returns the op's output."""
        if self.live:
            for operand in refs:
                if operand.r == PUB:
                    self.inputs.append(operand.v.bytes(VALUE))
        self.ops.append(op)
        return Ref(len(self.ops) - 1, Big())

    def mul(mut self, x: Ref, y: Ref) raises -> Ref:
        return self.emit(mul(x.r, y.r), [x.copy(), y.copy()])

    def sub(mut self, x: Ref, y: Ref) raises -> Ref:
        return self.emit(sub(x.r, y.r), [x.copy(), y.copy()])

    def add3(mut self, x: Ref, y: Ref, sy: Int, z: Ref, sz: Int) raises -> Ref:
        return self.emit(add(x.r, y.r, sy, z.r, sz), [x.copy(), y.copy(), z.copy()])

    def eq(mut self, x: Ref, y: Ref, mod: Int = MOD_P) raises:
        _ = self.emit(eq(x.r, y.r, -1, NIL, 0, mod), [x.copy(), y.copy()])

    def check(mut self, x: Ref, op: Op, const: Big) raises:
        _ = self.emit(op, [x.copy(), self.pub(const)])

    def slope(mut self, num: Big, den: Big) -> Ref:
        return self.hint(self.c.fmul(num, self.c.finv(den)) if self.live else Big())

    def start(mut self, b16: Point):
        self.acc = b16.copy()
        self.ax = self.pub(b16.x)
        self.ay = self.pub(b16.y)

    def addition(mut self, t: Point, x_only: Bool = False) raises:
        """acc + t for a public finite t != +-acc: dx canonical and nonzero, l dx = dy, x3, y3."""
        var ax = self.ax.copy()
        var ay = self.ay.copy()
        var x2 = self.pub(t.x)
        var y2 = self.pub(t.y)
        var dx = self.sub(x2, ax)
        self.check(dx, canon(dx.r), self.c.p - Big(1))
        self.check(dx, guard(dx.r), Big(1))
        var dy = self.sub(y2, ay)
        var l = self.slope(self.c.fsub(t.y, self.acc.y), self.c.fsub(t.x, self.acc.x))
        var ldx = self.mul(l, dx)
        self.eq(ldx, dy)
        var ll = self.mul(l, l)
        var x3 = self.add3(ll, ax, -1, x2, -1)
        if x_only:
            self.ax = x3^
            return
        var d = self.sub(ax, x3)
        var ld = self.mul(l, d)
        var y3 = self.sub(ld, ay)
        if self.live:
            self.acc = self.c.add(self.acc, t)
        self.ax = x3.copy()
        self.ay = y3.copy()

    def doubling(mut self) raises:
        """2 acc: l (2 y) = 3 x^2 + a (y != 0 on a curve of odd order), x3, y3. With a = 0 the numerator is
        one three-operand op, otherwise two (the second adds the public constant a)."""
        var ax = self.ax.copy()
        var ay = self.ay.copy()
        var xx = self.mul(ax, ax)
        var num: Ref
        if self.c.a.is_zero():
            num = self.add3(xx, xx, 1, xx, 1)
        else:
            var two = self.add3(xx, xx, 1, Ref(NIL, Big()), 0)
            num = self.add3(two, xx, 1, self.pub(self.c.a), 1)
        var yy = self.add3(ay, ay, 1, Ref(NIL, Big()), 0)
        var x2 = Big()
        if self.live:
            x2 = self.c.fmul(self.acc.x, self.acc.x)
        var l = self.slope(self.c.fadd(self.c.fadd(self.c.fadd(x2, x2), x2), self.c.a), self.c.fadd(self.acc.y, self.acc.y))
        var lyy = self.mul(l, yy)
        self.eq(lyy, num)
        var ll = self.mul(l, l)
        var x3 = self.add3(ll, ax, -1, ax, -1)
        var d = self.sub(ax, x3)
        var ld = self.mul(l, d)
        var y3 = self.sub(ld, ay)
        if self.live:
            self.acc = self.c.double(self.acc)
        self.ax = x3.copy()
        self.ay = y3.copy()

    def closing(mut self, r: Big) raises:
        """x_R canonical and x_R = r mod n on the mod-n chain."""
        var ax = self.ax.copy()
        self.check(ax, canon(ax.r), self.c.p - Big(1))
        self.eq(ax, self.pub(r), MOD_N)

    def addend(mut self, pt: Point, mut t: Big, i: Int, b: Point) raises:
        """Window i adds its public point, or B when that point is the identity (t records 2^(QW i))."""
        if self.live and pt.inf:
            t = t + Big(1).shl(QW * i)
            self.addition(b)
        else:
            self.addition(pt)

    def schedule(mut self, pts: List[Point], b: Point, b16: Point, r: Big) raises:
        """acc = 16 B; the top window; then per window QW doublings and one addition; the closing constant
        -(2^(4 + QW (windows - 1)) + t) B with x only; x_R = r mod n. Live values only when `live`."""
        var windows = len(pts)
        var t = Big()
        self.start(b16)
        self.addend(pts[windows - 1], t, windows - 1, b)
        for i in range(windows - 2, -1, -1):
            for _ in range(QW):
                self.doubling()
            self.addend(pts[i], t, i, b)
        var close = Point.identity()
        if self.live:
            var btop = b16.copy()
            for _ in range((windows - 1) * QW):
                btop = self.c.double(btop)
            close = self.c.neg(self.c.add(btop, self.c.mul(b, t)))
        self.addition(close, x_only=True)
        self.closing(r)


def digit_points[F: Field](c: Curve[F], t: Point, digits: List[Int], corr: Int) raises -> List[Point]:
    """d t per window, window 0 with the parity correction; a small table of multiples of t."""
    var tab: List[Point] = [Point.identity()]
    var top = 1 << QW
    for _ in range(top + 2):
        tab.append(c.add(tab[len(tab) - 1], t))
    var v = List[Point]()
    for w in range(len(digits)):
        var d = digits[w] + (corr if w == 0 else 0)
        v.append(c.neg(tab[-d]) if d < 0 else tab[d].copy())
    return v^


def window_points[F: Field](c: Curve[F], bases: List[Point], ks: List[Big], windows: Int) raises -> List[Point]:
    """Straus-Shamir: window w's public point P_w = sum over the bases of d_{i,w} T_i, a base's sign
    absorbed into the base, each odd magnitude recoded after the parity skew."""
    var pts = List[Point]()
    for _ in range(windows):
        pts.append(Point.identity())
    for i in range(len(bases)):
        var t = c.neg(bases[i]) if ks[i].neg else bases[i].copy()
        var sk = skew(ks[i].abs())
        var part = digit_points(c, t, recode(sk[0], QW, windows), sk[1])
        for j in range(windows):
            pts[j] = c.add(pts[j], part[j])
    return pts^


def check_inputs[F: Field](c: Curve[F], r: Big, s: Big, e: Big, q: Point) raises:
    if r.neg or s.neg or r.is_zero() or s.is_zero() or r >= c.n or s >= c.n or e.neg or e >= c.n:
        raise Error("r, s in (0, n), e below n")
    if q.inf or not c.on_curve(q):
        raise Error("Q is a finite point of the curve with canonical coordinates")


def public_bytes(r: Big, s: Big, e: Big, q: Point) raises -> List[UInt8]:
    var v = r.bytes(32)
    v.extend(s.bytes(32))
    v.extend(e.bytes(32))
    v.extend(q.x.bytes(32))
    v.extend(q.y.bytes(32))
    return v^


def slice32(bytes: List[UInt8], at: Int) -> Big:
    var v = List[UInt8](capacity=32)
    for i in range(32):
        v.append(bytes[at + i])
    return Big.from_bytes(v)
