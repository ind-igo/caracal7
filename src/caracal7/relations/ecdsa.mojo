"""secp256k1 on the host: field and curve arithmetic on `Big`, the GLV split, the digit recoding of
`docs/ecdsa.md` section 2, the blinding point of section 3, and the plain ECDSA check the verifier runs
before the circuit. Affine points, Fermat inversions, the field reduction by `2^256 = 2^32 + 977`.
ponytail: a few milliseconds per curve operation; Jacobian coordinates or a limb field if the verifier's
curve work ever shows in a profile."""

from caracal7.core.bytes import host_base
from caracal7.core.hash import Blake3
from caracal7.core.params import Params
from caracal7.relations.bigint import Big
from caracal7.relations.mulmod import Op, PUB, NIL, MOD_P, MOD_N, VALUE, mul, add, sub, eq, canon, guard, hint, mulmod_statement, circuit_values, circuit_trace, circuit_public_data
from caracal7.relations.statement import Statement, Layout
from caracal7.workload import Workload


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


# ---- the circuit ----

comptime WINDOWS = 32
comptime QW = 4             # window bits on the Q side (GLV halves below 2^128)
comptime GW = 8             # window bits on the fixed base
comptime INPUT = 5 * 32     # r, s, e, x_Q, y_Q, 32 little-endian bytes each


@fieldwise_init
struct Ref(Copyable, Movable):
    """An op operand: `r` the reference (`PUB`, an op index, a hint) and, for a public one, its value."""
    var r: Int
    var v: Big


struct Walk:
    """One pass over the schedule of docs/ecdsa.md section 2, emitting the ops and, when `live`, the public
    factor values in factor order (each op's public operands x, y, z, then a public s) and the slope hints.
    The circuit is fixed: which public point a step adds changes per signature, the ops do not."""
    var c: Curve
    var live: Bool
    var ops: List[Op]
    var inputs: List[List[UInt8]]
    var hints: List[Big]
    var acc: Point
    var ax: Ref
    var ay: Ref

    def __init__(out self, var c: Curve, live: Bool):
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
        """2 acc: l (2 y) = 3 x^2 (y != 0 on this curve of odd order), x3, y3."""
        var ax = self.ax.copy()
        var ay = self.ay.copy()
        var xx = self.mul(ax, ax)
        var three = self.add3(xx, xx, 1, xx, 1)
        var yy = self.add3(ay, ay, 1, Ref(NIL, Big()), 0)
        var x2 = Big()
        if self.live:
            x2 = self.c.fmul(self.acc.x, self.acc.x)
        var l = self.slope(self.c.fadd(self.c.fadd(x2, x2), x2), self.c.fadd(self.acc.y, self.acc.y))
        var lyy = self.mul(l, yy)
        self.eq(lyy, three)
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


def _digit_points(c: Curve, t: Point, digits: List[Int], corr: Int) raises -> List[Point]:
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


def _signed(c: Curve, t: Point, k: Big) -> Point:
    return c.neg(t) if k.neg else t.copy()


def walk(var c: Curve, r: Big, s: Big, e: Big, q: Point, live: Bool) raises -> Walk:
    """The fixed circuit; with `live`, the verifier's checks on (r, s, e, Q), then the public factor values
    and hints of that signature."""
    var w = Walk(c^, live)
    var pts1 = List[Point]()
    var pts2 = List[Point]()
    var ptsg = List[Point]()
    var b16 = Point.identity()
    var b = Point.identity()
    if live:
        if r.neg or s.neg or r.is_zero() or s.is_zero() or r >= w.c.n or s >= w.c.n or e.neg or e >= w.c.n:
            raise Error("r, s in (0, n), e below n")
        if q.inf or not w.c.on_curve(q):
            raise Error("Q is a finite point of the curve with canonical coordinates")
        var inv = s.inv_mod(w.c.n)
        var u1 = e.mulmod(inv, w.c.n)
        var u2 = r.mulmod(inv, w.c.n)
        var halves = w.c.split(u2)
        for i in range(2):
            var k = halves[0].copy() if i == 0 else halves[1].copy()
            var base = q.copy() if i == 0 else w.c.phi(q)
            var t = _signed(w.c, base, k)
            var sk = skew(k.abs())
            var pts = _digit_points(w.c, t, recode(sk[0], QW), sk[1])
            if i == 0:
                pts1 = pts^
            else:
                pts2 = pts^
        # ponytail: the fixed-base tables recomputed per signature (about 800 curve operations); constants later
        var sk = skew(u1)
        var dg = recode(sk[0], GW)
        var pw = w.c.g.copy()
        for i in range(WINDOWS):
            var d = dg[i] + (sk[1] if i == 0 else 0)
            ptsg.append(w.c.mul(pw, Big(d)))
            for _ in range(GW):
                pw = w.c.double(pw)
        b = w.c.blinding(public_bytes(r, s, e, q))
        b16 = b.copy()
        for _ in range(4):
            b16 = w.c.double(b16)
    else:
        for _ in range(WINDOWS):
            pts1.append(Point.identity())
            pts2.append(Point.identity())
            ptsg.append(Point.identity())
    var t = 0

    def addend(mut w: Walk, pt: Point, mut t: Int, b: Point) raises:
        if w.live and pt.inf:
            t += 1
            w.addition(b)
        else:
            w.addition(pt)

    w.start(b16)
    addend(w, pts1[WINDOWS - 1], t, b)
    addend(w, pts2[WINDOWS - 1], t, b)
    for i in range(WINDOWS - 2, -1, -1):
        for _ in range(QW):
            w.doubling()
        addend(w, pts1[i], t, b)
        addend(w, pts2[i], t, b)
    for i in range(WINDOWS):
        addend(w, ptsg[i], t, b)
    var close = Point.identity()
    if live:
        var b128 = b16.copy()
        for _ in range(WINDOWS * QW - 4):
            b128 = w.c.double(b128)
        close = w.c.neg(w.c.add(b128, w.c.mul(b, Big(t))))
    w.addition(close, x_only=True)
    w.closing(r)
    return w^


def public_bytes(r: Big, s: Big, e: Big, q: Point) raises -> List[UInt8]:
    var v = r.bytes(32)
    v.extend(s.bytes(32))
    v.extend(e.bytes(32))
    v.extend(q.x.bytes(32))
    v.extend(q.y.bytes(32))
    return v^


def _slice(bytes: List[UInt8], at: Int) -> Big:
    var v = List[UInt8](capacity=32)
    for i in range(32):
        v.append(bytes[at + i])
    return Big.from_bytes(v)


struct Ecdsa(Workload, Movable):
    """One secp256k1 signature on the mulmod chains (docs/ecdsa.md): public inputs (r, s, e, Q), the witness
    the slopes of the fixed addition chain."""
    var r: Big
    var s: Big
    var e: Big
    var q: Point

    def __init__(out self, var r: Big, var s: Big, var e: Big, var q: Point):
        self.r = r^
        self.s = s^
        self.e = e^
        self.q = q^

    @staticmethod
    def circuit() raises -> List[Op]:
        var w = walk(Curve(), Big(), Big(), Big(), Point.identity(), False)
        return w.ops.copy()

    def statement(self) raises -> Statement:
        return mulmod_statement(True, Ecdsa.circuit(), pin=False)

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        var w = walk(Curve(), self.r, self.s, self.e, self.q, True)
        return circuit_trace[p](layout, circuit_values(w.inputs, w.ops, w.hints), w.ops)

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        return public_bytes(self.r, self.s, self.e, self.q)

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        if len(public_inputs) != INPUT:
            raise Error("public inputs are r, s, e, x_Q, y_Q")
        var w = walk(Curve(), _slice(public_inputs, 0), _slice(public_inputs, 32), _slice(public_inputs, 64),
                     Point(_slice(public_inputs, 96), _slice(public_inputs, 128), False), True)
        var values = List[UInt8](capacity=len(w.inputs) * VALUE)
        for v in w.inputs:
            values.extend(v.copy())
        return circuit_public_data[p](w.ops, values, 0)
