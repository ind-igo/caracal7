"""ECDSA on secp256k1 (docs/ecdsa.md): the curve constants, libsecp256k1's endomorphism and lattice basis
for the GLV split, the walk of section 2 (Straus-Shamir over four GLV halves, 22 windows of 6 bits) and
the `EcdsaK1` workload on the mulmod chains. The curve arithmetic and the op walk are `ecurve.mojo`; the
field is `fp_k1.mojo`."""

from core.params import Params
from workloads.bigint import Big
from workloads.fp_k1 import FpK1
from workloads.ecurve import Curve, Point, Walk, window_points, check_inputs, public_bytes, slice32, INPUT
from workloads.mulmod import Op, VALUE, mulmod_statement, circuit_values, circuit_trace, circuit_public_data
from relations.statement import Statement, Layout
from workload import Workload

comptime K1 = Curve[FpK1]
comptime WINDOWS = 22       # ceil(128 / QW): the four GLV halves are below 2^128


def secp256k1() raises -> K1:
    return K1(Big.from_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"),
              Big.from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"),
              Big(), Big(7),
              Point(Big.from_hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"),
                    Big.from_hex("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"), False))


struct Glv(Movable):
    """The endomorphism phi(x, y) = (beta x, y) = lambda (x, y) and the lattice basis of the split."""
    var beta: Big
    var lam: Big
    var a1: Big
    var b1: Big
    var a2: Big
    var b2: Big

    def __init__(out self) raises:
        self.beta = Big.from_hex("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee")
        self.lam = Big.from_hex("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72")
        self.a1 = Big.from_hex("3086d221a7d46bcde86c90e49284eb15")
        self.b1 = -Big.from_hex("e4437ed6010e88286f547fa90abfe4c3")
        self.a2 = Big.from_hex("114ca50f7a8e2f3f657c1108d9d44cfd8")
        self.b2 = Big.from_hex("3086d221a7d46bcde86c90e49284eb15")

    def phi(self, c: K1, a: Point) -> Point:
        if a.inf:
            return a.copy()
        return Point(c.fmul(self.beta, a.x), a.y.copy(), False)

    @staticmethod
    def _round_div(a: Big, b: Big) raises -> Big:
        """round(a / b) for b > 0."""
        return (a.shl(1) + b).divmod(b.shl(1))[0].copy()

    def split(self, c: K1, k: Big) raises -> Tuple[Big, Big]:
        """k = k1 + k2 lambda mod n with |k1|, |k2| below 2^128 (the lattice basis (a1, b1), (a2, b2))."""
        var c1 = Glv._round_div(self.b2 * k, c.n)
        var c2 = Glv._round_div(-self.b1 * k, c.n)
        var k1 = k - c1 * self.a1 - c2 * self.a2
        var k2 = -c1 * self.b1 - c2 * self.b2
        return (k1^, k2^)


def walk(var c: K1, r: Big, s: Big, e: Big, q: Point, live: Bool) raises -> Walk[FpK1]:
    """The fixed circuit; with `live`, the verifier's checks on (r, s, e, Q), then the public factor values
    and hints of that signature. Window w adds one public point P_w = sum over the four GLV halves
    (u2 on Q, phi(Q); u1 on G, phi(G)) of d_w T (Straus-Shamir), built from four small tables."""
    var w = Walk[FpK1](c^, live)
    var pts = List[Point]()
    for _ in range(WINDOWS):
        pts.append(Point.identity())
    var b16 = Point.identity()
    var b = Point.identity()
    if live:
        check_inputs(w.c, r, s, e, q)
        var glv = Glv()
        var inv = s.inv_mod(w.c.n)
        var h1 = glv.split(w.c, e.mulmod(inv, w.c.n))
        var h2 = glv.split(w.c, r.mulmod(inv, w.c.n))
        var ks: List[Big] = [h2[0].copy(), h2[1].copy(), h1[0].copy(), h1[1].copy()]
        var bases: List[Point] = [q.copy(), glv.phi(w.c, q), w.c.g.copy(), glv.phi(w.c, w.c.g)]
        pts = window_points(w.c, bases, ks, WINDOWS)
        b = w.c.blinding(public_bytes(r, s, e, q))
        b16 = b.copy()
        for _ in range(4):
            b16 = w.c.double(b16)
    w.schedule(pts, b, b16, r)
    return w^


struct EcdsaK1(Workload, Movable):
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
        var w = walk(secp256k1(), Big(), Big(), Big(), Point.identity(), False)
        return w.ops.copy()

    def statement[p: Params](self) raises -> Statement:
        return mulmod_statement(True, EcdsaK1.circuit(), pin=False)

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        var w = walk(secp256k1(), self.r, self.s, self.e, self.q, True)
        return circuit_trace[p](layout, circuit_values(w.inputs, w.ops, w.hints), w.ops)

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        return public_bytes(self.r, self.s, self.e, self.q)

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        if len(public_inputs) != INPUT:
            raise Error("public inputs are r, s, e, x_Q, y_Q")
        var w = walk(secp256k1(), slice32(public_inputs, 0), slice32(public_inputs, 32), slice32(public_inputs, 64),
                     Point(slice32(public_inputs, 96), slice32(public_inputs, 128), False), True)
        var values = List[UInt8](capacity=len(w.inputs) * VALUE)
        for v in w.inputs:
            values.extend(v.copy())
        return circuit_public_data[p](w.ops, values, 0)
