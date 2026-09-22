"""ECDSA on secp256r1 (P-256; docs/ecdsa.md section 8): the curve constants (a = -3, no endomorphism), the
walk of section 2 over the two bases Q and G with the full 256-bit scalars (43 windows of 6 bits, 252
doublings, 44 additions) and the `EcdsaP256` workload on the mulmod chains with the P-256 product
reduction (`CURVE_P256`). The field is `fp_p256.mojo`, the curve arithmetic and the op walk `ecurve.mojo`."""

from workloads.bigint import Big
from workloads.fp_p256 import FpP256
from core.params import Params
from workloads.ecurve import Curve, Point, Walk, window_points, check_inputs, public_bytes, slice32, INPUT
from workloads.mulmod import Op, VALUE, CURVE_P256, mulmod_statement, circuit_values, circuit_trace, circuit_public_data
from relations.statement import Statement, Layout
from workload import Workload

comptime P256 = Curve[FpP256]
comptime WINDOWS = 43       # ceil(256 / QW): the scalars are below n < 2^256


def secp256r1() raises -> P256:
    var p = Big.from_hex("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff")
    return P256(p.copy(),
                Big.from_hex("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551"),
                p - Big(3),
                Big.from_hex("5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b"),
                Point(Big.from_hex("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"),
                      Big.from_hex("4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5"), False))


def walk(var c: P256, r: Big, s: Big, e: Big, q: Point, live: Bool) raises -> Walk[FpP256]:
    """The fixed circuit; with `live`, the verifier's checks on (r, s, e, Q), then the public factor values
    and hints of that signature. Window w adds the one public point P_w = d_w Q' + d'_w G (Straus-Shamir
    over u2 on Q and u1 on G), built from two small tables."""
    var w = Walk[FpP256](c^, live)
    var pts = List[Point]()
    for _ in range(WINDOWS):
        pts.append(Point.identity())
    var b16 = Point.identity()
    var b = Point.identity()
    if live:
        check_inputs(w.c, r, s, e, q)
        var inv = s.inv_mod(w.c.n)
        var ks: List[Big] = [r.mulmod(inv, w.c.n), e.mulmod(inv, w.c.n)]
        var bases: List[Point] = [q.copy(), w.c.g.copy()]
        pts = window_points(w.c, bases, ks, WINDOWS)
        b = w.c.blinding(public_bytes(r, s, e, q))
        b16 = b.copy()
        for _ in range(4):
            b16 = w.c.double(b16)
    w.schedule(pts, b, b16, r)
    return w^


struct EcdsaP256(Workload, Movable):
    """One secp256r1 signature on the mulmod chains: public inputs (r, s, e, Q), the witness the slopes of
    the fixed addition chain."""
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
        var w = walk(secp256r1(), Big(), Big(), Big(), Point.identity(), False)
        return w.ops.copy()

    def statement[p: Params](self) raises -> Statement:
        return mulmod_statement(True, EcdsaP256.circuit(), pin=False, curve=CURVE_P256)

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        var w = walk(secp256r1(), self.r, self.s, self.e, self.q, True)
        return circuit_trace[p](layout, circuit_values(w.inputs, w.ops, w.hints, CURVE_P256), w.ops, curve=CURVE_P256)

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        return public_bytes(self.r, self.s, self.e, self.q)

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        if len(public_inputs) != INPUT:
            raise Error("public inputs are r, s, e, x_Q, y_Q")
        var w = walk(secp256r1(), slice32(public_inputs, 0), slice32(public_inputs, 32), slice32(public_inputs, 64),
                     Point(slice32(public_inputs, 96), slice32(public_inputs, 128), False), True)
        var values = List[UInt8](capacity=len(w.inputs) * VALUE)
        for v in w.inputs:
            values.extend(v.copy())
        return circuit_public_data[p](w.ops, values, 0, CURVE_P256)
