"""ECDSA on secp256r1 (P-256; docs/ecdsa.md section 8): the curve constants (a = -3, no endomorphism), the
walk of section 2 over the two bases Q and G with the full 256-bit scalars (43 windows of 6 bits, 252
doublings, 44 additions) and the `EcdsaP256` circuit. The host side is complete: the field is
`fp_p256.mojo`, the curve arithmetic and the op walk `ecurve.mojo`.

Not a `Workload` yet: the MUL lane of `mulmod.mojo` reduces products by secp256k1's fold
(2^256 = 2^32 + 977), which does not hold mod this prime, so `mulmod_statement` cannot compile this
circuit. The reduction it needs is the open item of docs/ecdsa.md section 8."""

from workloads.bigint import Big
from workloads.fp_p256 import FpP256
from workloads.ecurve import Curve, Point, Walk, window_points, check_inputs, public_bytes
from workloads.mulmod import Op

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


struct EcdsaP256(Movable):
    """One secp256r1 signature: public inputs (r, s, e, Q), the witness the slopes of the fixed addition
    chain. `circuit` and `public_inputs` as for `EcdsaK1`; `statement`, `trace` and `public_data` wait on
    the P-256 product reduction (the module docstring)."""
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

    def public_inputs(self) raises -> List[UInt8]:
        return public_bytes(self.r, self.s, self.e, self.q)
