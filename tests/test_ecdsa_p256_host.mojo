"""P-256 on the host: the Montgomery field against Big, the group (G on the curve, n G = O, a = -3
doublings), the RFC 6979 A.2.5 signature verifying and a changed message not, the blinding point, then the
circuit shape: op counts, the chain count against the 144 x 1152 grid, the live walk emitting the fixed
circuit op for op, and the circuit satisfied over the P-256 moduli by the walk's values and hints. The prover round trip waits on the P-256 product reduction (docs/ecdsa.md section 8)."""

from std.testing import assert_equal, assert_true, assert_false, assert_raises, TestSuite

from core.params import CLIENT
from workloads.bigint import Big
from workloads.fp_p256 import FpP256, ONE
from workloads.ecurve import Point, recode, skew, QW
from workloads.ecdsa_p256 import secp256r1, EcdsaP256, walk, WINDOWS
from workloads.mulmod import MUL, CURVE_P256, chain_count, circuit_values

comptime p = CLIENT.grid(144, 1152)


def _rfc6979_sample() raises -> Tuple[Big, Big, Big, Point]:
    """RFC 6979 A.2.5, P-256 with SHA-256, message "sample"."""
    var q = Point(Big.from_hex("60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6"),
                  Big.from_hex("7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299"), False)
    var r = Big.from_hex("EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716")
    var s = Big.from_hex("F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8")
    var e = Big.from_hex("AF2BDBE1AA9B6EC1E2ADE1D694F41FC71A831D0268E9891562113D8A62ADD1BF")
    return (r^, s^, e^, q^)


def test_fp_matches_big() raises:
    """The Montgomery field against Big's modular arithmetic: products, sums, differences, inverses, on
    values across the range; the constants R mod p and R^2 mod p."""
    var c = secp256r1()
    assert_equal(FpP256.from_big(Big(1)).l, ONE)
    var vals: List[Big] = [Big(0), Big(1), Big(2), c.p - Big(1), c.p - Big(2), Big(1).shl(255), Big(1).shl(256) - Big(1) - c.p]
    var seed = Big(0x9E3779B97F4A7C15)
    for _ in range(12):
        seed = (seed * Big(6364136223846793005) + Big(1442695040888963407)).low(256).mod(c.p)
        vals.append(seed.copy())
    for a in vals:
        assert_equal(FpP256.from_big(a).to_big(), a)
        for b in vals:
            assert_equal((FpP256.from_big(a) * FpP256.from_big(b)).to_big(), a.mulmod(b, c.p))
            assert_equal((FpP256.from_big(a) + FpP256.from_big(b)).to_big(), (a + b).mod(c.p))
            assert_equal((FpP256.from_big(a) - FpP256.from_big(b)).to_big(), (a - b + c.p).mod(c.p))
        if not a.is_zero():
            assert_equal(FpP256.from_big(a).inv().to_big(), a.inv_mod(c.p))
    assert_true(FpP256.from_big(Big(0)).inv().to_big().is_zero())


def test_group() raises:
    """G on the curve, n G = O, doubling and addition agree, a = p - 3 enters the tangent."""
    var c = secp256r1()
    assert_equal(c.a, c.p - Big(3))
    assert_true(c.on_curve(c.g))
    assert_true(c.mul(c.g, c.n).inf)
    var two = c.double(c.g)
    assert_true(c.on_curve(two))
    assert_equal(c.add(c.mul(c.g, Big(5)), c.mul(c.g, Big(7))), c.mul(c.g, Big(12)))
    assert_true(c.add(two, c.neg(two)).inf)
    assert_equal(c.add(c.g, Point.identity()), c.g)
    assert_false(c.on_curve(Point(c.g.x.copy(), c.g.y + Big(1), False)))


def test_verify_and_blinding() raises:
    """The RFC 6979 vector verifies, a changed message or a bad s does not; the blinding point is on the
    curve, even, and depends on the message."""
    var c = secp256r1()
    var sig = _rfc6979_sample()
    assert_true(c.verify(sig[0], sig[1], sig[2], sig[3]))
    assert_false(c.verify(sig[0], sig[1], sig[2] + Big(1), sig[3]))
    assert_false(c.verify(sig[0], c.n, sig[2], sig[3]))
    assert_false(c.verify(sig[0], sig[1], sig[2], Point.identity()))
    var msg: List[UInt8] = [1, 2, 3]
    var b = c.blinding(msg)
    assert_true(c.on_curve(b))
    assert_equal(b.y.bit(0), 0)
    msg[0] = 9
    assert_true(c.blinding(msg) != b)


def test_recode_43_windows() raises:
    """Full scalars below n recoded in 43 windows of 6 bits: the digits sum back, the last digit and the
    window-0 digit with its correction stay inside the 67-entry table of digit_points."""
    var c = secp256r1()
    var ks: List[Big] = [Big(0), Big(1), Big(2), c.n - Big(1), c.n - Big(2), Big(1).shl(255)]
    var seed = Big(0x9E3779B97F4A7C15)
    for _ in range(20):
        seed = (seed * Big(6364136223846793005) + Big(1442695040888963407)).low(256).mod(c.n)
        ks.append(seed.copy())
    for k in ks:
        var s = skew(k)
        var d = recode(s[0], QW, WINDOWS)
        var acc = Big()
        for i in range(WINDOWS - 1, -1, -1):
            acc = acc.shl(QW) + Big(d[i])
            if i < WINDOWS - 1:
                assert_true(d[i] % 2 != 0 and d[i] >= -(1 << QW) and d[i] < (1 << QW))
        assert_true(abs(d[WINDOWS - 1]) <= (1 << QW) and abs(d[0] + s[1]) <= (1 << QW) + 2)
        assert_equal(acc + Big(s[1]), k)


def test_circuit_shape_and_host_walk() raises:
    """252 doublings of 4 MUL and 7 add-lane ops, 43 additions of 3 and 8, the x-only closing addition
    and the two closing ops; the chains fit 144 x 1152; the live walk emits the fixed circuit; its values
    satisfy every op over P-256's p and n, and a changed message does not."""
    var ops = EcdsaP256.circuit()
    var muls = 0
    for op in ops:
        if op.kind == MUL:
            muls += 1
    var doublings = (WINDOWS - 1) * 6
    assert_equal(muls, 4 * doublings + 3 * WINDOWS + 2)
    assert_equal(len(ops) - muls, 7 * doublings + 8 * WINDOWS + 6 + 2)
    assert_true(chain_count(ops) <= p.h2())
    print("ops:", len(ops), "muls:", muls, "chains:", chain_count(ops), "of", p.h2())
    var sig = _rfc6979_sample()
    var w = walk(secp256r1(), sig[0], sig[1], sig[2], sig[3], True)
    assert_equal(len(w.ops), len(ops))
    for j in range(len(ops)):
        var a = ops[j]
        var b = w.ops[j]
        assert_true(a.kind == b.kind and a.x == b.x and a.y == b.y and a.z == b.z and a.s == b.s and a.sy == b.sy and a.sz == b.sz and a.qz == b.qz and a.mod == b.mod)
    print("public factors:", len(w.inputs), "hints:", len(w.hints))
    _ = circuit_values(w.inputs, w.ops, w.hints, CURVE_P256)
    var bad = walk(secp256r1(), sig[0], sig[1], sig[2] + Big(1), sig[3], True)
    with assert_raises(contains="does not hold"):
        _ = circuit_values(bad.inputs, bad.ops, bad.hints, CURVE_P256)
    with assert_raises(contains="(0, n)"):
        _ = walk(secp256r1(), sig[0], Big(), sig[2], sig[3], True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
