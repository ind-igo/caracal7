from std.testing import assert_equal, assert_true, assert_false, assert_raises, TestSuite

from caracal7.relations.bigint import Big
from caracal7.relations.ecdsa import Curve, Point, recode, skew


def test_group_and_endomorphism() raises:
    """G on the curve, n G = O, phi(G) = lambda G, the lattice basis kills lambda."""
    var c = Curve()
    assert_true(c.on_curve(c.g))
    assert_true(c.mul(c.g, c.n).inf)
    assert_equal(c.phi(c.g), c.mul(c.g, c.lam))
    assert_true((c.a1 + c.b1 * c.lam).mod(c.n).is_zero())
    assert_true((c.a2 + c.b2 * c.lam).mod(c.n).is_zero())
    var two = c.double(c.g)
    assert_equal(c.add(c.g, c.g), two)
    assert_true(c.add(two, c.neg(two)).inf)
    assert_equal(c.add(c.g, Point.identity()), c.g)


def test_split_and_recode() raises:
    """Split: k1 + k2 lambda = k mod n with halves below 2^128; recoded digits sum back for both window sizes."""
    var c = Curve()
    var bound = Big(1).shl(128)
    var ks: List[Big] = [Big(0), Big(1), Big(2), c.n - Big(1), Big.from_hex("deadbeefcafebabe0123456789abcdef0011223344556677f00dfacefeedc0de")]
    for k in ks:
        var kk = c.split(k)
        assert_equal((kk[0] + kk[1] * c.lam).mod(c.n), k.mod(c.n))
        assert_true(kk[0].abs() < bound and kk[1].abs() < bound)
        for half in [kk[0].copy(), kk[1].copy()]:
            var s = skew(half.abs())
            for b in [4, 8]:
                var d = recode(s[0], b)
                var acc = Big()
                for i in range(31, -1, -1):
                    acc = acc.shl(b) + Big(d[i])
                    if i < 31:
                        assert_true(d[i] % 2 != 0 and d[i] >= -(1 << b) and d[i] < (1 << b))
                assert_equal(acc + Big(s[1]), half.abs())
    var d = recode(Big(-1), 4)
    assert_equal(d[0], 15)
    assert_equal(d[31], -1)
    with assert_raises(contains="odd"):
        _ = recode(Big(4), 4)


def test_blinding_and_verify() raises:
    """A signature from a known key verifies, a changed message does not, the blinding point is on the curve
    and depends on the message."""
    var c = Curve()
    var d = Big.from_hex("1e99423a4ed27608a15a2616a2b0e9e52ced330ac530edcc32c8ffc6a526aedd")
    var k = Big.from_hex("a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90")
    var e = Big.from_hex("4b688df40bcedbe641ddb16ff0a1842d9c67ea1c3bf63f3e0471baa664531d1a")
    var q = c.mul(c.g, d)
    var r = c.mul(c.g, k).x.mod(c.n)
    var s = k.inv_mod(c.n).mulmod(e + r.mulmod(d, c.n), c.n)
    assert_true(c.verify(r, s, e, q))
    assert_false(c.verify(r, s, e + Big(1), q))
    assert_false(c.verify(r, c.n, e, q))
    assert_false(c.verify(r, s, e, Point.identity()))
    var msg: List[UInt8] = [1, 2, 3]
    var b = c.blinding(msg)
    assert_true(c.on_curve(b))
    assert_equal(b.y.bit(0), 0)
    msg[0] = 9
    assert_true(c.blinding(msg) != b)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
